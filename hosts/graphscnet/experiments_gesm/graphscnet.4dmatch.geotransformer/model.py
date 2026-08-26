import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.layers import ConvBlock, NonRigidICP
from vision3d.ops import (
    apply_deformation,
    build_euclidean_deformation_graph,
    index_select,
    pairwise_distance,
)

# isort: split
from graphsc import GraphSCModule


from collections import OrderedDict
import numpy as np

from torch import Tensor
from typing import Optional, Tuple, Union

def apply_deformation_jacobian(
    points: Tensor,
    nodes: Tensor,
    jacobians: Tensor,          # 新增：(M, 3, 3)
    displacements: Tensor,      # 新增：(M, 3)  对应节点的位移
    anchor_indices: Tensor,
    anchor_weights: Tensor,
    eps: float = 1e-6):

    anchor_weights = anchor_weights / (anchor_weights.sum(dim=1, keepdim=True) + eps)
    anchor_masks = torch.ne(anchor_indices, -1)

    p_indices, col_indices = torch.nonzero(anchor_masks, as_tuple=True)
    n_indices = anchor_indices[p_indices, col_indices]
    weights = anchor_weights[p_indices, col_indices]

    sel_points = points[p_indices]                    # (C, 3)
    sel_nodes = nodes[n_indices]                      # (C, 3)
    sel_jacobians = jacobians[n_indices]              # (C, 3, 3)
    sel_displacements = displacements[n_indices]      # (C, 3)

    # 核心修改：用 Jacobian 代替旋转矩阵
    # warped = J @ (p - node) + node + disp
    delta = sel_points - sel_nodes                    # (C, 3)
    # 使用 einsum 进行批量矩阵乘法

    I = torch.eye(3, device=sel_jacobians.device, dtype=sel_jacobians.dtype).unsqueeze(0)
    F = I + sel_jacobians
    transformed_delta = torch.einsum('cij,cj->ci', F, delta)  # (C, 3)
    sel_warped_points = transformed_delta + sel_nodes + sel_displacements  # (C, 3)

    # 加权累加
    sel_warped_points = sel_warped_points * weights.unsqueeze(1)
    warped_points = torch.zeros_like(points)
    p_indices_expanded = p_indices.unsqueeze(1).expand_as(sel_warped_points)
    warped_points.scatter_add_(dim=0, index=p_indices_expanded, src=sel_warped_points)

    return warped_points


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class SirenVW(nn.Module):
    def __init__(self,
                 in_features=3,           # 输入坐标维度（3D点通常为3，可设为4带时间）
                 hidden_features=64, #128
                 hidden_layers=1, #3
                 first_omega_0=30,
                 hidden_omega_0=30.):
        super().__init__()

        # ================== SIREN 主干网络 ==================
        net = []
        # 第一层
        net.append(SineLayer(in_features, hidden_features,
                             is_first=True, omega_0=first_omega_0))

        # 隐藏层
        for _ in range(hidden_layers):
            net.append(SineLayer(hidden_features, hidden_features,
                                 is_first=False, omega_0=hidden_omega_0))

        self.backbone = nn.Sequential(*net)

        # ================== 最终输出层 ==================
        # 输出7维原始值：前3维速度场 + 后4维未归一化的权重
        self.final_linear = nn.Linear(hidden_features, 10)

        # SIREN风格初始化
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                               np.sqrt(6 / hidden_features) / hidden_omega_0)
            nn.init.zeros_(self.final_linear.bias)

    @staticmethod
    def skew_symmetric(omega):
        """
        将 (..., 3) 的向量转换为 (..., 3, 3) 的反对称矩阵
        """
        wx, wy, wz = omega.unbind(dim=-1)
        o = torch.zeros_like(wx)

        row0 = torch.stack([ o, -wz,  wy], dim=-1)
        row1 = torch.stack([ wz,  o, -wx], dim=-1)
        row2 = torch.stack([-wy,  wx,  o], dim=-1)

        return torch.stack([row0, row1, row2], dim=-2)

    @staticmethod
    def so3_to_rotation_matrix(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Rodrigues 公式：将 so(3) 向量 ω 映射为 SO(3) 旋转矩阵
        omega: (N, 3)
        return: (N, 3, 3)
        """
        theta = torch.norm(omega, dim=-1, keepdim=True)          # (N, 1)
        theta = torch.clamp(theta, min=eps)

        # 单位轴
        k = omega / theta                                        # (N, 3)

        # 构造反对称矩阵 K = [k]_×
        K = SirenVW.skew_symmetric(k)                            # (N, 3, 3)

        # Rodrigues 公式
        # R = I + sinθ K + (1 - cosθ) K²
        I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0)
        sin_theta = torch.sin(theta).unsqueeze(-1)               # (N, 1, 1)
        cos_theta = torch.cos(theta).unsqueeze(-1)

        R = I + sin_theta * K + (1.0 - cos_theta) * torch.bmm(K, K)
        return R

    def forward(self, coords):
        """
        输入:  coords shape = (N, in_features)
        输出:  两个张量（分开返回）
               velocity: (N, 3)   → 速度场
               weights:  (N, 4)   → 已做 softmax 归一化的权重（每行和为1）
        """
        # 通过SIREN主干网络
        features = self.backbone(coords)

        # 最终线性层得到7维
        x = self.final_linear(features)

        # 分离前3维和后4维
        velocity = x[:, :3]
        raw_weights = x[:, 3:7]
        jacobian_flat = x[:, 7:]

        # 对后4维做 softmax 归一化
        weights = torch.softmax(raw_weights, dim=-1)  # (N, 4)
        jacobian = self.skew_symmetric(jacobian_flat)

        #R = self.so3_to_rotation_matrix(jacobian_flat)   # (N, 3, 3) 真正的旋转矩阵
        #I = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0)
        #jacobian = R - I

        return velocity, weights, jacobian

    # 可选：保留用于调试/可视化的中间激活函数
    def forward_with_activations(self, coords, retain_grad=False):
        activations = OrderedDict()
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x

        for layer in self.backbone:
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                activations[f'SineLayer_{activation_count}'] = intermed
                activation_count += 1
            else:
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
            activations[f'Layer_{activation_count}'] = x
            activation_count += 1

        return activations


class NeuralGESM_PC(nn.Module):
    """
    用你现有的 SirenVW 实现 Neural GESM-PC
    完全替换原来的 NonRigidICP
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_normal = cfg.model.gesm.use_normal

        in_dim = 3   # 可扩展支持 normals

        # 直接使用你提供的 SirenVW
        self.siren = SirenVW(
            in_features=in_dim,
            hidden_features=128,
            hidden_layers=2,
            first_omega_0=30,
            hidden_omega_0=30.
        )

        # GESM-PC 的 4 个能量项权重（可学习全局系数，或直接用 SirenVW 输出的 point-wise weights）
        self.register_buffer('global_weights', torch.tensor([1.0, 1.0, 1.0, 1.0]))  # a1,b1,c1,a2

    def forward(self, coords, normals=None):
        """
        coords: (N, 3) 或 (N, 6) 如果带 normals
        返回:
            warped: (N, 3)
            velocity: (N, 3)
            adapt_weights: (N, 4)   ← SirenVW 已经 softmax 好的 point-wise weights
        """

        input_feat = coords

        velocity, adapt_weights, jacobian = self.siren(input_feat)   # 直接调用你写的 SirenVW

        warped = coords + velocity
        return warped, velocity, adapt_weights, jacobian


class GraphSCNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.gesm_net = NeuralGESM_PC(cfg).cuda()

        self.max_local_correspondences = cfg.model.max_local_correspondences
        self.min_local_correspondences = cfg.model.min_local_correspondences
        self.num_anchors = cfg.model.deformation_graph.num_anchors
        self.node_coverage = cfg.model.deformation_graph.node_coverage

        self.acceptance_score = cfg.eval.acceptance_score

        self.encoder = GraphSCModule(
            cfg.model.transformer.input_dim,
            cfg.model.transformer.output_dim,
            cfg.model.transformer.hidden_dim,
            cfg.model.transformer.num_heads,
            cfg.model.transformer.num_blocks,
            cfg.model.transformer.num_layers_per_block,
            cfg.model.transformer.sigma_d,
            embedding_k=cfg.model.transformer.embedding_k,
            embedding_dim=cfg.model.transformer.embedding_dim,
            dropout=cfg.model.transformer.dropout,
            act_cfg=cfg.model.transformer.activation_fn,
        )

        self.classifier = nn.Sequential(
            ConvBlock(
                in_channels=cfg.model.classifier.input_dim,
                out_channels=cfg.model.classifier.input_dim // 2,
                kernel_size=1,
                conv_cfg="Conv1d",
                norm_cfg="GroupNorm",
                act_cfg="LeakyReLU",
                dropout=cfg.model.classifier.dropout,
            ),
            ConvBlock(
                in_channels=cfg.model.classifier.input_dim // 2,
                out_channels=cfg.model.classifier.input_dim // 4,
                kernel_size=1,
                conv_cfg="Conv1d",
                norm_cfg="GroupNorm",
                act_cfg="LeakyReLU",
                dropout=cfg.model.classifier.dropout,
            ),
            ConvBlock(
                in_channels=cfg.model.classifier.input_dim // 4,
                out_channels=1,
                kernel_size=1,
                conv_cfg="Conv1d",
                norm_cfg="None",
                act_cfg="None",
            ),
        )

        self.sigma_d = cfg.model.transformer.sigma_d
        self.sigma_f = nn.Parameter(torch.as_tensor(1.0))

        #self.registration = NonRigidICP(
        #    corr_lambda=cfg.model.nicp.corr_lambda,
        #    arap_lambda=cfg.model.nicp.arap_lambda,
        #    lm_lambda=cfg.model.nicp.lm_lambda,
        #    num_iterations=cfg.model.nicp.num_iterations,
        #)

    def forward(self, data_dict):
        output_dict = {}

        # 1. unpack data
        src_points = data_dict["src_points"]  # (Ns, 3)
        tgt_points = data_dict["tgt_points"]  # (Nt, 3)

        src_corr_points = data_dict["src_corr_points"]  # (C,)
        tgt_corr_points = data_dict["tgt_corr_points"]  # (C,)
        num_correspondences = src_corr_points.shape[0]

        node_indices = data_dict["node_indices"]  # (M,)
        src_nodes = src_points[node_indices]  # (M, 3)
        num_nodes = src_nodes.shape[0]

        output_dict["src_points"] = src_points
        output_dict["tgt_points"] = tgt_points
        output_dict["src_nodes"] = src_nodes

        # 2. build deformation graph
        #print("src_corr shape: {}".format(src_corr_points.shape))
        #print("src_node shape: {}".format(src_nodes.shape))
        #print("src_points shape: {}".format(src_points.shape))
        corr_anchor_indices, corr_anchor_weights = build_euclidean_deformation_graph(
            src_corr_points,
            src_nodes,
            self.num_anchors,
            self.node_coverage,
            return_node_graph=False,
        )  # (C, Ka)

        # 2. compute node-to-correspondence weights
        anchor_masks = torch.ne(corr_anchor_indices, -1)  # (C, Ka)
        if anchor_masks.dim() == 3:
            anchor_masks = anchor_masks.squeeze(0)

        anchor_corr_indices, anchor_col_indices = torch.nonzero(
            anchor_masks, as_tuple=True
        )  # (S,), (S,)

        if corr_anchor_indices.dim() == 3:
            corr_anchor_indices = corr_anchor_indices.squeeze(0)

        anchor_node_indices = corr_anchor_indices[
            anchor_corr_indices, anchor_col_indices
        ]  # (S,)

        if corr_anchor_weights.dim() == 3:
            corr_anchor_weights = corr_anchor_weights.squeeze(0)

        anchor_weights = corr_anchor_weights[
            anchor_corr_indices, anchor_col_indices
        ]  # (S,)
        node_to_corr_weights = torch.zeros(
            size=(num_nodes, num_correspondences)
        ).cuda()  # (M, C)
        node_to_corr_weights[
            anchor_node_indices, anchor_corr_indices
        ] = anchor_weights  # (M, C)

        # 3. assign correspondences to nodes
        max_local_correspondences = (
            torch.gt(node_to_corr_weights, 0.0).sum(dim=1).max().item()
        )
        max_local_correspondences = min(
            max_local_correspondences, self.max_local_correspondences
        )
        local_corr_weights, local_corr_indices = node_to_corr_weights.topk(
            k=max_local_correspondences, dim=1, largest=True
        )  # (M, k), (M, k)
        local_corr_masks = torch.gt(local_corr_weights, 0.0)  # (M, k)

        # 4. remove small nodes
        local_corr_counts = local_corr_masks.sum(dim=-1)  # (M,)
        node_masks = torch.gt(local_corr_counts, self.min_local_correspondences)  # (M,)
        local_corr_indices = local_corr_indices[node_masks]  # (M', k)
        local_corr_weights = local_corr_weights[node_masks]  # (M', k)
        local_corr_masks = local_corr_masks[node_masks]  # (M', k)

        output_dict["local_corr_indices"] = local_corr_indices
        output_dict["local_corr_weights"] = local_corr_weights
        output_dict["local_corr_masks"] = local_corr_masks

        # 5. transformer encoder
        corr_feats, corr_masks = self.encoder(
            src_corr_points,
            tgt_corr_points,
            local_corr_indices,
            local_corr_weights,
            local_corr_masks,
        )  # (C, d) (C,)

        corr_feats_norm = F.normalize(corr_feats, p=2, dim=1)  # (C, d)
        output_dict["corr_feats"] = corr_feats_norm
        output_dict["sigma_f"] = self.sigma_f

        # 6. classifier
        corr_feats = corr_feats.transpose(0, 1).unsqueeze(0)  # (1, d, C)
        corr_logits = self.classifier(corr_feats)
        corr_logits = corr_logits.flatten()  # (C,)
        corr_scores = torch.sigmoid(corr_logits)

        # oracle
        # corr_scores = data_dict["corr_labels"].float()
        # all
        # corr_scores = torch.ones_like(corr_scores)

        output_dict["corr_logits"] = corr_logits
        output_dict["corr_scores"] = corr_scores
        output_dict["corr_masks"] = corr_masks

        # 8. feature consistency
        local_corr_feats_norm = index_select(
            corr_feats_norm, local_corr_indices, dim=0
        )  # (M', k, d)
        local_affinity_mat = pairwise_distance(
            local_corr_feats_norm, local_corr_feats_norm, normalized=True, squared=False
        )  # (M', k, k)
        local_fc_mat = torch.relu(1.0 - local_affinity_mat.pow(2) / self.sigma_f.pow(2))

        output_dict["feature_consistency"] = local_fc_mat
        if data_dict.get("registration", False):
            (
                anchor_indices,
                anchor_weights,
                edges_indices,
                edge_weights,
            ) = build_euclidean_deformation_graph(
                src_points, src_nodes, self.num_anchors, self.node_coverage
            )
            edge_weights = torch.ones_like(
                edge_weights
            )  # use the same weights for all edges
            #print(anchor_indices.shape)

            output_dict["anchor_indices"] = anchor_indices
            output_dict["anchor_weights"] = anchor_weights
            output_dict["edges_indices"] = edges_indices
            output_dict["edges_weights"] = edge_weights

            output_dict["node_normals"] = data_dict["node_normals"]
            output_dict["node_L"] = data_dict["node_L"]
            output_dict["node_M"] = data_dict["node_M"]

            # 0/1 weighting
            # corr_logits = torch.sigmoid(corr_logits)
            # corr_logits = torch.gt(corr_logits, self.acceptance_score).float()
            # corr_anchor_weights = corr_anchor_weights * corr_logits.unsqueeze(1)
            # corr_masks = torch.gt(corr_anchor_weights.sum(1), 0)
            # src_corr_points = src_corr_points[corr_masks]
            # tgt_corr_points = tgt_corr_points[corr_masks]
            # corr_anchor_indices = corr_anchor_indices[corr_masks]
            # corr_anchor_weights = corr_anchor_weights[corr_masks]

            # dynamic weighting
            corr_masks = torch.gt(corr_scores, self.acceptance_score)
            src_corr_points = src_corr_points[corr_masks]
            tgt_corr_points = tgt_corr_points[corr_masks]
            corr_anchor_indices = corr_anchor_indices[corr_masks]
            corr_anchor_weights = corr_anchor_weights[corr_masks]
            corr_scores = corr_scores[corr_masks]

            if "src_normals" in data_dict:
                src_normals = data_dict["src_normals"]
                if not torch.is_tensor(src_normals):
                    src_normals = torch.from_numpy(src_normals).float().cuda()
                else:
                    src_normals = src_normals.float().cuda()
                output_dict["src_normals"] = src_normals   # 存入 output_dict

            if "src_L" in data_dict:
                src_L = data_dict["src_L"]
                if not torch.is_tensor(src_L):
                    src_L = torch.from_numpy(src_L).float().cuda()
                else:
                    src_L = src_L.float().cuda()
                output_dict["src_corr_L"] = src_L

            if "src_M" in data_dict:
                src_M = data_dict["src_M"]
                if not torch.is_tensor(src_M):
                    src_M = torch.from_numpy(src_M).float().cuda()
                else:
                    src_M = src_M.float().cuda()
                output_dict["src_M"] = src_M

            if "src_corr_normals" in data_dict:
                src_corr_normals = data_dict["src_corr_normals"]
                if not torch.is_tensor(src_corr_normals):
                    src_corr_normals = torch.from_numpy(src_corr_normals).float().cuda()
                else:
                    src_corr_normals = src_corr_normals.float().cuda()
                src_corr_normals = src_corr_normals[corr_masks]
                output_dict["src_corr_normals"] = src_corr_normals   # 存入 output_dict

            if "src_corr_L" in data_dict:
                src_corr_L = data_dict["src_corr_L"]
                if not torch.is_tensor(src_corr_L):
                    src_corr_L = torch.from_numpy(src_corr_L).float().cuda()
                else:
                    src_corr_L = src_corr_L.float().cuda()
                src_corr_L = src_corr_L[corr_masks]
                output_dict["src_corr_L"] = src_corr_L

            if "src_corr_M" in data_dict:
                src_corr_M = data_dict["src_corr_M"]
                if not torch.is_tensor(src_corr_M):
                    src_corr_M = torch.from_numpy(src_corr_M).float().cuda()
                else:
                    src_corr_M = src_corr_M.float().cuda()
                src_corr_M = src_corr_M[corr_masks]
                output_dict["src_corr_M"] = src_corr_M

            (
                corr_anchor_indices,
                corr_anchor_weights,
                corr_edges_indices,
                corr_edge_weights,
            ) = build_euclidean_deformation_graph(src_corr_points, src_nodes, self.num_anchors, self.node_coverage)

            output_dict["corr_anchor_indices"] = corr_anchor_indices
            output_dict["corr_anchor_weights"] = corr_anchor_weights
            output_dict["corr_edges_indices"] = corr_edges_indices
            corr_edge_weights = torch.ones_like(corr_edge_weights)
            output_dict["corr_edge_weights"] = corr_edge_weights

            #transforms = self.registration(
            #    src_nodes,
            #    src_corr_points,
            #    tgt_corr_points,
            #    corr_anchor_indices,
            #    corr_anchor_weights,
            #    edges_indices,
                #corr_weights=corr_weights,
                #edge_weights=edge_weights,
            #)
            #warped_src_corr, disp, adapt_weights, jacobian = self.gesm_net(src_points)
            warped_src_nodes, disp, adapt_weights, jacobian = self.gesm_net(src_nodes)

            output_dict["warped_src_nodes"] = warped_src_nodes
            output_dict["displacements"] = disp
            output_dict["jacobian"] = jacobian

            warped_src_corr = apply_deformation_jacobian(src_corr_points, src_nodes, jacobian, disp, corr_anchor_indices, corr_anchor_weights)


            output_dict["src_corr_points"] = src_corr_points
            output_dict["tgt_corr_points"] = tgt_corr_points
            #output_dict["warped_src_points"] = warped_src
            output_dict["adapt_weights"] = adapt_weights

            output_dict["embedded_deformation_nodes"] = src_nodes

            output_dict["warped_src_corr"] = warped_src_corr

            output_dict["tgt_center"] = data_dict["tgt_center"]
            output_dict["tgt_scale"] = data_dict["tgt_scale"]

            with torch.no_grad():
                warped_src_points = apply_deformation_jacobian(src_points, src_nodes, jacobian, disp, anchor_indices, anchor_weights)
                warped_src_points = warped_src_points * output_dict["tgt_scale"] + output_dict["tgt_center"]
                output_dict["warped_src_points"] = warped_src_points
            #output_dict["embedded_deformation_transforms"] = transforms

            #warped_src_points = apply_deformation(
            #    src_points, src_nodes, transforms, anchor_indices, anchor_weights
            #)
            #output_dict["warped_src_points"] = warped_src_points


        return output_dict


def create_model(cfg):
    model = GraphSCNet(cfg)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == "__main__":
    main()
