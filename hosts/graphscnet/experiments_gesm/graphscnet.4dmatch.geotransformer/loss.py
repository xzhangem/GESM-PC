import ipdb
import torch
import torch.nn as nn

from vision3d.loss import SigmoidFocalLossWithLogits
from vision3d.ops import apply_deformation, apply_transform
from vision3d.ops.metrics import (
    compute_nonrigid_feature_matching_recall,
    compute_scene_flow_accuracy,
    compute_scene_flow_outlier_ratio,
    evaluate_binary_classification,
)

import torch

from torch import Tensor
from typing import Optional, Tuple, Union
import torch.nn.functional as F

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
    # 使用 einsum 进行批量矩阵乘
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

def arap_edge_regularization_jacobian(
    nodes: torch.Tensor,           # (M, 3)  deformation graph 节点
    jacobians: torch.Tensor,       # (M, 3, 3)  每个节点的 Jacobian 矩阵
    displacements: torch.Tensor,   # (M, 3)     每个节点的位移 (velocity)
    edge_indices: torch.Tensor,    # (E, 2)     边索引，每行 [u, v]
    edge_weights: torch.Tensor = None,  # (E,) 可选
    reduction: str = 'mean'):
    """
    基于 Jacobian 的 Deformation Graph Edge Regularization
    把原 ARAP 中的 R_u 替换为 GESM-PC 预测的 Jacobian

    Args:
        nodes:          (M, 3)
        jacobians:      (M, 3, 3)
        displacements:  (M, 3)
        edge_indices:   (E, 2)
        edge_weights:   (E,) 可选
        reduction:      'mean' 或 'sum'

    Returns:
        loss: 正则化损失
    """
    if edge_indices.numel() == 0:
        return torch.tensor(0.0, device=nodes.device, dtype=nodes.dtype)

    u = edge_indices[:, 0]
    v = edge_indices[:, 1]

    # 取出对应节点的属性
    v_u = nodes[u]                      # (E, 3)
    v_v = nodes[v]                      # (E, 3)
    J_u = jacobians[u]                  # (E, 3, 3)
    d_u = displacements[u]              # (E, 3)
    d_v = displacements[v]              # (E, 3)

    I = torch.eye(3, device=J_u.device, dtype=J_u.dtype).unsqueeze(0)
    F_u = I + J_u

    # 计算: J_u @ (v_v - v_u) + v_u + d_u
    delta = v_v - v_u                                   # (E, 3)
    transformed_delta = torch.bmm(F_u, delta.unsqueeze(-1)).squeeze(-1)  # (E, 3)
    transformed_v = transformed_delta + v_u + d_u       # (E, 3)

    # 目标位置: v_v + d_v
    target_v = v_v + d_v                                # (E, 3)

    # 计算残差
    residual = transformed_v - target_v                 # (E, 3)
    loss_per_edge = (residual ** 2).sum(dim=1)          # (E,)

    # 加权
    if edge_weights is not None:
        loss_per_edge = loss_per_edge * edge_weights

    # 聚合
    if reduction == 'mean':
        return loss_per_edge.mean()
    elif reduction == 'sum':
        return loss_per_edge.sum()
    else:
        return loss_per_edge


def compute_arap_loss(warped_points, orig_points, edges_indices, edge_weights=None):
    """
    经典 As-Rigid-As-Possible (ARAP) Loss
    在 deformation graph 的节点上计算局部刚性
    """
    device = warped_points.device
    N = warped_points.shape[0]

    if edge_weights is None:
        edge_weights = torch.ones(edges_indices.shape[0], device=device)

    # 取出边的两个端点
    i = edges_indices[:, 0]
    j = edges_indices[:, 1]

    # 原始边向量和变形后边向量
    orig_diff = orig_points[j] - orig_points[i]          # (E, 3)
    warped_diff = warped_points[j] - warped_points[i]    # (E, 3)

    # ==================== 1. 估计每个节点的局部旋转（使用协方差法） ====================
    # 这里我们用一个简化但有效的版本：对每个节点收集邻居，计算最佳旋转
    rotations = estimate_local_rotations(warped_points, orig_points, edges_indices)

    # ==================== 2. 计算 ARAP 能量 ====================
    # 对每条边计算刚性偏差
    R_i = rotations[i]                                   # (E, 3, 3)
    rotated_diff = torch.bmm(R_i, orig_diff.unsqueeze(-1)).squeeze(-1)

    diff = warped_diff - rotated_diff
    per_edge_energy = edge_weights * torch.sum(diff ** 2, dim=1)

    arap_loss = per_edge_energy.mean()
    return arap_loss


def estimate_local_rotations(warped_points, orig_points, edges_indices):
    """
    为每个节点估计局部最佳旋转（使用 SVD 方法）
    已修复 in-place 操作问题
    """
    device = warped_points.device
    N = warped_points.shape[0]

    # 构建稀疏邻接关系
    adj = torch.zeros((N, N), device=device)
    adj[edges_indices[:, 0], edges_indices[:, 1]] = 1
    adj[edges_indices[:, 1], edges_indices[:, 0]] = 1

    rotations = torch.eye(3, device=device).unsqueeze(0).repeat(N, 1, 1)

    for i in range(N):
        neighbors = torch.where(adj[i] > 0)[0]
        if len(neighbors) < 2:
            continue

        p = orig_points[neighbors] - orig_points[i]      # (k, 3)
        q = warped_points[neighbors] - warped_points[i]  # (k, 3)

        cov = p.T @ q                                    # (3, 3)
        U, S, Vh = torch.linalg.svd(cov)

        R = Vh.mT @ U.mT

        # 保证是合法旋转矩阵（det(R) = +1），避免 in-place 操作
        if torch.det(R) < 0:
            # 创建新张量而不是原地修改
            Vh_fixed = Vh.clone()
            Vh_fixed[-1, :] = -Vh_fixed[-1, :]
            R = Vh_fixed.mT @ U.mT

        rotations[i] = R

    return rotations


def estimate_jacobian_with_anchors(warped, orig, anchor_indices, anchor_weights):
    """用 anchor 做加权最小二乘估计 Jacobian"""
    delta_p = orig[anchor_indices] - orig.unsqueeze(1)
    delta_v = warped[anchor_indices] - warped.unsqueeze(1)
    W2 = torch.abs(anchor_weights.unsqueeze(-1))

    lhs = torch.einsum('nki,nkj->nij', delta_p * W2, delta_p)
    rhs = torch.einsum('nki,nkj->nij', delta_p * W2, delta_v)
    dv = torch.linalg.solve(lhs + 1e-6 * torch.eye(3, device=warped.device), rhs)
    #lhs_inv = torch.linalg.pinv(lhs + 1e-4 * torch.eye(3, device=warped.device))
    #dv = torch.bmm(lhs_inv, rhs)

    return dv

def compute_gesm_pc_loss(
    vertices,                  # (N, 3)  变形后的点（warped）
    displacements,             # (N, 3)  位移场
    normals,                   # (N, 3)  单位法线（必须提供）
    anchor_indices=None,       # (N, k)  可选：用 anchor 定义局部邻域
    anchor_weights=None,       # (N, k)  可选
    node_weights=None,         # (N, 4)  SirenVW 输出的自适应权重 [w_distort, w_stretch, w_bend, w_smooth]
    jacobians=None,            # (N, 3, 3) 可选：如果已缓存可直接传入
    L=None,                    # Laplacian 矩阵（可选）
    M=None,                    # mass 向量（可选）
    method='householder',      # 'householder' 或 'svd'
    eps=1e-6,
    a=1.0, b=1.0, c=1.0, d=0.1
):
    """
    GESM-PN Loss（严格对齐你原来的实现风格 + 支持 anchor 邻域）
    只支持带法线版本
    """
    device = vertices.device
    Np = vertices.shape[0]
    vel = displacements

    if normals is None:
        raise ValueError("normals 必须提供（GESM-PN 版本）")

    # ====================== 1. 构造 N0 和 P0（保留你原来的 Householder/SVD 逻辑） ======================
    n = F.normalize(normals, dim=1)

    if method == 'svd':
        N = n.unsqueeze(-1) @ n.unsqueeze(1)                     # (N, 3, 3)
        U, _, _ = torch.linalg.svd(N, full_matrices=False)
        diag_N0 = torch.diag_embed(torch.tensor([1., 0., 0.], device=device))
        N0 = torch.bmm(U.transpose(-2, -1), diag_N0.unsqueeze(0).expand(Np, -1, -1))
        P0 = torch.eye(3, device=device).unsqueeze(0).expand(Np, -1, -1) - N0
    else:  # householder（推荐，默认）
        e1 = torch.tensor([1., 0., 0.], device=device).unsqueeze(0).expand(Np, 3)
        v = n + torch.sign(n[:, 0:1]) * e1
        v = F.normalize(v, dim=1).unsqueeze(-1)
        H = torch.eye(3, device=device).unsqueeze(0).expand(Np, 3, 3) - 2 * (v @ v.transpose(-2, -1))
        diag_N0 = torch.diag_embed(torch.tensor([1., 0., 0.], device=device)).unsqueeze(0).expand(Np, 3, 3)
        N0 = torch.bmm(H.transpose(-2, -1), diag_N0)
        P0 = torch.eye(3, device=device).unsqueeze(0).expand(Np, 3, 3) - N0

    N0 = n.unsqueeze(-1) @ n.unsqueeze(1)
    P0 = torch.eye(3, device=device).unsqueeze(0).expand(Np, 3, 3) - N0

    # ====================== 2. 获取或计算 Jacobian ======================
    if jacobians is None:
        if anchor_indices is None or anchor_weights is None:
            raise ValueError("当 jacobians=None 时，必须提供 anchor_indices 和 anchor_weights")
        jacobians = estimate_jacobian_with_anchors(
            vertices, vertices - vel, anchor_indices, anchor_weights
        )  # 注意这里 orig = vertices - vel

    vel_gradient = jacobians
    t_vel_gradient = torch.bmm(P0, torch.bmm(vel_gradient, P0) + torch.bmm(vel_gradient.transpose(-1, -2), P0))
    n_vel_gradient = torch.bmm(torch.bmm(N0, vel_gradient), P0) + torch.bmm(torch.bmm(P0, vel_gradient.transpose(-1, -2)), N0)

    # ====================== 3. 一阶项（严格对齐你原来的公式） ======================
    t_D = 0.5 * (t_vel_gradient + t_vel_gradient.transpose(1, 2))
    trace_t_D = torch.diagonal(t_D, dim1=1, dim2=2).sum(dim=1)
    trace_t_D2 = torch.sum(t_D * t_D, dim=(1, 2))
    distortion_term = trace_t_D2 - 0.5 * (trace_t_D ** 2)

    stretch_term = (1.0 / 2.0) * (trace_t_D ** 2)

    n_D = 0.5 * (n_vel_gradient + n_vel_gradient.transpose(1, 2))
    trace_n_D = torch.diagonal(n_D, dim1=1, dim2=2).sum(dim=1)
    trace_n_D2 = torch.sum(n_D * n_D, dim=(1, 2))
    bend_term = trace_n_D2 # - trace_n_D ** 2

    # ====================== 4. 二阶项（Laplace） ======================
    if L is not None:
        laplace_term =  torch.abs(torch.sum(vel * (L @ vel), dim=1))
    else:
        # 如果没有 L，用 anchor 加权平均近似
        laplace_term = torch.zeros(Np, device=device)
    #laplace_term = torch.zeros(Np, device=device)

    # ====================== 5. 动态权重（直接使用 SirenVW 输出的 node_weights） ======================
    if node_weights is not None:
        w_a, w_b, w_c, w_d = node_weights.unbind(dim=-1)
    else:
        w_a = torch.full((Np,), a, device=device)
        w_b = torch.full((Np,), b, device=device)
        w_c = torch.full((Np,), c, device=device)
        w_d = torch.full((Np,), d, device=device)

    # ====================== 6. 加权求和 ======================
    if M is None:
        M = torch.ones(Np, device=device)

    weighted_term = (w_a * distortion_term +
                     w_b * stretch_term +
                     w_c * bend_term + 
                     w_d * laplace_term) * M

    #weighted_term = (distortion_term + stretch_term + bend_term + laplace_term) * M #(distortion_term + stretch_term + bend_term)
    #print("{}, {}, {}".format(torch.mean(distortion_term), torch.mean(stretch_term), torch.mean(bend_term)))

    enr = torch.mean(weighted_term) #/ (Np + eps)
    return enr


class LossFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.focal_loss = SigmoidFocalLossWithLogits(reduction="mean")
        self.f_loss_weight = cfg.loss.focal_loss.weight
        self.c_loss_weight = cfg.loss.consistency_loss.weight

        self.gesm_enabled = getattr(cfg.model.gesm, 'enabled', False)
        self.lambda_gesm = getattr(cfg.model.gesm, 'lambda_gesm', 1.0)
        self.gesm_method = getattr(cfg.model.gesm, 'method', 'householder')
        self.lambda_corr = getattr(cfg.model.gesm, 'lambda_corr', 5.0)
        self.jacb_reg = getattr(cfg.model.gesm, 'jacb_reg', 1.0)

    def forward(self, data_dict, output_dict):

        loss_dict = {}
        # focal loss
        logits = output_dict["corr_logits"]
        labels = data_dict["corr_labels"].float()
        f_loss = self.focal_loss(logits, labels) * self.f_loss_weight
        loss_dict["f_loss"] = f_loss

        # feature consistency loss
        fc_mat = output_dict["feature_consistency"]
        local_corr_indices = output_dict["local_corr_indices"]
        local_corr_masks = output_dict["local_corr_masks"]
        local_corr_labels = labels[local_corr_indices]
        fc_labels = local_corr_labels.unsqueeze(2) * local_corr_labels.unsqueeze(1)
        fc_masks = torch.logical_and(local_corr_masks.unsqueeze(2), local_corr_masks.unsqueeze(1))
        loss_mat = (fc_mat - fc_labels).pow(2)
        c_loss = loss_mat[fc_masks].mean() * self.c_loss_weight
        loss_dict["c_loss"] = c_loss

        # total loss
        loss = f_loss + c_loss

        if self.gesm_enabled and "warped_src_nodes" in output_dict:
            # 只有当 NeuralGESM_PC 已经接入并输出 adapt_weights 时才计算
            if "adapt_weights" in output_dict and "displacements" in output_dict:
                #print(output_dict["anchor_indices"].shape)
                gesm_loss = compute_gesm_pc_loss(
                    vertices=output_dict["warped_src_nodes"], #["warped_src_points"],
                    displacements=output_dict["displacements"],
                    normals=output_dict.get("node_normals"),
                    anchor_indices=output_dict["corr_anchor_indices"],
                    anchor_weights=output_dict["corr_anchor_weights"],
                    node_weights=output_dict["adapt_weights"],   # SirenVW 输出的 4 维动态权重
                    jacobians=output_dict["jacobian"],
                    L=output_dict.get("node_L"),
                    M=output_dict.get("node_M"),
                    method=self.gesm_method,
                )
                weighted_gesm = self.lambda_gesm * gesm_loss
                loss = loss + weighted_gesm

                loss_dict["gesm_loss"] = gesm_loss
                loss_dict["weighted_gesm_loss"] = weighted_gesm
            
            if "tgt_corr_points" in output_dict:
                #warped_src_corr = apply_deformation_jacobian(
                #        points=output_dict["src_corr_points"],
                #        nodes=output_dict["src_nodes"],                    # deformation graph 节点
                #        jacobians=output_dict["jacobian"],                 # GESM-PC 输出的 Jacobian
                #        displacements=output_dict["displacements"],             # GESM-PC 输出的位移
                #        anchor_indices=output_dict["corr_anchor_indices"],
                #        anchor_weights=output_dict["corr_anchor_weights"])
                warped_src_corr = output_dict["warped_src_corr"]

                #corr_mse = torch.mean((output_dict["warped_src_corr"] - output_dict["tgt_corr_points"])**2)
                corr_mse = torch.mean(torch.abs(output_dict["warped_src_corr"] - output_dict["tgt_corr_points"]))
                #print(corr_mse)
                loss_dict["corr_mse"] = corr_mse

                jacb_reg = arap_edge_regularization_jacobian(
                        nodes=output_dict["src_nodes"],                    # deformation graph 的节点
                        jacobians=output_dict["jacobian"],        # (M, 3, 3)
                        displacements=output_dict["displacements"],    # (M, 3)
                        edge_indices=output_dict["corr_edges_indices"],
                        edge_weights=output_dict["corr_edge_weights"],
                        reduction='mean')
                loss_dict["jabc_reg"] = jacb_reg

            if "weighted_gesm_loss" in loss_dict:
                total_registration_loss = self.lambda_corr * corr_mse + self.lambda_gesm * loss_dict["weighted_gesm_loss"] + self.jacb_reg * jacb_reg
                loss_dict["registration_loss"] = total_registration_loss
                loss = loss + total_registration_loss

            '''
            if "edges_indices" in output_dict:
                arap_loss = compute_arap_loss(
                        warped_points=output_dict["warped_src_points"],
                        orig_points=data_dict["src_points"],
                        edges_indices=output_dict["edges_indices"],
                        edge_weights=None)

                loss_dict["arap_loss"] = arap_loss
            # 使用 ARAP 作为正则
            if "registration_loss" in loss_dict:
                loss_dict["registration_loss"] = loss_dict["corr_mse"] + self.lambda_gesm * arap_lose
            else:
                loss_dict["registration_loss"] = arap_loss
            '''

            loss_dict["loss"] = loss

        return loss_dict


class EvalFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.acceptance_score = cfg.eval.acceptance_score
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.distance_limit = cfg.eval.distance_limit

    def forward(self, data_dict, output_dict):
        result_dict = {}

        # inlier/outlier classification
        scores = output_dict["corr_scores"]
        labels = data_dict["corr_labels"].float()
        precision, recall = evaluate_binary_classification(
            scores, labels, positive_threshold=self.acceptance_score, use_logits=False
        )

        corr_masks = output_dict["corr_masks"]
        hit_ratio = corr_masks.float().mean().nan_to_num_()

        result_dict["precision"] = precision
        result_dict["recall"] = recall
        result_dict["hit_ratio"] = hit_ratio

        # non-rigid inlier ratio, non-rigid feature matching recall
        src_points = data_dict["src_points"]
        scene_flows = data_dict["scene_flows"]
        transform = data_dict["transform"]
        src_corr_points = data_dict["src_corr_points"]
        tgt_corr_points = data_dict["tgt_corr_points"]
        corr_masks = torch.gt(scores, self.acceptance_score)
        if corr_masks.sum() > 0:
            src_corr_points = src_corr_points[corr_masks]
            tgt_corr_points = tgt_corr_points[corr_masks]

        if "test_indices" in data_dict:
            test_indices = data_dict["test_indices"]
            nfmr = compute_nonrigid_feature_matching_recall(
                src_corr_points,
                tgt_corr_points,
                src_points,
                scene_flows,
                test_indices,
                transform=transform,
                acceptance_radius=self.acceptance_radius,
                distance_limit=self.distance_limit,
            )
            result_dict["NFMR"] = nfmr

        # overlap coverage
        gt_src_corr_indices = data_dict["gt_src_corr_indices"]
        src_overlap_indices = torch.unique(gt_src_corr_indices)
        coverage = compute_nonrigid_feature_matching_recall(
            src_corr_points,
            tgt_corr_points,
            src_points,
            scene_flows,
            src_overlap_indices,
            transform=transform,
            acceptance_radius=self.acceptance_radius,
            distance_limit=self.distance_limit,
        )
        result_dict["coverage"] = coverage


        if "warped_src_points" in output_dict or "embedded_deformation_transforms" in output_dict:

            src_points = data_dict["src_points"]
            scene_flows = data_dict["scene_flows"]
            transform = data_dict["transform"]


            if True:
                # ==================== 新流程：Neural GESM-PC ====================
                print("[EVAL] 使用 Neural GESM-PC 的 warped_src_points 计算注册指标")
                warped_src_points = output_dict["warped_src_points"]
                #nodes = output_dict["embedded_deformation_nodes"]
                #node_transforms = output_dict["embedded_deformation_transforms"]
                #anchor_indices = output_dict["anchor_indices"]
                #anchor_weights = output_dict["anchor_weights"]
                #warp_src_points = apply_deformation_jacobian(points=src_points,nodes=nodes,jacobians=nodes_jacb,displacements=nodes_disp,anchor_indices=anchor_indices,anchor_weights=anchor_weights)
            else:
                # ==================== 旧流程：N-ICP ====================
                print("[EVAL] 使用传统 N-ICP 的 embedded_deformation_transforms 计算注册指标")
                nodes = output_dict["embedded_deformation_nodes"]
                node_transforms = output_dict["embedded_deformation_transforms"]
                anchor_indices = output_dict["anchor_indices"]
                anchor_weights = output_dict["anchor_weights"]
                warped_src_points = apply_deformation(src_points, nodes, node_transforms, anchor_indices, anchor_weights)

            # ==================== 统一计算注册指标 ====================
            src_points = src_points * data_dict["src_scale"] + data_dict["src_center"]
            aligned_src_points = apply_transform(src_points + scene_flows, transform)
            warped_scene_flows = warped_src_points - src_points
            aligned_scene_flows = aligned_src_points - src_points

            epe = torch.linalg.norm(warped_scene_flows - aligned_scene_flows, dim=1).mean()
            acc_s = compute_scene_flow_accuracy(warped_scene_flows, aligned_scene_flows, 0.025, 0.025)
            acc_r = compute_scene_flow_accuracy(warped_scene_flows, aligned_scene_flows, 0.05, 0.05)
            outlier_ratio = compute_scene_flow_outlier_ratio(warped_scene_flows, aligned_scene_flows, None, 0.3)

            result_dict["EPE"] = epe
            result_dict["AccS"] = acc_s
            result_dict["AccR"] = acc_r
            result_dict["OR"] = outlier_ratio

        #if "embedded_deformation_transforms" in output_dict:
        #    nodes = output_dict["embedded_deformation_nodes"]  # (M, 3)
        #    node_transforms = output_dict["embedded_deformation_transforms"]  # (M, 4, 4)
        #    anchor_indices = output_dict["anchor_indices"]
        #    anchor_weights = output_dict["anchor_weights"]

        #    warped_src_points = apply_deformation(src_points, nodes, node_transforms, anchor_indices, anchor_weights)
        #    aligned_src_points = apply_transform(src_points + scene_flows, transform)
        #    warped_scene_flows = warped_src_points - src_points
        #    aligned_scene_flows = aligned_src_points - src_points
        #    epe = torch.linalg.norm(warped_scene_flows - aligned_scene_flows, dim=1).mean()
        #    acc_s = compute_scene_flow_accuracy(warped_scene_flows, aligned_scene_flows, 0.025, 0.025)
        #    acc_r = compute_scene_flow_accuracy(warped_scene_flows, aligned_scene_flows, 0.05, 0.05)
        #    outlier_ratio = compute_scene_flow_outlier_ratio(warped_scene_flows, aligned_scene_flows, None, 0.3)

        #    result_dict["EPE"] = epe
        #    result_dict["AccS"] = acc_s
        #    result_dict["AccR"] = acc_r
        #    result_dict["OR"] = outlier_ratio

        result_dict["nCorr"] = src_corr_points.shape[0]

        return result_dict
