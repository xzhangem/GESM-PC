import torch
from torch import Tensor
from typing import Tuple

from .conversion import batch_to_pack, pack_to_batch



def keops_knn(q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """kNN with PyKeOps. (Pure PyTorch 干净版 - 推荐)

    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k (int)

    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """
    num_batch_dims = q_points.dim() - 2
    num_s_points = s_points.shape[-2]

    # 关键修复：k 不能超过 s_points 数量
    effective_k = min(k, num_s_points)

    dist = torch.cdist(q_points, s_points)          # (*, N, M)
    knn_distances, knn_indices = torch.topk(
        dist, k=effective_k, dim=num_batch_dims + 1, largest=False
    )

    return knn_distances, knn_indices

'''
def keops_knn(q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """kNN with PyKeOps. (Pure PyTorch version - 完全等价实现)

    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k (int)

    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """
    # 完全保留原始逻辑：计算 (*, N, M) 的 pairwise distance，然后 top-k
    num_batch_dims = q_points.dim() - 2

    # 使用 torch.cdist 替代 KeOps 的 (xi - xj).norm2()
    # cdist 会自动处理 batch 维度
    dist = torch.cdist(q_points, s_points)          # (*, N, M)

    # 保留原始的 Kmin 行为（取最小的 k 个）
    knn_distances, knn_indices = torch.topk(
        dist, k=k, dim=num_batch_dims + 1, largest=False
    )

    return knn_distances, knn_indices
'''

def knn(
    q_points: Tensor,
    s_points: Tensor,
    k: int,
    dilation: int = 1,
    distance_limit: float = None,
    return_distance: bool = False,
    remove_nearest: bool = False,
    transposed: bool = False,
    padding_mode: str = "nearest",
    padding_value: float = 1e10,
    squeeze: bool = False,
):
    """Compute the kNNs of the points in `q_points` from the points in `s_points`.
    Pure PyTorch version (完全遵循原 KeOps 逻辑).
    """
    if transposed:
        q_points = q_points.transpose(-1, -2)  # (*, C, N) -> (*, N, C)
        s_points = s_points.transpose(-1, -2)  # (*, C, M) -> (*, M, C)

    q_points = q_points.contiguous()
    s_points = s_points.contiguous()

    num_s_points = s_points.shape[-2]

    dilated_k = (k - 1) * dilation + 1
    if remove_nearest:
        dilated_k += 1
    final_k = min(dilated_k, num_s_points)

    # 调用之前提供的 keops_knn（纯 PyTorch 版）
    knn_distances, knn_indices = keops_knn(q_points, s_points, final_k)  # (*, N, k)

    if remove_nearest:
        knn_distances = knn_distances[..., 1:]
        knn_indices = knn_indices[..., 1:]

    if dilation > 1:
        knn_distances = knn_distances[..., ::dilation]
        knn_indices = knn_indices[..., ::dilation]

    knn_distances = knn_distances.contiguous()
    knn_indices = knn_indices.contiguous()

    if distance_limit is not None:
        assert padding_mode in ["nearest", "empty"]
        knn_masks = torch.ge(knn_distances, distance_limit)
        if padding_mode == "nearest":
            knn_distances = torch.where(knn_masks, knn_distances[..., :1], knn_distances)
            knn_indices = torch.where(knn_masks, knn_indices[..., :1], knn_indices)
        else:
            knn_distances[knn_masks] = padding_value
            knn_indices[knn_masks] = num_s_points

    if squeeze and k == 1:
        knn_distances = knn_distances.squeeze(-1)
        knn_indices = knn_indices.squeeze(-1)

    if return_distance:
        return knn_distances, knn_indices

    return knn_indices



def knn_pack_mode(
    q_points: Tensor,
    s_points: Tensor,
    q_lengths: Tensor,
    s_lengths: Tensor,
    k: int,
    return_distance: bool = False,
    inf: float = 1e10,
):
    """Pack mode KNN using pure torch"""
    assert torch.all(torch.ge(s_lengths, k)), f"The number of support points less than {k}."
    
    batch_q_points, batch_q_masks = pack_to_batch(q_points, q_lengths, fill_value=inf)
    batch_s_points, batch_s_masks = pack_to_batch(s_points, s_lengths, fill_value=inf)
    
    batch_knn_distances, batch_knn_indices = keops_knn(batch_q_points, batch_s_points, k)
    
    knn_indices, _ = batch_to_pack(batch_knn_indices, masks=batch_q_masks)
    if not return_distance:
        return knn_indices
    
    knn_distances, _ = batch_to_pack(batch_knn_distances, masks=batch_q_masks)
    return knn_distances, knn_indices
