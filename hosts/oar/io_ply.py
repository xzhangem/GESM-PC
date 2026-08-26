"""PLY I/O, unit-sphere normalization, NN-RMSE."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from scipy.spatial import KDTree


def load_xyz(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise FileNotFoundError(path)
    return np.asarray(pcd.points, dtype=np.float32)


def save_xyz(path: str, points: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float32))
    o3d.io.write_point_cloud(path, pcd)


def normalize_points(points: np.ndarray):
    center = points.mean(axis=0, keepdims=True)
    x = points - center
    scale = np.linalg.norm(x, axis=1).max()
    scale = float(scale) if scale > 0 else 1.0
    return (x / scale).astype(np.float32), center.reshape(3), scale


def denormalize(points: np.ndarray, center, scale) -> np.ndarray:
    return points * scale + center


def nn_rmse(deformed: np.ndarray, target: np.ndarray) -> float:
    dist, _ = KDTree(target).query(deformed, k=1)
    return float(np.sqrt(np.mean(dist ** 2)))


def to_torch(points: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(points, dtype=np.float32)).to(device)
