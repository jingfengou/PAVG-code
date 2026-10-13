"""PhysCtrl-compatible point-cloud trajectory metrics.

The public PhysCtrl evaluator computes vIoU, Chamfer distance, and xyz MSE
with a fixed voxel grid. Corr is the correspondence-aware extension used by
the project evaluators: the point ordering is preserved and the mean
Euclidean error is averaged over frames and samples.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


VOXEL_SIZE = 0.1
GRID_MIN = np.array([-1.5, -1.5, -1.5], dtype=np.float32)
GRID_MAX = np.array([1.5, 1.5, 1.5], dtype=np.float32)


def _validate_sequence(points: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [T, N, 3], got {points.shape}")
    return points


def _occupied_voxels(
    points: np.ndarray,
    voxel_size: float = VOXEL_SIZE,
    grid_min: np.ndarray = GRID_MIN,
    grid_max: np.ndarray = GRID_MAX,
) -> set[tuple[int, int, int]]:
    valid = np.all((points >= grid_min) & (points < grid_max), axis=1)
    valid_points = points[valid]
    if valid_points.size == 0:
        return set()
    indices = np.floor((valid_points - grid_min) / voxel_size).astype(np.int64)
    return {tuple(index) for index in indices}


def volume_iou(
    prediction: np.ndarray,
    target: np.ndarray,
    voxel_size: float = VOXEL_SIZE,
    grid_min: np.ndarray = GRID_MIN,
    grid_max: np.ndarray = GRID_MAX,
) -> float:
    """Compute the official PhysCtrl occupied-voxel IoU for one frame."""
    pred_voxels = _occupied_voxels(prediction, voxel_size, grid_min, grid_max)
    target_voxels = _occupied_voxels(target, voxel_size, grid_min, grid_max)
    union = pred_voxels | target_voxels
    if not union:
        return 0.0
    return float(len(pred_voxels & target_voxels) / len(union))


def chamfer_distance(prediction: np.ndarray, target: np.ndarray) -> float:
    """Compute PhysCtrl's sum of directed squared nearest-neighbor means."""
    pred_tree = cKDTree(prediction)
    target_tree = cKDTree(target)
    pred_to_target = target_tree.query(prediction, k=1)[0]
    target_to_pred = pred_tree.query(target, k=1)[0]
    return float(np.mean(pred_to_target**2) + np.mean(target_to_pred**2))


def sequence_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    voxel_size: float = VOXEL_SIZE,
    grid_min: np.ndarray = GRID_MIN,
    grid_max: np.ndarray = GRID_MAX,
    include_velocity_mse: bool = False,
) -> dict[str, float]:
    """Compute metrics for one [T, N, 3] trajectory.

    vIoU and CD are averaged per frame, matching ``volume_iou.py``. xyz_mse
    is the mean over all frames, points, and coordinates. Corr preserves the
    known point identities and averages the per-point Euclidean error.
    """
    prediction = _validate_sequence(prediction, "prediction")
    target = _validate_sequence(target, "target")
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got "
            f"{prediction.shape} and {target.shape}"
        )

    frame_iou = []
    frame_cd = []
    frame_corr = []
    frame_xyz_mse = []
    frame_velocity_mse = []
    for frame_index in range(prediction.shape[0]):
        pred_frame = prediction[frame_index]
        target_frame = target[frame_index]
        frame_iou.append(
            volume_iou(pred_frame, target_frame, voxel_size, grid_min, grid_max)
        )
        frame_cd.append(chamfer_distance(pred_frame, target_frame))
        frame_corr.append(float(np.linalg.norm(pred_frame - target_frame, axis=-1).mean()))
        frame_xyz_mse.append(float(np.mean((pred_frame - target_frame) ** 2)))
        if frame_index > 0:
            pred_velocity = pred_frame - prediction[frame_index - 1]
            target_velocity = target_frame - target[frame_index - 1]
            frame_velocity_mse.append(float(np.mean((pred_velocity - target_velocity) ** 2)))

    result = {
        "vIoU": float(np.mean(frame_iou)),
        "CD": float(np.mean(frame_cd)),
        "Corr": float(np.mean(frame_corr)),
        "xyz_mse": float(np.mean(frame_xyz_mse)),
    }
    if include_velocity_mse:
        result["velocity_mse"] = float(np.mean(frame_velocity_mse)) if frame_velocity_mse else 0.0
    return result


def macro_average(records: Iterable[dict[str, float]]) -> dict[str, float]:
    """Average already per-sample metric records with equal sample weight."""
    records = list(records)
    if not records:
        raise ValueError("cannot average an empty metric collection")
    keys = tuple(records[0].keys())
    return {key: float(np.mean([record[key] for record in records])) for key in keys}


def metric_protocol() -> dict[str, object]:
    """Return reproducibility metadata for a result JSON."""
    return {
        "name": "physctrl_official_v1_plus_corr",
        "voxel_size": VOXEL_SIZE,
        "grid_min": GRID_MIN.tolist(),
        "grid_max": GRID_MAX.tolist(),
        "cd": "mean squared directed nearest-neighbor distance sum; no 1/2 factor",
        "corr": "mean Euclidean distance between corresponding ordered points",
        "xyz_mse": "mean squared xyz error over frames, points, and coordinates",
        "aggregation": "per-frame then per-sample; final tables use equal-sample macro average",
    }
