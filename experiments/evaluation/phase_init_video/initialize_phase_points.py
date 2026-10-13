"""Initialize phase-local point clouds for phase-aware image inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ply_utils import load_ply_vertices


def _sample_mask(
    path: Path,
    *,
    count: int,
    seed: int,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    alpha = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    ys, xs = np.where(alpha > 0.02)
    if len(xs) < 8:
        raise ValueError(f"Mask is empty or too small: {path}")
    weights = alpha[ys, xs]
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(xs), size=count, replace=len(xs) < count, p=weights)
    px = xs[selected].astype(np.float32)
    py = ys[selected].astype(np.float32)
    selected_alpha = alpha[ys[selected], xs[selected]]

    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    pixel_center = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32)
    pixel_scale = np.float32(max(x1 - x0, y1 - y0, 1.0))
    local = np.stack(
        [
            (px - pixel_center[0]) / pixel_scale,
            (pixel_center[1] - py) / pixel_scale,
            rng.normal(0.0, thickness * (0.25 + 0.75 * selected_alpha), count),
        ],
        axis=-1,
    ).astype(np.float32)
    local_mean = local.mean(axis=0).astype(np.float32)
    centered = local - local_mean
    model_scale = np.float32(1.0 / max(float(np.ptp(centered, axis=0).max()), 1e-6))
    model_points = centered * model_scale
    pixel_points = np.stack(
        [px, py, np.full(count, 1.0, dtype=np.float32)], axis=-1
    )
    transform = {
        "pixel_center": pixel_center.tolist(),
        "pixel_scale": float(pixel_scale),
        "local_mean": local_mean.tolist(),
        "model_scale": float(model_scale),
        "bbox_xyxy": [x0, y0, x1, y1],
    }
    return model_points, pixel_points, transform


def _farthest_point_indices(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Deterministic NumPy FPS used when PhysCtrl indices are unavailable."""

    if points.shape[0] < count:
        raise ValueError(
            f"Solid point cloud contains {points.shape[0]} points, fewer than {count}"
        )
    rng = np.random.default_rng(seed)
    indices = np.empty(count, dtype=np.int64)
    indices[0] = int(rng.integers(points.shape[0]))
    minimum_distance = np.full(points.shape[0], np.inf, dtype=np.float64)
    for sample_index in range(1, count):
        selected = points[indices[sample_index - 1]]
        distance = np.sum((points - selected) ** 2, axis=-1, dtype=np.float64)
        minimum_distance = np.minimum(minimum_distance, distance)
        indices[sample_index] = int(np.argmax(minimum_distance))
    return indices


def _load_solid_lgm(
    path: Path,
    *,
    count: int,
    seed: int,
    fps_index_path: Path | None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Reproduce PhysCtrl's LGM point normalization and 2,048-point sampling."""

    vertices = load_ply_vertices(path)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < count:
        raise ValueError(f"Unexpected solid vertices in {path}: {vertices.shape}")

    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    center = (minimum + maximum) / 2.0
    extent = float(np.max(maximum - minimum))
    if extent <= 1e-8:
        raise ValueError(f"Degenerate solid point cloud: {path}")
    world_scale = np.float32(2.0 / extent)
    normalized = (vertices - center) * world_scale / 2.0

    if fps_index_path is not None:
        fps_indices = np.asarray(np.load(fps_index_path), dtype=np.int64).reshape(-1)
        if fps_indices.shape[0] != count:
            raise ValueError(
                f"Expected {count} PhysCtrl FPS indices, got {fps_indices.shape[0]}"
            )
        if fps_indices.min() < 0 or fps_indices.max() >= vertices.shape[0]:
            raise ValueError("PhysCtrl FPS indices fall outside the LGM point cloud")
        sampler = "physctrl_fps_indices"
    else:
        fps_indices = _farthest_point_indices(normalized, count, seed)
        sampler = "numpy_farthest_point_sampling"

    model_points = normalized[fps_indices].astype(np.float32)
    arrays = {
        "solid_world_points": vertices[fps_indices].astype(np.float32),
        "solid_fps_idx": fps_indices.astype(np.int64),
        "solid_world_center": center.astype(np.float32),
        "solid_world_scale": np.asarray(world_scale, dtype=np.float32),
    }
    transform = {
        "type": "physctrl_lgm",
        "point_cloud": str(path),
        "sampler": sampler,
        "world_center": center.tolist(),
        "world_scale": float(world_scale),
        "model_to_world": "world = model * 2 / world_scale + world_center",
    }
    return model_points, arrays, transform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solid-mask", type=Path)
    parser.add_argument("--solid-ply", type=Path)
    parser.add_argument("--solid-fps-index", type=Path)
    parser.add_argument("--gas-mask", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--solid-thickness", type=float, default=0.22)
    parser.add_argument("--gas-thickness", type=float, default=0.12)
    args = parser.parse_args()

    model_points = []
    phase_presence = []
    arrays: dict[str, np.ndarray] = {}
    transforms = {}
    for phase in ("gas", "solid"):
        mask_path = args.gas_mask if phase == "gas" else args.solid_mask
        thickness = args.gas_thickness if phase == "gas" else args.solid_thickness
        seed = args.seed if phase == "gas" else args.seed + 1
        if phase == "solid" and args.solid_ply is not None:
            points, solid_arrays, transform = _load_solid_lgm(
                args.solid_ply,
                count=args.count,
                seed=seed,
                fps_index_path=args.solid_fps_index,
            )
            arrays.update(solid_arrays)
            present = True
        elif mask_path is None:
            points = np.zeros((args.count, 3), dtype=np.float32)
            pixels = np.zeros((args.count, 3), dtype=np.float32)
            transform = None
            present = False
        else:
            points, pixels, transform = _sample_mask(
                mask_path, count=args.count, seed=seed, thickness=thickness
            )
            present = True
        model_points.append(points)
        phase_presence.append(present)
        if not (phase == "solid" and args.solid_ply is not None):
            arrays[f"{phase}_pixel_points"] = pixels
        if transform is not None and transform.get("type") != "physctrl_lgm":
            arrays[f"{phase}_pixel_center"] = np.asarray(
                transform["pixel_center"], dtype=np.float32
            )
            arrays[f"{phase}_pixel_scale"] = np.asarray(
                transform["pixel_scale"], dtype=np.float32
            )
            arrays[f"{phase}_local_mean"] = np.asarray(
                transform["local_mean"], dtype=np.float32
            )
            arrays[f"{phase}_model_scale"] = np.asarray(
                transform["model_scale"], dtype=np.float32
            )
        transforms[phase] = transform

    if not any(phase_presence):
        raise ValueError("At least one phase mask is required")
    points_src = np.stack(model_points, axis=0).astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "phase_inputs.npz",
        points_src=points_src,
        phase_presence=np.asarray(phase_presence, dtype=bool),
        **arrays,
    )
    for phase, points in zip(("gas", "solid"), model_points):
        np.save(args.output_dir / f"{phase}_points_model.npy", points)
    report = {
        "slot_order": ["gas", "solid"],
        "points_src_shape": list(points_src.shape),
        "phase_presence": phase_presence,
        "transforms": transforms,
        "note": (
            "A mask-derived solid cloud is only a geometry-contract proxy. When "
            "solid_ply is provided, the points follow PhysCtrl's LGM normalization and "
            "must be projected with the saved LGM camera contract."
        ),
    }
    (args.output_dir / "initialization_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
