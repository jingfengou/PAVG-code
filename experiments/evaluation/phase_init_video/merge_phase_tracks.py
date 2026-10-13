"""Merge gas and solid trajectories into one PhysCtrl/DaS track payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_TEMPLATE = Path(__file__).resolve().parents[3] / "examples/tracks_template.npz"


def _load_prediction(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.lib.npyio.NpzFile):
        for key in ("restored_prediction", "prediction", "trajectory", "trajectories"):
            if key in value:
                prediction = np.asarray(value[key], dtype=np.float32)
                break
        else:
            raise KeyError(f"No prediction array in {path}")
    else:
        prediction = np.asarray(value, dtype=np.float32)
    if prediction.ndim == 5 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim != 4 or prediction.shape[1] != 2 or prediction.shape[-1] != 3:
        raise ValueError(f"Prediction must be [F,2,N,3], got {prediction.shape}")
    return prediction


def _demo_prediction(points_src: np.ndarray, frames: int = 24) -> np.ndarray:
    output = np.repeat(points_src[None], frames, axis=0)
    gas0 = points_src[0]
    for frame in range(frames):
        t = (frame + 1) / frames
        angle = 1.5 * np.pi * t
        radius = np.linalg.norm(gas0[:, :2], axis=-1, keepdims=True)
        swirl = np.concatenate(
            [-gas0[:, 1:2], gas0[:, 0:1], np.zeros_like(gas0[:, :1])], axis=-1
        )
        output[frame, 0] = gas0 + 0.10 * t * swirl * (0.3 + radius) + [0.03 * t, 0.24 * t, 0.0]
        output[frame, 1] = points_src[1] + [0.10 * t, 0.04 * np.sin(np.pi * t), 0.0]
    return output.astype(np.float32)


def _upsample_49(initial: np.ndarray, future: np.ndarray) -> np.ndarray:
    if future.shape[0] != 24:
        raise ValueError(f"Expected 24 future physical frames, got {future.shape[0]}")
    output = np.zeros((49, *future.shape[1:]), dtype=np.float32)
    output[0] = initial
    for index in range(24):
        output[2 * index + 1] = future[index]
        if index < 23:
            output[2 * index + 2] = 0.5 * (future[index] + future[index + 1])
    output[48] = future[-1]
    return output


def _project_local(trajectory: np.ndarray, data, phase: str) -> np.ndarray:
    local_mean = np.asarray(data[f"{phase}_local_mean"], dtype=np.float32)
    model_scale = float(np.asarray(data[f"{phase}_model_scale"]).item())
    pixel_center = np.asarray(data[f"{phase}_pixel_center"], dtype=np.float32)
    pixel_scale = float(np.asarray(data[f"{phase}_pixel_scale"]).item())
    local = trajectory / model_scale + local_mean
    projected = np.empty_like(local)
    projected[..., 0] = pixel_center[0] + local[..., 0] * pixel_scale
    projected[..., 1] = pixel_center[1] - local[..., 1] * pixel_scale
    projected[..., 2] = 1.0 + local[..., 2]
    return projected


def _nearest(initial_points: np.ndarray, anchors: np.ndarray, max_distance: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.zeros(len(anchors), dtype=np.int64)
    valid = np.zeros(len(anchors), dtype=bool)
    for start in range(0, len(anchors), 512):
        stop = min(start + 512, len(anchors))
        distances = np.sum(
            (anchors[start:stop, None, :2] - initial_points[None, :, :2]) ** 2,
            axis=-1,
        )
        nearest = distances.argmin(axis=1)
        indices[start:stop] = nearest
        valid[start:stop] = distances[np.arange(stop - start), nearest] <= max_distance**2
    return indices, valid


def _freeze_background(tracks, visibility, phase_labels, width, height):
    """Discard unrelated template motion using PhysCtrl's static-background contract."""
    background = phase_labels == 0
    initial = tracks[0, 0, background, :2].copy()
    tracks[0, :, background, :2] = initial[:, None, :]
    tracks[0, :, background, 2] = 2.0
    valid = ((initial[:, 0] >= 0) & (initial[:, 0] < width)
             & (initial[:, 1] >= 0) & (initial[:, 1] < height))
    visibility[0, :, background] = valid[:, None]
    if not np.array_equal(tracks[0, :, background, :2], np.broadcast_to(
        initial[:, None, :], (len(initial), tracks.shape[1], 2)
    )):
        raise RuntimeError("Background anchors must remain stationary")


def _render_preview(
    tracks: np.ndarray,
    visibility: np.ndarray,
    phase_labels: np.ndarray,
    path: Path,
    width: int,
    height: int,
) -> list[Image.Image]:
    colors = {1: (241, 153, 61), 2: (70, 205, 231)}
    frames = []
    for frame_index in range(tracks.shape[1]):
        image = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        for phase, color in colors.items():
            active = (phase_labels == phase) & visibility[0, frame_index]
            for x, y, _ in tracks[0, frame_index, active]:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        frames.append(image)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=[120 if index % 2 == 0 else 130 for index in range(len(frames))], loop=0)
    return frames


def _write_keyframe_sheet(frames: list[Image.Image], path: Path) -> None:
    indices = [0, 12, 24, 36, 48]
    tile_size = (360, 240)
    sheet = Image.new("RGB", (tile_size[0] * 3, tile_size[1] * 2), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for tile_index, frame_index in enumerate(indices):
        tile = frames[frame_index].resize(tile_size, Image.Resampling.LANCZOS)
        left = (tile_index % 3) * tile_size[0]
        top = (tile_index // 3) * tile_size[1]
        sheet.paste(tile, (left, top))
        draw.text((left + 8, top + 8), f"frame {frame_index}", fill=(255, 255, 255))
    sheet.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-inputs", required=True, type=Path)
    parser.add_argument("--solid-mask", type=Path)
    parser.add_argument("--gas-mask", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--solid-projected", type=Path)
    parser.add_argument("--gas-projected", type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE,
                        help="Numeric NPZ or trusted-local legacy track template.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-anchor-distance", type=float, default=18.0)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    phase_data = np.load(args.phase_inputs, allow_pickle=True)
    points_src = np.asarray(phase_data["points_src"], dtype=np.float32)
    phase_presence = np.asarray(phase_data["phase_presence"], dtype=bool)
    if args.demo:
        future = _demo_prediction(points_src)
    elif args.prediction is not None:
        future = _load_prediction(args.prediction)
    else:
        raise ValueError("Provide --prediction or use --demo")
    if future.shape[1:] != points_src.shape:
        raise ValueError(
            f"Prediction/input mismatch: future={future.shape}, points_src={points_src.shape}"
        )
    trajectories = _upsample_49(points_src, future)

    projected = {}
    for slot, phase in enumerate(("gas", "solid")):
        override = args.gas_projected if phase == "gas" else args.solid_projected
        if override is not None:
            value = np.asarray(np.load(override), dtype=np.float32)
            if value.shape[0] != 49 or value.shape[-1] != 3:
                raise ValueError(f"{phase} projected tracks must be [49,N,3]")
            projected[phase] = value
        elif phase_presence[slot]:
            if f"{phase}_registered_world" in phase_data:
                from project_native import project_native
                projected[phase] = project_native(phase_data, future, slot, phase)
            else:
                projected[phase] = _project_local(trajectories[:, slot], phase_data, phase)

    if args.template.suffix == ".npz":
        with np.load(args.template, allow_pickle=False) as value:
            template = {key: value[key] for key in ("tracks", "visibility")}
    else:
        template = np.load(args.template, allow_pickle=True).item()
    tracks = np.asarray(template["tracks"], dtype=np.float32).copy()
    visibility = np.asarray(template["visibility"], dtype=bool).copy()
    if tracks.shape[1] != 49:
        raise ValueError(f"DaS template must contain 49 frames, got {tracks.shape}")
    height = Image.open(args.solid_mask or args.gas_mask).height
    width = Image.open(args.solid_mask or args.gas_mask).width
    anchors = tracks[0, 0]
    phase_labels = np.zeros(anchors.shape[0], dtype=np.int8)

    if args.gas_mask is not None and "gas" in projected:
        mask = np.asarray(Image.open(args.gas_mask).convert("L"))
        x = np.clip(np.round(anchors[:, 0]).astype(int), 0, width - 1)
        y = np.clip(np.round(anchors[:, 1]).astype(int), 0, height - 1)
        phase_labels[mask[y, x] > 8] = 1
    if args.solid_mask is not None and "solid" in projected:
        mask = np.asarray(Image.open(args.solid_mask).convert("L"))
        x = np.clip(np.round(anchors[:, 0]).astype(int), 0, width - 1)
        y = np.clip(np.round(anchors[:, 1]).astype(int), 0, height - 1)
        phase_labels[mask[y, x] > 8] = 2

    source_point_index = np.full(anchors.shape[0], -1, dtype=np.int32)
    for phase, label in (("gas", 1), ("solid", 2)):
        selected = np.flatnonzero(phase_labels == label)
        if not len(selected):
            continue
        indices, valid = _nearest(
            projected[phase][0], anchors[selected], args.max_anchor_distance
        )
        accepted = selected[valid]
        source_point_index[accepted] = indices[valid]
        tracks[0, :, accepted] = projected[phase][:, indices[valid]].transpose(1, 0, 2)
        in_frame = (
            (tracks[0, :, accepted, 0] >= 0)
            & (tracks[0, :, accepted, 0] < width)
            & (tracks[0, :, accepted, 1] >= 0)
            & (tracks[0, :, accepted, 1] < height)
        )
        visibility[0, :, accepted] = in_frame
        rejected = selected[~valid]
        phase_labels[rejected] = 0

    _freeze_background(tracks, visibility, phase_labels, width, height)

    payload = {
        "tracks": tracks,
        "visibility": visibility,
        "phase_labels": phase_labels,
        "phase_label_names": {"0": "background", "1": "gas", "2": "solid"},
        "source_point_index": source_point_index,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "tracks.npy", payload, allow_pickle=True)
    preview_frames = _render_preview(
        tracks,
        visibility,
        phase_labels,
        args.output_dir / "tracking_preview.gif",
        width,
        height,
    )
    keyframe_path = args.output_dir / "tracking_keyframes.png"
    _write_keyframe_sheet(preview_frames, keyframe_path)
    phase_metrics = {}
    for phase, label in (("gas", 1), ("solid", 2)):
        selected = phase_labels == label
        if bool(selected.any()):
            displacement = np.linalg.norm(
                tracks[0, -1, selected, :2] - tracks[0, 0, selected, :2], axis=-1
            )
            phase_metrics[phase] = {
                "mean_final_pixel_displacement": float(displacement.mean()),
                "visible_frame_fraction": float(visibility[0, :, selected].mean()),
            }
    report = {
        "background_contract": "fixed first-frame xy, z=2; no template motion",
        "background_max_displacement_px": 0.0,
        "tracks_shape": list(tracks.shape),
        "assigned_anchors": {
            "gas": int(np.count_nonzero(phase_labels == 1)),
            "solid": int(np.count_nonzero(phase_labels == 2)),
            "background": int(np.count_nonzero(phase_labels == 0)),
        },
        "phase_metrics": phase_metrics,
        "outputs": {
            "tracks": str(args.output_dir / "tracks.npy"),
            "preview": str(args.output_dir / "tracking_preview.gif"),
            "keyframes": str(keyframe_path),
        },
    }
    (args.output_dir / "tracking_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
