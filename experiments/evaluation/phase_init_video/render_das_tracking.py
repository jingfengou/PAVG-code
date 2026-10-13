"""Render a merged track payload into the RGB condition consumed by DaS."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _normalize(value: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((value - low) / max(high - low, 1e-8), 0.0, 1.0)


def _ffmpeg_executable() -> str | None:
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _track_colors(tracks: np.ndarray) -> np.ndarray:
    initial = tracks[0, 0]
    x = initial[:, 0]
    y = initial[:, 1]
    inverse_depth = 1.0 / np.maximum(initial[:, 2], 1e-6)
    z_low, z_high = np.percentile(inverse_depth, [0.5, 99.9])
    color = np.stack(
        [
            _normalize(x, float(x.min()), float(x.max())),
            _normalize(y, float(y.min()), float(y.max())),
            _normalize(inverse_depth, float(z_low), float(z_high)),
        ],
        axis=-1,
    )
    return np.round(color * 255.0).astype(np.uint8)


def render_frames(
    payload: dict,
    *,
    width: int,
    height: int,
    point_radius: int,
) -> list[Image.Image]:
    tracks = np.asarray(payload["tracks"], dtype=np.float32)
    visibility = np.asarray(payload["visibility"], dtype=bool)
    colors = _track_colors(tracks)
    initial = tracks[0, 0]
    if point_radius > 0:
        radii = np.full(initial.shape[0], point_radius, dtype=np.int32)
    else:
        radii = np.empty(initial.shape[0], dtype=np.int32)
        for point_index in range(initial.shape[0]):
            distance = np.linalg.norm(initial - initial[point_index], axis=-1)
            positive = distance[distance > 0]
            radii[point_index] = max(1, int(positive.min()) // 2)
    frames = []
    for frame_index in range(tracks.shape[1]):
        image = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        order = np.argsort(tracks[0, frame_index, :, 2])[::-1]
        for point_index in order:
            if not visibility[0, frame_index, point_index]:
                continue
            x, y = tracks[0, frame_index, point_index, :2]
            if not (0.0 <= x < width and 0.0 <= y < height):
                continue
            color = tuple(int(value) for value in colors[point_index])
            radius = int(radii[point_index])
            draw.rectangle(
                (
                    x - radius,
                    y - radius / 1.5,
                    x + radius,
                    y + radius / 1.5,
                ),
                fill=color,
            )
        frames.append(image)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--point-radius",
        type=int,
        default=0,
        help="Fixed radius; zero reproduces PhysCtrl's grid-spacing rule.",
    )
    parser.add_argument("--write-mp4", action="store_true")
    args = parser.parse_args()

    payload = np.load(args.tracks, allow_pickle=True).item()
    frames = render_frames(
        payload,
        width=args.width,
        height=args.height,
        point_radius=args.point_radius,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(frame_dir / f"{index:03d}.png")
    gif_path = args.output_dir / "tracks_tracking.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / args.fps),
        loop=0,
    )

    mp4_path = args.output_dir / "tracks_tracking.mp4"
    wrote_mp4 = False
    if args.write_mp4:
        ffmpeg = _ffmpeg_executable()
        if ffmpeg is not None:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-framerate",
                    str(args.fps),
                    "-i",
                    str(frame_dir / "%03d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(mp4_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            wrote_mp4 = True

    report = {
        "frame_count": len(frames),
        "image_size": [args.width, args.height],
        "fps": args.fps,
        "renderer": "physctrl_rainbow_grid",
        "ffmpeg_available": _ffmpeg_executable() is not None,
        "outputs": {
            "gif": str(gif_path),
            "mp4": str(mp4_path) if wrote_mp4 else None,
            "frames": str(frame_dir),
        },
    }
    (args.output_dir / "render_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
