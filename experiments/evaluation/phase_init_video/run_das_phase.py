"""Run frozen DaS on a phase-aware merged tracking video."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "libs" / "das"))
sys.path.insert(0, str(ROOT / "libs"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", "--input_path", required=True, type=Path)
    parser.add_argument("--tracking-video", "--tracking_path", required=True, type=Path)
    parser.add_argument("--checkpoint", "--checkpoint_path", required=True, type=Path)
    parser.add_argument("--output-dir", "--output_dir", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate input media and show the protocol without loading models.")
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Keep inactive DaS modules on CPU to fit lower-memory GPUs.",
    )
    args = parser.parse_args()
    import cv2
    from PIL import Image
    if args.fps != 24:
        raise ValueError("The 49-frame video interface requires 24 fps")
    capture = cv2.VideoCapture(str(args.tracking_video))
    if not capture.isOpened():
        raise ValueError("Cannot open the tracking video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    if frames != 49 or abs(fps - 24) > 1e-6 or (width, height) != (720, 480):
        raise ValueError("Tracking control must decode to 49 frames at 24 fps and 720 x 480")
    with Image.open(args.input_image) as image:
        image.verify()
    record = {"seed": args.seed, "steps": 50, "CFG": 6.0, "frames": 49,
              "fps": 24, "width": 720, "height": 480, "first_to_last_seconds": 2.0,
              "file_duration_seconds": 49 / 24, "cpu_offload": args.cpu_offload,
              "tracking_vae_global_rng_seeded": True}
    if args.check_only:
        print(json.dumps({"status": "media_checked_no_model_inference", **record}, indent=2))
        return
    import torch
    if args.cpu_offload:
        os.environ["DAS_DISABLE_TORCH_COMPILE"] = "1"
    output_root = args.output_dir.resolve()
    for folder in (output_root, *output_root.parents):
        marker = folder / "PROTOCOL_STATUS.json"
        if marker.exists():
            status = json.loads(marker.read_text(encoding="utf-8-sig"))
            if (status.get("generation_allowed") is False
                    or status.get("state", "").startswith(("superseded", "retired"))):
                raise RuntimeError(f"Generation disabled for retired protocol: {marker}")

    # The tracking VAE samples from the global RNG, separately from video noise.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing DaS checkpoint: {args.checkpoint}")
    from das.infer import load_media
    from das.models.pipelines import DiffusionAsShaderPipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_tensor, _, _ = load_media(str(args.input_image))
    tracking_tensor, _, _ = load_media(str(args.tracking_video))
    pipeline = DiffusionAsShaderPipeline(
        gpu_id=args.gpu,
        output_dir=str(args.output_dir),
        cpu_offload=args.cpu_offload,
    )
    pipeline.apply_tracking(
        video_tensor=video_tensor,
        fps=args.fps,
        tracking_tensor=tracking_tensor,
        img_cond_tensor=None,
        prompt=args.prompt,
        checkpoint_path=str(args.checkpoint),
        seed=args.seed,
    )
    result = args.output_dir / "result.mp4"
    output_capture = cv2.VideoCapture(str(result))
    result_fps = float(output_capture.get(cv2.CAP_PROP_FPS))
    count = 0
    while output_capture.read()[0]:
        count += 1
    output_capture.release()
    if count != 49 or abs(result_fps - 24) > 1e-6:
        raise RuntimeError("Generated video does not meet the frame/time contract")
    (args.output_dir / "inference_protocol.json").write_text(
        json.dumps({"status": "complete", **record}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
