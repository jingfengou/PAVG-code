#!/usr/bin/env python3
"""Run one fixed-split composite sample through the joint object/gas model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import DDIMScheduler
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.joint_pipeline import JointTrajPipeline  # noqa: E402
from train_interaction import load_runtime  # noqa: E402


def _load_checkpoint(model, checkpoint: Path, accelerator: Accelerator) -> None:
    """Load an Accelerate checkpoint, preferring its direct model state file."""
    model_files = sorted(
        [
            path
            for path in checkpoint.iterdir()
            if path.name.startswith("model") and path.suffix in {".bin", ".safetensors"}
        ]
    )
    if model_files:
        path = model_files[0]
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path), device="cpu")
        else:
            state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"checkpoint_state missing={len(missing)} unexpected={len(unexpected)}")
        return

    # Older Accelerate versions may only expose the state through load_state.
    prepared = accelerator.prepare(model)
    accelerator.load_state(str(checkpoint))
    model.load_state_dict(accelerator.unwrap_model(prepared).state_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--phase-mode",
        default="dataset",
        choices=["dataset", "gas", "solid", "interaction"],
        help="Use the dataset phase label or explicitly select active Stage 2 streams.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    mdm_model, dataset_mod, model_mod = load_runtime(cfg)
    dataset = dataset_mod.ObjectGasDataset(args.split, cfg)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"index {args.index} outside {args.split} length {len(dataset)}")

    accelerator = Accelerator(mixed_precision=str(cfg.mixed_precision))
    model = model_mod.JointMDM_ST(int(cfg.pc_size), int(cfg.n_training_frames), cfg.model_config, mdm_model)
    _load_checkpoint(model, Path(args.checkpoint), accelerator)
    device = accelerator.device
    model.to(device).eval()

    data, meta = dataset[args.index]
    batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in data.items()
        if isinstance(value, torch.Tensor)
    }
    if args.phase_mode != "dataset":
        presence_by_mode = {
            "gas": [True, False],
            "solid": [False, True],
            "interaction": [True, True],
        }
        batch["phase_presence"] = torch.tensor(
            [presence_by_mode[args.phase_mode]], device=device, dtype=torch.bool
        )
    pipeline = JointTrajPipeline(
        model,
        DDIMScheduler(num_train_timesteps=1000, prediction_type="sample", clip_sample=False),
    )
    prediction = pipeline(
        batch["points_src"],
        batch["init_vel"],
        batch["force"],
        batch["drag_point"],
        batch["drag_mask"],
        batch["physics_E"],
        batch["physics_nu"],
        batch["physics_mat_type"],
        batch["physics_floor_height"],
        batch["physics_gravity"],
        batch["physics_coeff"],
        batch["physics_volume"],
        batch["phase_presence"],
        device=device,
        generator=torch.Generator(device="cpu").manual_seed(args.seed),
        n_frames=int(cfg.n_training_frames),
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        prediction=prediction[0].float().cpu().numpy(),
        target=data["points_tgt"].numpy(),
        init_pc=data["points_src"].numpy(),
        phase_presence=batch["phase_presence"][0].cpu().numpy(),
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "split": args.split,
                "index": args.index,
                "phase_mode": args.phase_mode,
                "phase_presence": batch["phase_presence"][0].tolist(),
                "meta": meta,
            },
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
