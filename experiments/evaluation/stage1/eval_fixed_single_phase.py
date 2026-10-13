"""Evaluate a Stage 1 checkpoint on one locked single-phase manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from dataset.traj_dataset import TrajDataset
from model.spacetime import MDM_ST
from pipeline_traj import TrajPipeline
from utils.physctrl_metrics import macro_average, metric_protocol, sequence_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-mode", choices=("stream", "per-sample"), default="per-sample")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def select_manifest_range(dataset, manifest, start=0, end=None, max_samples=None):
    expected = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not isinstance(expected, list) or not all(isinstance(value, str) for value in expected):
        raise ValueError("Stage 1 manifest must be a JSON array of sample paths")
    models = {str(value["model"]): value for value in dataset.models}
    if len(models) != len(dataset.models) or len(set(expected)) != len(expected):
        raise ValueError("Duplicate dataset or manifest entries")
    if set(models) != set(expected):
        raise ValueError("Dataset membership does not match fixed manifest")
    end = len(expected) if end is None else end
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        end = min(end, start + max_samples)
    if not 0 <= start < end <= len(expected):
        raise ValueError(f"Invalid manifest range [{start}, {end})")
    dataset.models = [models[path] for path in expected[start:end]]
    return expected, start, end


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.train_dataset.dataset_list = args.test_manifest
    config.train_dataset.explicit_split_manifest = True

    dataset = TrajDataset("test", config.train_dataset)
    expected, start, end = select_manifest_range(
        dataset, args.test_manifest, args.start_index, args.end_index, args.max_samples,
    )
    if int(config.eval_batch_size) < 1:
        raise ValueError("eval_batch_size must be positive")
    if args.seed_mode == "per-sample" and int(config.eval_batch_size) != 1:
        raise ValueError("Per-sample seed mode requires eval_batch_size=1")
    if args.seed_mode == "stream" and start != 0:
        raise ValueError("Historical stream mode must start at manifest index 0")
    loader = DataLoader(
        dataset,
        batch_size=int(config.eval_batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cls = MDM_ST
    if str(config.get("model_type", "")) == "dit_st_phase":
        from model.phase_single import PhaseConditionedMDM_ST

        model_cls = PhaseConditionedMDM_ST
    model = model_cls(
        int(config.pc_size),
        int(config.train_dataset.n_training_frames),
        n_feats=3,
        model_config=config.model_config,
    ).to(device)
    state = load_file(args.checkpoint, device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    model.eval().requires_grad_(False)
    pipeline = TrajPipeline(
        model=model,
        scheduler=DDIMScheduler(
            num_train_timesteps=1000,
            prediction_type="sample",
            clip_sample=False,
        ),
    )
    pipeline.set_progress_bar_config(disable=True)

    records = []
    predictions, targets, sources = [], [], []
    # TrajPipeline samples on CPU before transferring to CUDA, so the
    # generator must remain a CPU generator for the remote PyTorch build.
    seed = (
        int(args.seed) if args.seed is not None
        else 20260820 if args.seed_mode == "per-sample"
        else int(config.seed)
    )
    generator = torch.Generator().manual_seed(seed)
    with torch.inference_mode():
        for batch, _ in loader:
            sample_index = start + len(records)
            if args.seed_mode == "per-sample":
                generator = torch.Generator().manual_seed(seed + sample_index)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction = pipeline(
                    batch["points_src"],
                    batch["force"],
                    batch["E"],
                    batch["nu"],
                    batch["mask"][..., :1],
                    batch["drag_point"],
                    batch["floor_height"],
                    batch["gravity"],
                    batch["base_drag_coeff"],
                    y=None if "mat_type" not in batch else batch["mat_type"],
                    init_vel=batch.get("init_vel"),
                    physics_volume=batch.get("physics_volume"),
                    device=device,
                    batch_size=prediction_batch_size(batch),
                    generator=generator,
                    n_frames=int(config.train_dataset.n_training_frames),
                    num_inference_steps=args.num_inference_steps,
                )
            target = batch["points_tgt"].to(device)
            for item in range(prediction.shape[0]):
                pred_array = prediction[item].float().cpu().numpy()
                target_array = target[item].float().cpu().numpy()
                if not np.isfinite(pred_array).all() or not np.isfinite(target_array).all():
                    raise ValueError(f"Non-finite arrays at index {sample_index + item}")
                record = sequence_metrics(
                    pred_array,
                    target_array,
                    include_velocity_mse=True,
                )
                record.update(
                    {
                        "index": sample_index + item,
                        "sample": str(batch["model"][item]),
                        "input_sha256": input_sha256(batch, item),
                        "target_sha256": hashlib.sha256(target_array.tobytes()).hexdigest(),
                    }
                )
                if record["sample"] != expected[record["index"]]:
                    raise ValueError("Sample order differs from the manifest")
                if args.seed_mode == "per-sample":
                    record["seed"] = seed + record["index"]
                records.append(record)
                if args.save_predictions:
                    predictions.append(pred_array)
                    targets.append(target_array)
                    sources.append(record["sample"])
            print(f"evaluated {len(records)}/{len(dataset)}", flush=True)

    metric_keys = ("vIoU", "CD", "Corr", "xyz_mse", "velocity_mse")
    aggregate = macro_average([{key: record[key] for key in metric_keys} for record in records])
    result = {
        "method": "Stage 1 single-phase evaluator",
        "dataset_label": args.dataset_label,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "test_manifest": args.test_manifest,
        "num_samples": len(records),
        "num_inference_steps": args.num_inference_steps,
        "sample_range": [start, end],
        "seed": seed,
        "seed_mode": args.seed_mode,
        "manifest_order_enforced": True,
        "test_manifest_sha256": file_sha256(args.test_manifest),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "config_sha256": file_sha256(args.config),
        "evaluator_sha256": file_sha256(__file__),
        "metric_protocol": metric_protocol(),
        "metrics": aggregate,
        "per_sample": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    if args.save_predictions:
        np.savez_compressed(
            output.with_suffix(".npz"), predictions=np.stack(predictions),
            targets=np.stack(targets), sources=np.asarray(sources),
            indices=np.asarray([row["index"] for row in records]),
        )
    print(json.dumps(result["metrics"], indent=2), flush=True)


def prediction_batch_size(batch: dict[str, object]) -> int:
    points = batch["points_src"]
    if not isinstance(points, torch.Tensor):
        raise TypeError("points_src must be a tensor")
    return int(points.shape[0])


def input_sha256(batch, item):
    digest = hashlib.sha256()
    for key in (
        "points_src", "init_vel", "physics_volume", "force", "E", "nu", "mask",
        "drag_point", "floor_height", "gravity", "base_drag_coeff", "mat_type",
    ):
        value = batch.get(key)
        if value is None:
            continue
        array = value[item].detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


if __name__ == "__main__":
    main()
