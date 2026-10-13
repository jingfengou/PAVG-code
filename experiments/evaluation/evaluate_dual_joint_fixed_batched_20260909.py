#!/usr/bin/env python3
"""Batched fixed-index Stage 2 evaluation with per-sample deterministic seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader, default_collate

from evaluate_dual_joint_fixed_20260905 import (
    _sample_metrics,
    _strip_module_prefix,
)
import fixed_stage2_results
from fixed_stage2_results import summarize_rows
from train_interaction import configure_attention_backend, load_runtime
from metrics import metric_protocol, sequence_metrics


LOGGER = logging.getLogger("evaluate_joint_batched")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_names(meta, batch_size: int) -> list[str]:
    if isinstance(meta, list):
        names = []
        for item in meta:
            value = item.get("file", "") if isinstance(item, dict) else ""
            if isinstance(value, (list, tuple)):
                value = value[0] if value else ""
            names.append(str(value))
        return names
    value = meta.get("file", "") if isinstance(meta, dict) else ""
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)] * batch_size


def _collate_samples(samples):
    """Batch tensors while retaining heterogeneous per-sample metadata."""
    data = default_collate([sample[0] for sample in samples])
    metadata = [sample[1] for sample in samples]
    return data, metadata


@torch.no_grad()
def _predict_batch(model, scheduler, batch, steps, device, seeds):
    target = batch["points_tgt"].to(device)
    source = batch["points_src"].to(device)
    init_vel = batch["init_vel"].to(device)
    force = batch["force"].to(device)
    drag_point = batch["drag_point"].to(device)
    drag_mask = batch["drag_mask"].to(device)
    physics_E = batch["physics_E"].to(device)
    physics_nu = batch["physics_nu"].to(device)
    physics_mat_type = batch["physics_mat_type"].to(device)
    physics_floor_height = batch["physics_floor_height"].to(device)
    physics_gravity = batch["physics_gravity"].to(device)
    physics_coeff = batch["physics_coeff"].to(device)
    physics_volume = batch["physics_volume"].to(device)
    phase_presence = batch["phase_presence"].to(device=device, dtype=torch.bool)
    if phase_presence.ndim != 2 or phase_presence.shape[1] != 2:
        raise ValueError(f"phase_presence must be [B,2], got {tuple(phase_presence.shape)}")

    noises = []
    for row, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(int(seed))
        noises.append(
            torch.randn(
                target[row : row + 1].shape,
                device=device,
                dtype=target.dtype,
                generator=generator,
            )
        )
    sample = torch.cat(noises, dim=0)

    scheduler.set_timesteps(steps, device=device)
    for timestep in scheduler.timesteps:
        t = torch.full(
            (target.shape[0],),
            int(timestep),
            device=device,
            dtype=torch.long,
        )
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            model_output = model(
                sample,
                t,
                source,
                init_vel,
                force,
                drag_point,
                drag_mask,
                physics_E=physics_E,
                physics_nu=physics_nu,
                physics_mat_type=physics_mat_type,
                physics_floor_height=physics_floor_height,
                physics_gravity=physics_gravity,
                physics_coeff=physics_coeff,
                physics_volume=physics_volume,
                phase_presence=phase_presence,
            )
        sample = scheduler.step(model_output.float(), timestep, sample).prev_sample
    return sample.float().cpu().numpy(), target.float().cpu().numpy(), source.float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--seed-index-offset",
        type=int,
        default=0,
        help=(
            "Global manifest-index offset used in the per-sample seed policy. "
            "Set this when evaluating a sliced manifest whose rows correspond "
            "to a later range of the canonical test manifest."
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, required=True)
    parser.add_argument(
        "--disable-cross-attention",
        action="store_true",
        help="Disable the learned solid-to-gas cross-attention at inference time.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions, targets, and source names beside the JSON result.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = OmegaConf.load(args.config)
    cfg.test_list = str(Path(args.test_manifest).resolve())
    configure_attention_backend("math")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mdm_model, dataset_mod, model_mod = load_runtime(cfg)
    dataset = dataset_mod.ObjectGasDataset("test", cfg)
    full_size = len(dataset.files)
    start = max(0, int(args.start_index))
    end = min(full_size, int(args.end_index))
    if not 0 <= start < end <= full_size:
        raise ValueError(f"invalid range [{start}, {end}) for dataset size {full_size}")
    dataset.files = dataset.files[start:end]
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=_collate_samples,
    )

    model = model_mod.JointMDM_ST(
        int(cfg.pc_size), int(cfg.n_training_frames), cfg.model_config, mdm_model
    ).to(device)
    state = _strip_module_prefix(load_file(args.checkpoint, device="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    model.set_cross_attention_enabled(not args.disable_cross_attention)
    model.eval()
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, prediction_type="sample", clip_sample=False
    )

    per_sample = []
    predictions = []
    targets = []
    initial_pcs = []
    sources = []
    processed = 0
    for batch, meta in loader:
        batch_size = int(batch["points_tgt"].shape[0])
        sample_indices = list(range(start + processed, start + processed + batch_size))
        seed_indices = [
            int(args.seed_index_offset) + sample_index for sample_index in sample_indices
        ]
        pred, target, source = _predict_batch(
            model,
            scheduler,
            batch,
            args.num_inference_steps,
            device,
            [args.seed + index for index in seed_indices],
        )
        names = _source_names(meta, batch_size)
        if args.save_predictions:
            predictions.extend(pred)
            targets.extend(target)
            initial_pcs.extend(source)
            sources.extend(names)
        presence_batch = batch["phase_presence"].bool().tolist()
        for row, sample_index in enumerate(sample_indices):
            presence = [bool(value) for value in presence_batch[row]]
            has_gas, has_solid = presence
            if not (has_gas or has_solid):
                raise RuntimeError(f"sample {sample_index} has no active phase")
            gas_metrics = None
            solid_metrics = None
            interaction_metrics = None
            if has_gas:
                gas_metrics = _sample_metrics(
                    pred[row, :, 0], target[row, :, 0], source[row, 0]
                )
            if has_solid:
                solid_metrics = _sample_metrics(
                    pred[row, :, 1], target[row, :, 1], source[row, 1]
                )
            if has_gas and has_solid:
                interaction_metrics = sequence_metrics(
                    np.concatenate([pred[row, :, 0], pred[row, :, 1]], axis=1),
                    np.concatenate([target[row, :, 0], target[row, :, 1]], axis=1),
                    include_velocity_mse=True,
                )
                interaction_solid_metrics = _sample_metrics(
                    pred[row, :, 1], target[row, :, 1], source[row, 1]
                )
                interaction_gas_metrics = _sample_metrics(
                    pred[row, :, 0], target[row, :, 0], source[row, 0]
                )
            per_sample.append(
                {
                    "index": sample_index,
                    "seed": args.seed + seed_indices[row],
                    "source": names[row],
                    "phase_presence": presence,
                    "gas": gas_metrics,
                    "object": solid_metrics,
                    "interaction": interaction_metrics,
                    "interaction_gas": (
                        interaction_gas_metrics
                        if has_gas and has_solid
                        else None
                    ),
                    "interaction_solid": (
                        interaction_solid_metrics
                        if has_gas and has_solid
                        else None
                    ),
                }
            )
        processed += batch_size
        LOGGER.info("evaluated %d/%d", processed, len(dataset))

    result = {
        "method": "PAVG fixed Stage 2 evaluator",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "evaluator_sha256": _sha256(__file__),
        "metrics_sha256": _sha256(
            Path(__file__).resolve().parents[2] / "src/utils/physctrl_metrics.py"
        ),
        "aggregation_sha256": _sha256(fixed_stage2_results.__file__),
        "manifest": str(args.test_manifest),
        "manifest_sha256": _sha256(args.test_manifest),
        "split": "fixed source-disjoint full test shard",
        "num_samples": len(per_sample),
        "sample_range": [start, end],
        "num_inference_steps": args.num_inference_steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "seed_index_offset": int(args.seed_index_offset),
        "seed_policy": "seed_plus_seed_index_offset_plus_manifest_index",
        "cross_attention_enabled": not args.disable_cross_attention,
        "variant": (
            "w/o solid-to-gas Cross-Attention"
            if args.disable_cross_attention
            else "full PAVG"
        ),
        "metric_protocol": metric_protocol(),
        **summarize_rows(per_sample),
        "per_sample": per_sample,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.save_predictions:
        np.savez_compressed(
            output.with_suffix(".npz"),
            predictions=np.asarray(predictions, dtype=np.float32),
            targets=np.asarray(targets, dtype=np.float32),
            init_pc=np.asarray(initial_pcs, dtype=np.float32),
            sources=np.asarray(sources),
            indices=np.asarray([row["index"] for row in per_sample]),
        )
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
