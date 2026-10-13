"""Evaluate a dual-backbone PhysCtrl gas/object checkpoint on the full test split.

The evaluator uses only the observable inputs consumed by training: the two
initial point clouds, their per-particle initial velocities, and each stream's
force, drag point, and drag mask.  It performs one joint model call per DDIM
step and reports metrics separately for gas, object, and their macro average.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from train_interaction import configure_attention_backend, load_runtime
from metrics import macro_average, metric_protocol, sequence_metrics


LOGGER = logging.getLogger("evaluate_joint")
METRIC_KEYS = ("vIoU", "CD", "Corr", "xyz_mse", "velocity_mse")


def _strip_module_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def _initial_pairs(x0: np.ndarray, max_pairs: int = 8192) -> np.ndarray:
    from scipy.spatial import cKDTree

    if len(x0) < 2:
        return np.empty((0, 2), dtype=np.int64)
    k = min(5, len(x0))
    _, neighbours = cKDTree(x0).query(x0, k=k)
    pairs = np.stack(
        [np.repeat(np.arange(len(x0)), k - 1), neighbours[:, 1:].reshape(-1)], axis=1
    )
    return pairs[:max_pairs].astype(np.int64)


def _sample_metrics(
    pred: np.ndarray, target: np.ndarray, init: np.ndarray
) -> dict[str, float]:
    """Compute metrics for one stream with pred/target [T,N,3]."""
    values = sequence_metrics(pred, target, include_velocity_mse=True)
    pairs = _initial_pairs(init)
    deformation = []
    gt_deformation = []
    for frame in range(pred.shape[0]):
        p, g = pred[frame], target[frame]
        if pairs.size:
            p0, p1 = init[pairs[:, 0]], init[pairs[:, 1]]
            d0 = np.linalg.norm(p1 - p0, axis=1).clip(min=1e-6)
            pred_rel = np.linalg.norm(p[pairs[:, 1]] - p[pairs[:, 0]], axis=1)
            target_rel = np.linalg.norm(g[pairs[:, 1]] - g[pairs[:, 0]], axis=1)
            deformation.append(float(np.mean(np.abs(pred_rel - target_rel) / d0)))
            gt_deformation.append(float(np.mean(np.abs(target_rel - d0) / d0)))
        else:
            deformation.append(0.0)
            gt_deformation.append(0.0)
    values["deform"] = float(np.mean(deformation))
    values["gt_deform"] = float(np.mean(gt_deformation))
    return values


def _mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    return macro_average(records)


def _weighted_overall(groups: dict[str, dict[str, float]]) -> dict[str, float]:
    if not all(groups.get(name) for name in ("solid", "gas", "interaction")):
        return {}
    # Interaction metrics intentionally omit stream-only deformation helpers;
    # aggregate only keys defined for every benchmark group.
    keys = set(groups["solid"]) & set(groups["gas"]) & set(groups["interaction"])
    return {
        key: float(
            0.25 * groups["solid"][key]
            + 0.25 * groups["gas"][key]
            + 0.50 * groups["interaction"][key]
        )
        for key in sorted(keys)
    }


@torch.no_grad()
def _predict(
    model,
    scheduler,
    batch: dict[str, torch.Tensor],
    steps: int,
    device: torch.device,
    seed: int,
):
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
    if "phase_presence" not in batch:
        raise KeyError(
            "Stage 2 evaluation requires phase_presence for every sample; "
            "refusing to infer a default dual-phase path"
        )
    phase_presence = batch["phase_presence"].to(device=device, dtype=torch.bool)
    if phase_presence.ndim != 2 or phase_presence.shape[1] != 2:
        raise ValueError(
            f"phase_presence must have shape [B,2], got {tuple(phase_presence.shape)}"
        )
    if not bool(phase_presence.any(dim=1).all()):
        raise ValueError("phase_presence contains a sample with no active phase")
    sample = torch.randn(
        target.shape,
        device=device,
        dtype=target.dtype,
        generator=torch.Generator(device=device).manual_seed(int(seed)),
    )
    scheduler.set_timesteps(steps, device=device)
    for timestep in scheduler.timesteps:
        t = torch.full((target.shape[0],), int(timestep), device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
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
    parser.add_argument(
        "--test-manifest",
        default=None,
        help="Optional fixed manifest override used for validation or audit runs.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = OmegaConf.load(args.config)
    if args.test_manifest is not None:
        cfg.test_list = str(Path(args.test_manifest).resolve())
    configure_attention_backend("math")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdm_model, dataset_mod, model_mod = load_runtime(cfg)
    dataset = dataset_mod.ObjectGasDataset("test", cfg)
    full_dataset_size = len(dataset.files)
    start_index = max(0, int(args.start_index))
    end_index = full_dataset_size if args.end_index is None else int(args.end_index)
    if args.max_samples is not None:
        end_index = min(end_index, start_index + int(args.max_samples))
    if start_index > end_index or end_index > full_dataset_size:
        raise ValueError(
            f"invalid sample range [{start_index}, {end_index}) for test size {full_dataset_size}"
        )
    dataset.files = dataset.files[start_index:end_index]
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = model_mod.JointMDM_ST(
        int(cfg.pc_size), int(cfg.n_training_frames), cfg.model_config, mdm_model
    ).to(device)
    state = _strip_module_prefix(load_file(args.checkpoint, device="cpu"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}")
    model.eval()
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, prediction_type="sample", clip_sample=False
    )

    stream_records = {"gas": [], "object": []}
    interaction_records = []
    per_sample = []
    predictions, targets, sources = [], [], []
    for index, (batch, meta) in enumerate(loader):
        sample_index = start_index + index
        # Make sharded evaluation reproducible with the corresponding sample
        # in an unsharded run.
        torch.manual_seed(args.seed + sample_index)
        np.random.seed(args.seed + sample_index)
        pred, target, source = _predict(
            model,
            scheduler,
            batch,
            args.num_inference_steps,
            device,
            args.seed + sample_index,
        )
        presence = batch["phase_presence"]
        presence = presence[0].bool().tolist()
        has_gas, has_solid = bool(presence[0]), bool(presence[1])
        sample_stream_metrics = {"gas": None, "solid": None, "interaction": None}
        if has_gas:
            sample_stream_metrics["gas"] = _sample_metrics(
                pred[0, :, 0], target[0, :, 0], source[0, 0]
            )
            stream_records["gas"].append(sample_stream_metrics["gas"])
        if has_solid:
            sample_stream_metrics["solid"] = _sample_metrics(
                pred[0, :, 1], target[0, :, 1], source[0, 1]
            )
            stream_records["object"].append(sample_stream_metrics["solid"])
        if has_gas and has_solid:
            interaction_prediction = np.concatenate(
                [pred[0, :, 0], pred[0, :, 1]], axis=1
            )
            interaction_target = np.concatenate(
                [target[0, :, 0], target[0, :, 1]], axis=1
            )
            sample_stream_metrics["interaction"] = sequence_metrics(
                interaction_prediction,
                interaction_target,
                include_velocity_mse=True,
            )
            interaction_records.append(sample_stream_metrics["interaction"])
        source_name = meta["file"] if isinstance(meta, dict) and "file" in meta else ""
        if isinstance(source_name, (list, tuple)):
            source_name = source_name[0]
        per_sample.append(
            {
                "index": sample_index,
                "source": str(source_name),
                "gas": sample_stream_metrics["gas"],
                "object": sample_stream_metrics["solid"],
                "interaction": sample_stream_metrics["interaction"],
            }
        )
        if args.save_predictions:
            predictions.append(pred[0])
            targets.append(target[0])
            sources.append(source_name)
        if (index + 1) % 10 == 0:
            LOGGER.info("evaluated %d/%d", index + 1, len(dataset))

    gas = _mean_metrics(stream_records["gas"])
    obj = _mean_metrics(stream_records["object"])
    interaction = _mean_metrics(interaction_records)
    overall = _weighted_overall(
        {"gas": gas, "solid": obj, "interaction": interaction}
    )
    result = {
        "method": "PhysCtrl dual-backbone joint gas/object",
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "split": "fixed source-disjoint full test",
        "num_samples": len(dataset),
        "sample_range": [start_index, end_index],
        "num_inference_steps": args.num_inference_steps,
        "metric_protocol": metric_protocol(),
        "metrics": {
            "interaction": interaction,
            "gas_stream_diagnostic": gas,
            "solid_stream_diagnostic": obj,
            "overall": overall,
        },
        "overall_aggregation": (
            "0.5 * interaction + 0.25 * solid + 0.25 * gas; "
            "group means are computed before weighting"
        ),
        "per_sample": per_sample,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.save_predictions:
        np.savez_compressed(
            output.with_suffix(".npz"),
            predictions=np.asarray(predictions, dtype=np.float32),
            targets=np.asarray(targets, dtype=np.float32),
            sources=np.asarray(sources),
        )
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
