from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, default_collate

from physics_losses import DeformLoss
from exposure_plan import ExposurePlanDataset
from performance_options import (
    aggregate_nonfinite, configure_model_performance, configure_restored_optimizer,
    loader_performance_kwargs,
)


ROOT = Path(__file__).resolve().parent


def build_cosine_scheduler(optimizer, warmup, max_steps, world_size, split_batches=False):
    # AcceleratedScheduler advances once per process when split_batches=False.
    # Match PhysCtrl's horizon convention in optimizer-update units.
    multiplier = 1 if split_batches else int(world_size)
    return get_cosine_schedule_with_warmup(
        optimizer, int(warmup) * multiplier, int(max_steps) * multiplier
    )


def configure_attention_backend(backend: str = "default"):
    if not torch.cuda.is_available():
        return
    if backend == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif backend == "flash":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
    torch.set_float32_matmul_precision("high")


def nonfinite_gradients(model, limit: int = 8, aggregate: bool = False):
    if aggregate:
        return aggregate_nonfinite(
            ((n, p.grad) for n, p in model.named_parameters()), limit,
            implementation=aggregate if isinstance(aggregate, str) else "elementwise")
    bad = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            bad.append(name)
            if len(bad) >= limit:
                break
    return bad


def nonfinite_parameters(model, limit: int = 8, aggregate: bool = False):
    if aggregate:
        return aggregate_nonfinite(
            model.named_parameters(), limit,
            implementation=aggregate if isinstance(aggregate, str) else "elementwise")
    bad = []
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            bad.append(name)
            if len(bad) >= limit:
                break
    return bad


def batch_files(meta):
    if not isinstance(meta, dict):
        return []
    files = meta.get("file", [])
    if isinstance(files, (list, tuple)):
        return [str(path) for path in files]
    return [str(files)]


def collate_training_batch(items):
    # object_* compatibility aliases occur only in interaction samples.
    # The trainer consumes their canonical physics_* equivalents.
    data, metadata = zip(*items)
    tensors = [{key: value for key, value in item.items() if not key.startswith("object_")}
               for item in data]
    result = {"file": [item["file"] for item in metadata]}
    for key in ("exposure_source", "exposure_position", "phase_slot", "exposure_valid"):
        if key in metadata[0]:
            result[key] = [item[key] for item in metadata]
    return default_collate(tensors), result


def any_rank_bad(accelerator, local_bad: bool, device: torch.device) -> bool:
    flag = torch.tensor(1 if local_bad else 0, device=device, dtype=torch.int32)
    return int(accelerator.reduce(flag, reduction="sum").item()) > 0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runtime(cfg):
    mdm_root = Path(str(cfg.mdm_root))
    sys.path.insert(0, str(mdm_root))
    mdm_model = load_module(mdm_root / "model/spacetime.py", "physctrl_mdm_spacetime")
    # The checked-in layout uses interaction/{dataset,model}; the AFS
    # training snapshot keeps the same modules under interaction_dataset and
    # interaction_model. Resolve both layouts so the exact same implementation
    # can be audited locally and run remotely.
    model_root = ROOT / "model"
    dataset_root = ROOT / "dataset"
    if not model_root.is_dir():
        model_root = ROOT / "interaction_model"
    if not dataset_root.is_dir():
        dataset_root = ROOT / "interaction_dataset"
    interaction_spacetime = load_module(
        model_root / "spacetime_interaction.py",
        "physctrl_mdm_spacetime_interaction",
    )
    interaction_spacetime.configure(mdm_model)
    mdm_model = interaction_spacetime
    dataset_mod = load_module(
        dataset_root / "object_gas_dataset.py", "physctrl_object_gas_dataset"
    )
    model_mod = load_module(
        model_root / "joint_model.py", "physctrl_joint_model"
    )
    solid_mod = load_module(model_root / "solid_stage1.py", "physctrl_solid_stage1")
    model_mod.SolidStage1 = solid_mod.SolidStage1
    return mdm_model, dataset_mod, model_mod


def build_model(cfg, mdm_model, model_mod):
    if bool(cfg.model_config.get("solid_stage1_only", False)):
        return model_mod.SolidStage1(
            int(cfg.pc_size), int(cfg.n_training_frames), cfg.model_config, mdm_model
        )
    return model_mod.JointMDM_ST(
        int(cfg.pc_size), int(cfg.n_training_frames), cfg.model_config, mdm_model
    )


def diffusion_inputs(batch, scheduler):
    target = torch.nan_to_num(batch["points_tgt"].float(), nan=0.0, posinf=0.0, neginf=0.0)
    noise = torch.randn_like(target)
    phase_presence = batch.get("phase_presence")
    if phase_presence is not None:
        if phase_presence.ndim != 2 or phase_presence.shape != (target.shape[0], 2):
            raise ValueError(
                "phase_presence must have shape [B,2] for diffusion inputs"
            )
        # Keep absent streams out of the diffusion path. They are represented
        # by zero-filled tensors and are excluded from the trajectory loss.
        noise = noise * phase_presence[:, None, :, None, None].to(noise.dtype)
    batch_size = target.shape[0]
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (batch_size,), device=target.device
    ).long()
    noisy = scheduler.add_noise(target, noise, timesteps)
    return target, noisy, timesteps


def _masked_mean(values: torch.Tensor, valid: torch.Tensor, graph_tensor: torch.Tensor):
    if valid.any():
        return values[valid].mean()
    return graph_tensor.sum() * 0.0


def _masked_rms(
    values: torch.Tensor,
    valid: torch.Tensor,
    graph_tensor: torch.Tensor,
    minimum: float,
) -> torch.Tensor:
    if valid.any():
        scale = values[valid].square().mean().sqrt().detach()
        return scale.clamp_min(float(minimum))
    return graph_tensor.sum() * 0.0 + float(minimum)


def compute_gas_advection_loss(pred: torch.Tensor, batch, cfg) -> torch.Tensor:
    """Enforce the MantaFlow tracer update over each sampled interval."""
    required = (
        "points_src",
        "gas_advection_displacement",
        "gas_advection_alive",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(
            "Gas advection loss requires dataset fields: " + ", ".join(missing)
        )

    gas_position = pred[:, :, 0].float()
    initial_position = batch["points_src"][:, 0].float().unsqueeze(1)
    previous_position = torch.cat([initial_position, gas_position[:, :-1]], dim=1)
    predicted_displacement = gas_position - previous_position
    target_displacement = batch["gas_advection_displacement"].float()
    valid = batch["gas_advection_alive"].bool()
    if "sample_valid" in batch:
        valid = valid & batch["sample_valid"].bool()[:, None, None]
    scale = _masked_rms(
        target_displacement,
        valid,
        pred,
        float(cfg.get("gas_advection_scale_min", 1.0e-4)),
    )
    error = ((predicted_displacement - target_displacement) / scale).square().mean(
        dim=-1
    )
    return _masked_mean(error, valid, pred)


def compute_gas_divergence_loss(pred: torch.Tensor, batch, cfg) -> torch.Tensor:
    """Match the point-cloud divergence proxy induced by the gas trajectory."""
    required = ("points_src", "points_tgt", "gas_advection_alive")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(
            "Gas divergence loss requires dataset fields: " + ", ".join(missing)
        )

    gas_pred = pred[:, :, 0].float() if pred.ndim == 5 else pred.float()
    source = batch["points_src"]
    target = batch["points_tgt"]
    source = source[:, 0] if source.ndim == 4 else source
    target = target[:, :, 0] if target.ndim == 5 else target
    source = source.float()
    target = target.float()
    alive = batch["gas_advection_alive"].bool()
    if "sample_valid" in batch:
        alive = alive & batch["sample_valid"].bool()[:, None, None]
    if "phase_presence" in batch:
        alive = alive & batch["phase_presence"][:, 0].bool()[:, None, None]
    if not bool(alive.any()):
        return pred.sum() * 0.0

    batch_size, num_points, _ = source.shape
    frame_count = gas_pred.shape[1]
    if target.shape[1] != frame_count:
        raise ValueError(
            "Gas divergence target and prediction frame counts differ: "
            f"{target.shape[1]} vs {frame_count}"
        )
    center_count = min(
        int(cfg.get("gas_divergence_max_centers", 256)), num_points
    )
    neighbor_count = min(
        int(cfg.get("gas_divergence_num_neighbors", 8)), num_points - 1
    )
    if center_count < 1 or neighbor_count < 3:
        return pred.sum() * 0.0

    center_idx = torch.linspace(
        0, num_points - 1, center_count, device=source.device
    ).round().long()
    with torch.no_grad():
        distances = torch.cdist(source[:, center_idx], source)
        self_idx = center_idx.reshape(1, -1, 1).expand(batch_size, -1, 1)
        distances.scatter_(2, self_idx, float("inf"))
        neighbor_idx = distances.topk(
            neighbor_count, dim=2, largest=False, sorted=False
        ).indices

    def estimate_divergence(positions):
        positions = positions.float()
        displacements = positions[:, 1:] - positions[:, :-1]
        time_count = displacements.shape[1]
        time_neighbor_idx = neighbor_idx[:, None].expand(
            batch_size, time_count, center_count, neighbor_count
        )

        def gather_neighbors(values):
            values = values.unsqueeze(2).expand(
                -1, -1, center_count, -1, -1
            )
            index = time_neighbor_idx.unsqueeze(-1).expand(
                -1, -1, -1, -1, 3
            )
            return values.gather(3, index)

        current_positions = positions[:, :-1]
        center_positions = current_positions.index_select(2, center_idx)
        neighbor_positions = gather_neighbors(current_positions)
        center_displacements = displacements.index_select(2, center_idx)
        neighbor_displacements = gather_neighbors(displacements)
        delta_x = neighbor_positions - center_positions.unsqueeze(3)
        delta_u = neighbor_displacements - center_displacements.unsqueeze(3)

        interval_alive = alive[:, :time_count]
        center_alive = interval_alive.index_select(2, center_idx)
        neighbor_alive = interval_alive.unsqueeze(2).expand(
            -1, -1, center_count, -1
        ).gather(3, time_neighbor_idx)
        edge_valid = neighbor_alive & center_alive.unsqueeze(-1)
        delta_x = delta_x * edge_valid.unsqueeze(-1).float()
        delta_u = delta_u * edge_valid.unsqueeze(-1).float()

        normal = torch.einsum("btmki,btmkj->btmij", delta_x, delta_x)
        rhs = torch.einsum("btmki,btmkj->btmij", delta_x, delta_u)
        eye = torch.eye(3, device=normal.device, dtype=normal.dtype)
        normal = normal + float(
            cfg.get("gas_divergence_regularization", 1.0e-3)
        ) * eye.reshape(1, 1, 1, 3, 3)
        jacobian = torch.linalg.solve(normal, rhs)
        divergence = torch.diagonal(jacobian, dim1=-2, dim2=-1).sum(dim=-1)
        valid_centers = center_alive & (edge_valid.sum(dim=-1) >= 3)
        return divergence, valid_centers

    predicted_positions = torch.cat([source.unsqueeze(1), gas_pred], dim=1)
    target_positions = torch.cat([source.unsqueeze(1), target], dim=1)
    predicted_divergence, valid_centers = estimate_divergence(predicted_positions)
    with torch.no_grad():
        target_divergence, target_valid_centers = estimate_divergence(target_positions)
    valid_centers = valid_centers & target_valid_centers
    if not bool(valid_centers.any()):
        return pred.sum() * 0.0

    scale = _masked_rms(
        target_divergence,
        valid_centers,
        pred,
        float(cfg.get("gas_divergence_scale_min", 1.0)),
    )
    residual = (
        predicted_divergence - target_divergence.detach()
    ) / scale
    clip = float(cfg.get("gas_divergence_residual_clip", 20.0))
    residual = residual.clamp(-clip, clip)
    per_center = F.smooth_l1_loss(
        residual,
        torch.zeros_like(residual),
        beta=float(cfg.get("gas_divergence_huber_beta", 1.0)),
        reduction="none",
    )
    weights = valid_centers.float()
    return (per_center * weights).sum() / weights.sum().clamp_min(1.0)


def compute_velocity_transfer_loss(
    pred: torch.Tensor, batch, cfg
) -> torch.Tensor:
    """Supervise object-to-gas transfer in the sampled-frame discretization."""
    total_key = (
        "interaction_total_delta_v"
        if "interaction_total_delta_v" in batch
        else "gas_velocity_delta"
    )
    transfer_key = (
        "interaction_transfer_delta_v"
        if "interaction_transfer_delta_v" in batch
        else "interaction_delta_v"
    )
    required = (
        "points_src",
        "gas_initial_velocity",
        total_key,
        transfer_key,
        "interaction_alive",
        "interaction_valid",
        "interaction_dt",
        "sample_frame_steps",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(
            "Velocity-transfer loss requires dataset fields: " + ", ".join(missing)
        )

    gas_position = pred[:, :, 0].float()
    initial_position = batch["points_src"][:, 0].float().unsqueeze(1)
    previous_position = torch.cat([initial_position, gas_position[:, :-1]], dim=1)
    duration = (
        batch["interaction_dt"].float()[:, None]
        * batch["sample_frame_steps"].float()
    ).clamp_min(1.0e-8)
    predicted_velocity = (
        gas_position - previous_position
    ) / duration[..., None, None]
    previous_velocity = torch.cat(
        [
            batch["gas_initial_velocity"].float().unsqueeze(1),
            predicted_velocity[:, :-1],
        ],
        dim=1,
    )
    predicted_total_delta_v = predicted_velocity - previous_velocity

    target_transfer = batch[transfer_key].float()
    target_total_delta_v = batch[total_key].float()
    target_noninteraction_delta_v = target_total_delta_v - target_transfer
    predicted_transfer = predicted_total_delta_v - target_noninteraction_delta_v

    valid = batch["interaction_alive"].bool()
    valid = valid & batch["interaction_valid"].bool()[..., None]
    if "sample_valid" in batch:
        valid = valid & batch["sample_valid"].bool()[:, None, None]
    if "phase_presence" in batch:
        phase_presence = batch["phase_presence"].bool()
        if phase_presence.ndim != 2 or phase_presence.shape[1] != 2:
            raise ValueError(
                "phase_presence must have shape [B,2] for coupling loss"
            )
        valid = valid & (phase_presence[:, 0] & phase_presence[:, 1])[:, None, None]
    threshold = float(cfg.get("coupling_min_magnitude", 0.0))
    if threshold > 0.0:
        valid = valid & (target_transfer.norm(dim=-1) >= threshold)
    configured_scale = float(cfg.get("coupling_target_scale", 0.0))
    scale = (
        target_transfer.new_tensor(configured_scale)
        if configured_scale > 0.0
        else _masked_rms(
            target_transfer,
            valid,
            pred,
            float(cfg.get("coupling_scale_min", 1.0e-4)),
        )
    )
    beta = float(cfg.get("coupling_huber_beta", 1.0))
    error = F.smooth_l1_loss(
        predicted_transfer / scale,
        target_transfer / scale,
        reduction="none",
        beta=beta,
    ).mean(dim=-1)
    return _masked_mean(error, valid, pred)


def compute_loss(model, batch, scheduler, cfg, deform_loss=None, prediction=None):
    if bool(cfg.get("require_phase_presence", False)) and "phase_presence" not in batch:
        raise KeyError(
            "Combined Stage 2 training requires phase_presence in every batch"
        )
    if prediction is None:
        target, noisy, timesteps = diffusion_inputs(batch, scheduler)
        pred = model(
            noisy,
            timesteps,
            batch["points_src"],
            batch["init_vel"],
            batch["force"],
            batch["drag_point"],
            batch["drag_mask"],
            physics_E=batch.get("physics_E"),
            physics_nu=batch.get("physics_nu"),
            physics_mat_type=batch.get("physics_mat_type"),
            physics_floor_height=batch.get("physics_floor_height"),
            physics_gravity=batch.get("physics_gravity"),
            physics_coeff=batch.get("physics_coeff"),
            physics_volume=batch.get("physics_volume"),
            phase_presence=batch.get("phase_presence"),
        )
    else:
        target = torch.nan_to_num(batch["points_tgt"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        pred, noisy, timesteps = prediction, None, None
    if bool(cfg.get("samplewise_loss_reduction", False)) and pred.shape[0] > 1:
        # Match the reference's batch-1 phase weights and physical normalizers,
        # while retaining one batched model forward/backward.
        terms = [
            compute_loss(
                None, {key: value[i:i + 1] for key, value in batch.items()},
                scheduler, cfg, deform_loss, prediction=pred[i:i + 1],
            )
            for i in range(pred.shape[0])
        ]
        loss = torch.stack([item[0] for item in terms]).mean()
        metrics = {}
        for key in terms[0][1]:
            values = torch.stack([item[1][key] for item in terms])
            metrics[key] = values.sum() if key in {"invalid_gas", "invalid_object"} else values.mean()
        return loss, metrics

    pred_float = pred.float()
    target_float = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
    # [B,T,2,N,3] -> one validity flag per sample and stream.
    pred_finite = torch.isfinite(pred_float).all(dim=(1, 3, 4))
    safe_pred = torch.nan_to_num(pred_float, nan=0.0, posinf=0.0, neginf=0.0)
    xyz_per_sample = (safe_pred - target_float).square().mean(dim=(1, 3, 4))

    target_velocity = target[:, 1:] - target[:, :-1]
    pred_velocity = safe_pred[:, 1:] - safe_pred[:, :-1]
    velocity_per_sample = (pred_velocity - target_velocity.float()).square().mean(dim=(1, 3, 4))
    sample_valid = batch.get("sample_valid", torch.ones(target.shape[0], device=target.device, dtype=torch.bool))
    valid = pred_finite & torch.isfinite(xyz_per_sample) & torch.isfinite(velocity_per_sample) & sample_valid[:, None]
    if "phase_presence" in batch:
        phase_presence = batch["phase_presence"].bool()
        if phase_presence.shape != (target.shape[0], 2):
            raise ValueError(
                "phase_presence must have shape [B,2] in trajectory loss"
            )
        valid = valid & phase_presence

    gas_xyz = _masked_mean(xyz_per_sample[:, 0], valid[:, 0], safe_pred)
    object_xyz = _masked_mean(xyz_per_sample[:, 1], valid[:, 1], safe_pred)
    gas_vel = _masked_mean(velocity_per_sample[:, 0], valid[:, 0], safe_pred)
    object_vel = _masked_mean(velocity_per_sample[:, 1], valid[:, 1], safe_pred)

    gas_weight = float(cfg.gas_weight)
    object_weight = float(cfg.object_weight)
    gas_loss = float(cfg.lambda_xyz) * gas_xyz + float(cfg.lambda_vel) * gas_vel
    object_loss = float(cfg.lambda_xyz) * object_xyz + float(cfg.lambda_vel) * object_vel
    # Do not dilute a single-phase sample by the absent branch. Within a
    # mixed batch, each branch keeps its configured weight and is averaged
    # over the valid samples that actually contain that phase.
    branch_present = valid.any(dim=0)
    active_weight = (
        gas_weight * branch_present[0].to(gas_loss.dtype)
        + object_weight * branch_present[1].to(object_loss.dtype)
    ).clamp_min(1.0e-8)
    trajectory_loss = (
        gas_weight * gas_loss * branch_present[0].to(gas_loss.dtype)
        + object_weight * object_loss * branch_present[1].to(object_loss.dtype)
    ) / active_weight
    physical_deform_loss = safe_pred.sum() * 0.0
    if float(cfg.get("lambda_deform", 0.0)) > 0.0:
        if deform_loss is None:
            raise RuntimeError("lambda_deform > 0 requires a DeformLoss instance")
        required = ("physics_volume", "physics_F", "physics_C")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "Solid deformation loss requires dataset fields: " + ", ".join(missing)
            )
        solid_valid = valid[:, 1]
        if bool(cfg.get("solid_stage1_mpm_only", False)):
            solid_valid = solid_valid & (batch["physics_mat_type"][:, 1] != 3)
        if bool(solid_valid.any()):
            physical_deform_loss = deform_loss(
                safe_pred[solid_valid, :, 1].clamp(
                    min=-float(cfg.get("deform_prediction_clip", 2.2)),
                    max=float(cfg.get("deform_prediction_clip", 2.2)),
                ),
                batch["physics_volume"][solid_valid, 1, :, 0],
                batch["physics_F"][solid_valid, 1],
                batch["physics_C"][solid_valid, 1],
                frame_interval=int(cfg.n_frames_interval),
                norm_fac=float(cfg.norm_fac),
            )
    physical_floor_loss = safe_pred.sum() * 0.0
    if float(cfg.get("lambda_floor", 0.0)) > 0.0:
        if "physics_floor_height" not in batch:
            raise KeyError("Floor loss requires physics_floor_height")
        solid_valid = valid[:, 1]
        if bool(solid_valid.any()):
            floor_height = batch["physics_floor_height"][:, 1, 0].float()
            target_min = target_float[:, :, 1, :, 1].amin(dim=(1, 2))
            floor_height = torch.minimum(floor_height, target_min)
            floor_error = (
                torch.relu(
                    floor_height[:, None, None]
                    - safe_pred[:, :, 1, :, 1]
                ).square().mean(dim=(1, 2))
            )
            physical_floor_loss = _masked_mean(
                floor_error, solid_valid, safe_pred
            )
    lambda_gas_divergence = float(cfg.get("lambda_gas_divergence", 0.0))
    gas_divergence_loss = (
        compute_gas_divergence_loss(safe_pred, batch, cfg)
        if lambda_gas_divergence > 0.0
        else safe_pred.sum() * 0.0
    )
    lambda_gas_advection = float(cfg.get("lambda_gas_advection", 0.0))
    gas_advection_loss = (
        compute_gas_advection_loss(safe_pred, batch, cfg)
        if lambda_gas_advection > 0.0
        else safe_pred.sum() * 0.0
    )
    lambda_coupling = float(cfg.get("lambda_coupling", 0.0))
    coupling_loss = (
        compute_velocity_transfer_loss(safe_pred, batch, cfg)
        if lambda_coupling > 0.0
        else safe_pred.sum() * 0.0
    )
    # Average complete phase objectives; absent branches must not dilute them.
    solid_physics = (
        float(cfg.get("lambda_deform", 0.0)) * physical_deform_loss
        + float(cfg.get("lambda_floor", 0.0)) * physical_floor_loss
    )
    gas_physics = (
        lambda_gas_divergence * gas_divergence_loss
        + lambda_gas_advection * gas_advection_loss
    )
    trajectory_loss = trajectory_loss + (
        object_weight * solid_physics * branch_present[1].to(solid_physics.dtype)
        + gas_weight * gas_physics * branch_present[0].to(gas_physics.dtype)
    ) / active_weight
    loss = trajectory_loss + lambda_coupling * coupling_loss
    unweighted_loss = loss
    if "phase_loss_weight" in batch:
        if pred.shape[0] != 1:
            raise ValueError("Phase sample weighting requires samplewise_loss_reduction=true")
        loss = loss * batch["phase_loss_weight"].reshape(-1)[0]

    metrics = {
        "loss": loss.detach(),
        "loss_unweighted": unweighted_loss.detach(),
        "xyz": (
            gas_weight * gas_xyz * branch_present[0].to(gas_loss.dtype)
            + object_weight * object_xyz * branch_present[1].to(object_loss.dtype)
        ).div(active_weight).detach(),
        "xyz_gas": gas_xyz.detach(),
        "xyz_object": object_xyz.detach(),
        "loss_vel": (
            gas_weight * gas_vel * branch_present[0].to(gas_loss.dtype)
            + object_weight * object_vel * branch_present[1].to(object_loss.dtype)
        ).div(active_weight).detach(),
        "loss_vel_gas": gas_vel.detach(),
        "loss_vel_object": object_vel.detach(),
        "loss_deform": physical_deform_loss.detach(),
        "loss_floor": physical_floor_loss.detach(),
        "loss_gas_divergence": gas_divergence_loss.detach(),
        "loss_gas_advection": gas_advection_loss.detach(),
        "loss_coupling": coupling_loss.detach(),
        # An inactive stream is expected to be invalid by construction for a
        # single-phase sample; report only active-stream failures.
        "invalid_gas": ((~valid[:, 0]) & phase_presence[:, 0]).sum().detach()
        if "phase_presence" in batch
        else (~valid[:, 0]).sum().detach(),
        "invalid_object": ((~valid[:, 1]) & phase_presence[:, 1]).sum().detach()
        if "phase_presence" in batch
        else (~valid[:, 1]).sum().detach(),
    }
    del target, noisy, timesteps, pred, pred_float, target_float, safe_pred
    return loss, metrics


@torch.no_grad()
def evaluate(model, loader, scheduler, cfg, accelerator, deform_loss=None):
    model.eval()
    sums = {}
    count = 0
    for batch_index, (batch, _) in enumerate(loader):
        if batch_index >= int(cfg.eval_max_batches):
            break
        _, metrics = compute_loss(model, batch, scheduler, cfg, deform_loss)
        for key, value in metrics.items():
            gathered = accelerator.gather(value.reshape(1).float()).mean()
            sums[key] = sums.get(key, torch.zeros_like(gathered)) + gathered
        count += 1
    model.train()
    return {f"val/{key}": (value / max(count, 1)).item() for key, value in sums.items()}


def save_checkpoint(accelerator, output_dir: Path, step: int, limit: int):
    if accelerator.is_main_process:
        checkpoints = sorted(
            [path for path in output_dir.glob("checkpoint-*") if path.is_dir()],
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
        import shutil

        for old in checkpoints[:max(0, len(checkpoints) - limit + 1)]:
            shutil.rmtree(old)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(output_dir / f"checkpoint-{step}"))


def resolve_resume_checkpoint(output_dir: Path, requested: str | None) -> Path | None:
    """Resolve a checkpoint path without silently resuming an unrelated run."""
    if not requested:
        return None
    if requested.lower() == "latest":
        candidates = [
            path
            for path in output_dir.glob("checkpoint-*")
            if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name)
        ]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-* directory found under {output_dir}")
        return max(candidates, key=lambda path: int(path.name.rsplit("-", 1)[1]))
    path = Path(requested)
    if not path.is_absolute():
        path = output_dir / path
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    if not re.fullmatch(r"checkpoint-\d+", path.name):
        raise ValueError(f"Resume checkpoint must be named checkpoint-<step>: {path}")
    return path


def checkpoint_step(path: Path) -> int:
    return int(path.name.rsplit("-", 1)[1])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol_lock(cfg) -> None:
    lock_value = cfg.get("protocol_lock_path")
    manifest_keys = cfg.get("protocol_manifest_keys")
    if not lock_value and not manifest_keys:
        return
    if not lock_value or not manifest_keys:
        raise ValueError(
            "protocol_lock_path and protocol_manifest_keys must be provided together"
        )

    lock_path = Path(str(lock_value))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not bool(lock.get("immutable", False)):
        raise ValueError(f"Dataset protocol is not marked immutable: {lock_path}")
    locked_files = lock.get("files", {})
    for config_field in ("train_list", "val_list", "test_list"):
        relative_path = str(manifest_keys.get(config_field, ""))
        if not relative_path:
            raise ValueError(f"Missing protocol key for {config_field}")
        expected = locked_files.get(relative_path)
        if expected is None:
            raise ValueError(f"Protocol lock does not contain {relative_path}")
        manifest_path = Path(str(cfg.get(config_field)))
        actual_hash = file_sha256(manifest_path)
        if actual_hash != expected["sha256"]:
            raise ValueError(
                f"Dataset manifest changed for {config_field}: "
                f"{actual_hash} != {expected['sha256']} ({manifest_path})"
            )
        expected_count = expected.get("count")
        if expected_count is not None:
            values = json.loads(manifest_path.read_text(encoding="utf-8"))
            if len(values) != int(expected_count):
                raise ValueError(
                    f"Dataset manifest count changed for {config_field}: "
                    f"{len(values)} != {expected_count} ({manifest_path})"
                )


def validate_experiment_config(cfg) -> None:
    initialization = cfg.get("initialization", {})
    initialization_mode = str(initialization.get("mode", "pretrained"))
    backbone_mode = str(cfg.model_config.get("backbone_mode", "separate"))
    if initialization_mode not in {"pretrained", "random"}:
        raise ValueError("initialization.mode must be pretrained or random")
    if backbone_mode not in {"separate", "shared"}:
        raise ValueError("model_config.backbone_mode must be separate or shared")
    legacy_transfer = float(cfg.get("lambda_transfer", 0.0))
    if legacy_transfer != 0.0:
        raise ValueError(
            "lambda_transfer is disabled; use the masked interaction velocity loss"
        )
    for name in ("lambda_gas_advection", "lambda_coupling"):
        if float(cfg.get(name, 0.0)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    lambda_coupling = float(cfg.get("lambda_coupling", 0.0))
    if lambda_coupling > 0.0:
        if not bool(cfg.model_config.get("dit_cross_attention", False)) and not bool(
            cfg.get("allow_coupling_supervision_without_cpi", False)
        ):
            raise ValueError(
                "lambda_coupling requires solid-to-gas cross-attention"
            )
        if bool(cfg.model_config.get("coupling_head", False)):
            raise ValueError(
                "trajectory-derived lambda_coupling requires coupling_head=false"
            )
    if cfg.get("exposure_plan", {}).get("mode") == "split_phase":
        if lambda_coupling != 0 or not bool(cfg.get("samplewise_loss_reduction", False)):
            raise ValueError("Independent phase samples require zero coupling and samplewise losses")
        if backbone_mode != "shared" or not cfg.model_config.get("independent_phase_forward"):
            raise ValueError("Split PhysCtrl requires a shared, independently applied predictor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint directory or 'latest'; restores model, optimizer, scheduler, and RNG state.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.run_config_path = str(Path(args.config).resolve())
    validate_experiment_config(cfg)
    validate_protocol_lock(cfg)
    configure_attention_backend(str(cfg.get("sdpa_backend", "default")))
    output_dir = Path(str(cfg.output_dir))
    runtime_dir = Path(str(cfg.get("runtime_dir", output_dir)))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.gradient_accumulation_steps),
        mixed_precision=str(cfg.mixed_precision),
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(log_dir)),
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("physctrl_object_gas_joint")
    set_seed(int(cfg.seed))
    deform_loss = None
    if float(cfg.get("lambda_deform", 0.0)) > 0.0:
        deform_loss = DeformLoss(
            num_points=int(cfg.pc_size),
            frame_dt=float(cfg.get("physics_frame_dt", 0.0417)),
            flat_grid_storage=bool(cfg.get("flat_deform_grid", False)),
            skip_unused_stencil_terms=bool(cfg.get("skip_unused_stencil_terms", False)),
        ).to(accelerator.device)

    mdm_model, dataset_mod, model_mod = load_runtime(cfg)
    train_ds = dataset_mod.ObjectGasDataset("train", cfg)
    val_ds = dataset_mod.ObjectGasDataset("val", cfg)
    test_ds = dataset_mod.ObjectGasDataset("test", cfg)
    exposure_settings = cfg.get("exposure_plan")
    if exposure_settings:
        if file_sha256(Path(exposure_settings.path)) != exposure_settings.sha256:
            raise ValueError("Exposure plan hash mismatch")
        if exposure_settings.get("mode") == "split_phase":
            from split_phase_exposure import SplitPhaseExposureDataset
            train_ds = SplitPhaseExposureDataset(
                train_ds, exposure_settings, output_dir, cfg.get("batch_transition")
            )
        else:
            train_ds = ExposurePlanDataset(train_ds, exposure_settings, output_dir)
        assert int(cfg.gradient_accumulation_steps) == 1
        assert accelerator.num_processes == int(exposure_settings.world_size)
        if cfg.get("batch_transition"):
            assert len(train_ds) == train_ds.sample_cursor(int(cfg.max_train_steps))
        else:
            assert len(train_ds) == int(cfg.max_train_steps) * int(cfg.train_batch_size) * accelerator.num_processes
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train_batch_size),
        shuffle=not bool(exposure_settings),
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        collate_fn=collate_training_batch,
        **loader_performance_kwargs(cfg),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.eval_batch_size),
        shuffle=False,
        num_workers=int(cfg.get("eval_num_workers", cfg.num_workers)),
        pin_memory=True,
        collate_fn=collate_training_batch,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg.eval_batch_size),
        shuffle=False,
        num_workers=int(cfg.get("eval_num_workers", cfg.num_workers)),
        pin_memory=True,
        collate_fn=collate_training_batch,
    )

    model = build_model(cfg, mdm_model, model_mod)
    requested_resume = args.resume_from_checkpoint or cfg.get("resume_from_checkpoint")
    resume_path = resolve_resume_checkpoint(
        output_dir, str(requested_resume) if requested_resume else None
    )
    initialization_report = None
    if resume_path is None:
        initialization = cfg.get("initialization", {})
        initialization_mode = str(initialization.get("mode", "pretrained"))
        backbone_mode = str(cfg.model_config.get("backbone_mode", "separate"))
        if initialization_mode == "pretrained" and backbone_mode == "shared":
            shared_init = initialization.get(
                "shared_checkpoint", cfg.get("shared_init_checkpoint")
            )
            if not shared_init:
                raise ValueError(
                    "Shared pretrained training requires initialization.shared_checkpoint"
                )
            initialization_report = model.initialize_shared_pretrained(str(shared_init))
        elif initialization_mode == "pretrained":
            object_init = initialization.get(
                "solid_checkpoint", cfg.get("object_init_checkpoint")
            )
            gas_init = initialization.get(
                "gas_checkpoint", cfg.get("gas_init_checkpoint")
            )
            if not object_init or not gas_init:
                raise ValueError(
                    "Separate pretrained training requires solid and gas checkpoints"
                )
            initialization_report = model.initialize_pretrained_branches(
                object_checkpoint=str(object_init),
                gas_checkpoint=str(gas_init),
            )
        else:
            initialization_report = {
                "mode": "random",
                "backbone_mode": backbone_mode,
                "trainable_tensors": sum(1 for _ in model.parameters()),
                "trainable_numel": sum(int(p.numel()) for p in model.parameters()),
                "frozen_tensors": 0,
            }
        if accelerator.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "initialization_report.json").write_text(
                json.dumps(initialization_report, indent=2), encoding="utf-8"
            )
        logger.info("INITIALIZATION %s", initialization_report)
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if frozen:
        raise RuntimeError(f"No parameters may be frozen: {frozen[:8]}")
    logger.info(
        "PARAMETERS trainable_tensors=%d trainable_numel=%d frozen_tensors=0",
        sum(1 for parameter in model.parameters() if parameter.requires_grad),
        sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay),
        **({"fused": True} if bool(cfg.get("fused_adamw", False)) else {}),
    )
    max_steps = int(args.max_steps if args.max_steps is not None else cfg.max_train_steps)
    if cfg.get("batch_transition"):
        from batch_transition import transition_scheduler
        scheduler = transition_scheduler(
            optimizer, cfg, accelerator.num_processes, accelerator.split_batches
        )
    else:
        scheduler = build_cosine_scheduler(
            optimizer, cfg.lr_warmup_steps, max_steps,
            accelerator.num_processes, accelerator.split_batches,
        )
    contract = {
        "version": "stage2_complete_phase_objectives_v4_20260922",
        "phase_objective_reduction": "weighted_complete_objectives_active_branches",
        "samplewise_loss_reduction": bool(cfg.get("samplewise_loss_reduction", False)),
        "solid_gravity": "stage1_material_center_full_mask_zero_drag_coefficient",
        "scheduler": "cosine_world_scaled",
        "world_size": accelerator.num_processes,
        "split_batches": accelerator.split_batches,
        "warmup_optimizer_steps": int(cfg.lr_warmup_steps),
        "max_optimizer_steps": max_steps,
        "conditioning": OmegaConf.to_container(cfg.model_config, resolve=True),
        "loss_weights": {
            "lambda_xyz": float(cfg.get("lambda_xyz", 1.0)),
            "lambda_vel": float(cfg.get("lambda_vel", 1.0)),
            "lambda_deform": float(cfg.get("lambda_deform", 0.0)),
            "lambda_floor": float(cfg.get("lambda_floor", 0.0)),
            "lambda_gas_divergence": float(
                cfg.get("lambda_gas_divergence", 0.0)
            ),
            "lambda_gas_advection": float(cfg.get("lambda_gas_advection", 0.0)),
            "lambda_coupling": float(cfg.get("lambda_coupling", 0.0)),
        },
    }
    contract_path = output_dir / "training_contract.json"
    if exposure_settings:
        contract["exposure_plan"] = OmegaConf.to_container(exposure_settings, resolve=True)
        contract["initialization"] = str(cfg.get("initialization", {}).get("mode", "pretrained"))
    if cfg.get("validation_accuracy"):
        contract["validation_accuracy"] = OmegaConf.to_container(cfg.validation_accuracy, resolve=True)
    if cfg.get("batch_transition"):
        contract["batch_transition"] = OmegaConf.to_container(cfg.batch_transition, resolve=True)
    if resume_path is not None:
        previous_contract_path = Path(resume_path).parent / "training_contract.json"
        if not previous_contract_path.is_file():
            raise ValueError("Checkpoint training contract is missing")
        from batch_transition import validate_resume_contract
        validate_resume_contract(
            json.loads(previous_contract_path.read_text(encoding="utf-8")),
            contract, cfg, resume_path,
        )
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        logger.info("TRAINING_CONTRACT %s", contract)
    if bool(cfg.get("gradient_checkpointing", True)) and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    configure_model_performance(model, cfg)
    fusion = getattr(model, "_operator_fusion_report", None)
    if fusion is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"native_operator_rank{accelerator.process_index}.json").write_text(
            json.dumps(fusion, indent=2), encoding="utf-8"
        )

    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )
    global_step = 0
    if resume_path is not None and not exposure_settings and cfg.get("resume_sampler_initial_sha256"):
        generator = train_loader.synchronized_generator
        actual = hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
        if actual != str(cfg.resume_sampler_initial_sha256):
            raise ValueError("Initial shuffled sampler does not match the audited original run")
    if resume_path is not None:
        accelerator.load_state(str(resume_path))
        configure_restored_optimizer(optimizer, cfg)
        global_step = checkpoint_step(resume_path)
        logger.info("RESUMED checkpoint=%s step=%d", resume_path, global_step)
        if cfg.get("batch_transition"):
            logger.info("BATCH_TRANSITION samples=%d optimizer_steps=%d lr=%s",
                        train_ds.sample_cursor(global_step), global_step,
                        scheduler.get_last_lr())
    exposure_counts = [0, 0, 0]
    phase_exposure_counts = [0, 0]
    if exposure_settings and global_step:
        exposure_counts = train_ds.expected_counts(
            global_step, accelerator.num_processes, int(cfg.train_batch_size),
            accelerator.process_index,
        )
        skipped = (train_ds.sample_cursor(global_step) //
                   int(exposure_settings.global_batch)
                   if cfg.get("batch_transition") else global_step)
        train_loader = accelerator.skip_first_batches(train_loader, skipped)
        if hasattr(train_ds, "expected_phase_counts"):
            phase_exposure_counts = train_ds.expected_phase_counts(
                global_step, accelerator.num_processes, int(cfg.train_batch_size),
                accelerator.process_index,
            )
    elif global_step and cfg.get("resume_sampler_initial_sha256"):
        microbatches = global_step * int(cfg.gradient_accumulation_steps)
        if microbatches >= len(train_loader):
            raise ValueError("Audited shuffled replay is restricted to the first epoch")
        train_loader = accelerator.skip_first_batches(train_loader, microbatches)
        logger.info("AUDITED_SAMPLER_RESUME microbatches=%d", microbatches)
    finish_step = int(cfg.get("diagnostic_stop_step", max_steps))
    if not global_step < finish_step <= max_steps:
        raise ValueError("diagnostic_stop_step must be after resume and within the training horizon")
    total_batch = int(cfg.train_batch_size) * accelerator.num_processes * int(cfg.gradient_accumulation_steps)
    logger.info(
        "train=%d val=%d test=%d world=%d effective_batch=%d steps=%d simultaneous_streams=true",
        len(train_ds), len(val_ds), len(test_ds), accelerator.num_processes, total_batch, max_steps,
    )
    logger.info(
        "architecture=phase_unified backbone_mode=%s initialization=%s cross_attention=%s "
        "dit_direction=%s dit_spatial=%s dit_temporal=%s cond_tokens=%s layers=%s "
        "heads=%s all_parameters_trainable=true",
        cfg.model_config.get("backbone_mode", "separate"),
        cfg.get("initialization", {}).get("mode", "pretrained"),
        bool(cfg.model_config.get("dit_cross_attention", False)),
        cfg.model_config.get("dit_cross_attention_direction", "object_to_gas"),
        cfg.model_config.get("dit_cross_attention_spatial", "-"),
        cfg.model_config.get("dit_cross_attention_temporal", "-"),
        (cfg.model_config.get("gas_conditioning", "legacy"),
         cfg.model_config.get("object_conditioning", "legacy")),
        cfg.model_config.get("n_layers", "-"),
        cfg.model_config.get("dit_cross_attention_heads", "-"),
    )
    logger.info(
        "physics_condition=%s physics_tokens=%s physics_num_materials=%s "
        "embedded_object_metadata_required=%s",
        bool(cfg.model_config.get("physics_condition", False)),
        (cfg.model_config.get("gas_conditioning", "legacy"),
         cfg.model_config.get("object_conditioning", "legacy")),
        cfg.model_config.get("physics_num_materials", 5),
        str(cfg.get("object_physics_mode", "off")),
    )
    logger.info(
        "loss_weights xyz=%s velocity=%s deform=%s floor=%s gas_divergence=%s "
        "gas_advection=%s coupling=%s",
        cfg.get("lambda_xyz", 1.0),
        cfg.get("lambda_vel", 1.0),
        cfg.get("lambda_deform", 0.0),
        cfg.get("lambda_floor", 0.0),
        cfg.get("lambda_gas_divergence", 0.0),
        cfg.get("lambda_gas_advection", 0.0),
        cfg.get("lambda_coupling", 0.0),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type="sample", clip_sample=False)
    profile_runtime = bool(cfg.get("profile_runtime", False))
    runtime_start = runtime_previous = time.monotonic()
    runtime_previous_step = global_step
    if profile_runtime:
        torch.cuda.reset_peak_memory_stats(accelerator.device)
    bad_accumulation = False
    while global_step < finish_step:
        for batch, batch_meta in train_loader:
            if exposure_settings:
                cursor = (train_ds.sample_cursor(global_step)
                          if cfg.get("batch_transition") else global_step * total_batch)
                first_position = cursor + accelerator.process_index * int(cfg.train_batch_size)
                expected_positions = list(range(first_position, first_position + int(cfg.train_batch_size)))
                if list(batch_meta["exposure_position"]) != expected_positions:
                    raise ValueError("Frozen exposure cursor differs from the optimizer update")
            with accelerator.accumulate(model):
                metrics = None
                if not bad_accumulation:
                    if exposure_settings and not bool(batch["sample_valid"].all().item()):
                        raise FloatingPointError("Exposure plan delivered an invalid sample")
                    loss, metrics = compute_loss(
                        model, batch, noise_scheduler, cfg, deform_loss
                    )
                    if exposure_settings and (
                        not bool(torch.isfinite(loss).item())
                        or metrics["invalid_gas"].item() or metrics["invalid_object"].item()
                    ):
                        raise FloatingPointError("Non-finite prediction/loss; exposure update not counted")
                    if metrics["invalid_gas"].item() or metrics["invalid_object"].item():
                        logger.warning(
                            "non-finite prediction step=%d rank=%s files=%s invalid_gas=%d invalid_object=%d",
                            global_step,
                            accelerator.process_index,
                            batch_files(batch_meta),
                            metrics["invalid_gas"].item(),
                            metrics["invalid_object"].item(),
                        )
                    if "sample_valid" in batch and not bool(batch["sample_valid"].all().item()):
                        logger.warning(
                            "invalid input sample step=%d rank=%s files=%s",
                            global_step, accelerator.process_index, batch_files(batch_meta),
                        )
                    accelerator.backward(loss)
                    bad_grads = nonfinite_gradients(
                        model, aggregate=cfg.get("aggregate_finite_checks", False)
                    )
                    if any_rank_bad(accelerator, bool(bad_grads), batch["points_src"].device):
                        if exposure_settings:
                            raise FloatingPointError("Non-finite gradients; refusing to skip exposure")
                        bad_accumulation = True
                        if bad_grads:
                            logger.warning(
                                "skip non-finite gradients step=%d rank=%s files=%s params=%s",
                                global_step, accelerator.process_index, batch_files(batch_meta), bad_grads,
                            )
                        optimizer.zero_grad(set_to_none=True)
                    del loss

                if accelerator.sync_gradients:
                    if bad_accumulation:
                        logger.warning(
                            "skipped optimizer update step=%d rank=%s files=%s",
                            global_step, accelerator.process_index, batch_files(batch_meta),
                        )
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                    else:
                        accelerator.clip_grad_norm_(model.parameters(), float(cfg.max_grad_norm))
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        bad_params = nonfinite_parameters(
                            model, aggregate=cfg.get("aggregate_finite_checks", False)
                        )
                        if bad_params:
                            raise FloatingPointError(
                                f"non-finite parameters after step {global_step}: {bad_params}"
                            )
                    bad_accumulation = False

            if accelerator.sync_gradients:
                global_step += 1
                if exposure_settings:
                    for source in batch_meta["exposure_source"]:
                        if int(source) >= 0:
                            exposure_counts[int(source)] += 1
                    if "phase_slot" in batch_meta:
                        for slot, real in zip(batch_meta["phase_slot"], batch_meta["exposure_valid"]):
                            if real:
                                phase_exposure_counts[int(slot)] += 1
                if profile_runtime and (
                    global_step % int(cfg.logging_steps) == 0 or global_step == max_steps
                ):
                    now = time.monotonic()
                    runtime = {
                        "step": global_step, "rank": accelerator.process_index,
                        "world_size": accelerator.num_processes,
                        "batch_per_gpu": int(cfg.train_batch_size),
                        "gradient_accumulation_steps": int(cfg.gradient_accumulation_steps),
                        "global_batch": total_batch,
                        "elapsed_seconds": now - runtime_start,
                        "window_seconds_per_update":
                            (now - runtime_previous) / (global_step - runtime_previous_step),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(accelerator.device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(accelerator.device),
                    }
                    if exposure_settings:
                        runtime["sample_cursor"] = (
                            train_ds.sample_cursor(global_step)
                            if cfg.get("batch_transition") else global_step * total_batch
                        )
                        runtime["learning_rate"] = scheduler.get_last_lr()
                        expected = train_ds.expected_counts(
                            global_step, accelerator.num_processes, int(cfg.train_batch_size),
                            accelerator.process_index,
                        )
                        assert exposure_counts == expected, (exposure_counts, expected)
                        runtime["source_exposure_counts"] = exposure_counts.copy()
                        runtime["exposure_plan_sha256"] = exposure_settings.sha256
                        if hasattr(train_ds, "expected_phase_counts"):
                            assert phase_exposure_counts == train_ds.expected_phase_counts(
                                global_step, accelerator.num_processes, int(cfg.train_batch_size),
                                accelerator.process_index,
                            )
                            runtime["phase_exposure_counts_gas_solid"] = phase_exposure_counts.copy()
                    if metrics is not None:
                        runtime["loss"] = float(metrics["loss"].item())
                        runtime["loss_unweighted"] = float(metrics["loss_unweighted"].item())
                    runtime_path = runtime_dir / f"runtime_rank{accelerator.process_index}.json"
                    temporary = runtime_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
                    temporary.replace(runtime_path)
                    with runtime_path.with_suffix(".jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(runtime) + "\n")
                    runtime_previous, runtime_previous_step = now, global_step
                if metrics is not None and accelerator.is_main_process and global_step % int(cfg.logging_steps) == 0:
                    logger.info(
                        "step=%d loss=%.8g xyz_gas=%.8g xyz_object=%.8g "
                        "loss_vel=%.8g loss_deform=%.8g loss_floor=%.8g "
                        "loss_gas_divergence=%.8g loss_gas_advection=%.8g "
                        "loss_coupling=%.8g",
                        global_step,
                        metrics["loss"].item(),
                        metrics["xyz_gas"].item(),
                        metrics["xyz_object"].item(),
                        metrics["loss_vel"].item(),
                        metrics["loss_deform"].item(),
                        metrics["loss_floor"].item(),
                        metrics["loss_gas_divergence"].item(),
                        metrics["loss_gas_advection"].item(),
                        metrics["loss_coupling"].item(),
                    )
                eval_due = global_step % int(cfg.eval_steps) == 0
                if cfg.get("batch_transition"):
                    eval_due = train_ds.sample_cursor(global_step) % (
                        int(cfg.eval_steps) * int(cfg.batch_transition.global_batch)
                    ) == 0
                if cfg.get("validation_accuracy") and (
                    eval_due
                    or global_step == int(cfg.validation_accuracy.get("first_step", -1))
                    or global_step == finish_step
                ):
                    from validation_accuracy import validate_accuracy
                    summary = validate_accuracy(
                        model, val_ds, cfg, accelerator, global_step, output_dir
                    )
                    if accelerator.is_main_process:
                        logger.info("FULL_VAL_ACCURACY step=%d %s", global_step, summary)
                elif not cfg.get("validation_accuracy") and global_step % int(cfg.eval_steps) == 0:
                    val_logs = evaluate(
                        model, val_loader, noise_scheduler, cfg, accelerator, deform_loss
                    )
                    test_logs = evaluate(
                        model, test_loader, noise_scheduler, cfg, accelerator, deform_loss
                    )
                    if accelerator.is_main_process:
                        logger.info(
                            "Loss eval step %d: %s",
                            global_step,
                            {**val_logs, **{key.replace("val/", "test/"): value for key, value in test_logs.items()}},
                        )
                if global_step % int(cfg.checkpointing_steps) == 0:
                    save_checkpoint(
                        accelerator,
                        output_dir,
                        global_step,
                        int(cfg.checkpoints_total_limit),
                    )
                if global_step >= finish_step:
                    break
        if global_step >= finish_step:
            break

    accelerator.wait_for_everyone()
    if global_step % int(cfg.checkpointing_steps):
        save_checkpoint(accelerator, output_dir, global_step, int(cfg.checkpoints_total_limit))
    accelerator.wait_for_everyone()
    if exposure_settings:
        actual = accelerator.reduce(
            torch.tensor(exposure_counts, device=accelerator.device, dtype=torch.long),
            reduction="sum",
        ).cpu().tolist()
        expected = (list(exposure_settings.expected_counts) if finish_step == max_steps else
                    [sum(train_ds.expected_counts(global_step, accelerator.num_processes,
                                                  int(cfg.train_batch_size), rank)[source]
                         for rank in range(accelerator.num_processes)) for source in range(3)])
        assert actual == expected, actual
        phase_actual = None
        if hasattr(train_ds, "expected_phase_counts"):
            phase_actual = accelerator.reduce(
                torch.tensor(phase_exposure_counts, device=accelerator.device, dtype=torch.long),
                reduction="sum",
            ).cpu().tolist()
            if finish_step == max_steps:
                assert phase_actual == list(exposure_settings.expected_phase_counts), phase_actual
        if accelerator.is_main_process:
            (output_dir / "exposure_final.json").write_text(json.dumps({
                "state": "verified", "global_step": global_step, "actual_counts": actual,
                "expected_counts": expected,
                "phase_counts_gas_solid": phase_actual,
                "scope": "full_training" if finish_step == max_steps else "diagnostic_prefix",
                "plan_sha256": exposure_settings.sha256,
                "definition": "valid samples in successful optimizer updates; rejected reads excluded",
            }, indent=2))
    if accelerator.is_main_process:
        logger.info("%s step=%d",
                    "TRAINING_COMPLETE" if finish_step == max_steps else "DIAGNOSTIC_COMPLETE",
                    global_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
