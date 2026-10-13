from __future__ import annotations

from typing import Any

import torch
from diffusers import DDIMScheduler


def _repeat_for_cfg(value: torch.Tensor | None) -> torch.Tensor | None:
    return torch.cat([value, value], dim=0) if value is not None else None


class JointTrajPipeline:
    """DDIM inference for the two-stream object/gas PhysCtrl model.

    The model is called once per diffusion timestep with both streams present.
    All stream-specific physical fields are duplicated together during CFG, so
    unconditional and conditional predictions see the same paired sample.
    """

    def __init__(self, model, scheduler: DDIMScheduler | None = None):
        self.model = model
        self.scheduler = scheduler or DDIMScheduler(
            num_train_timesteps=1000,
            prediction_type="sample",
            clip_sample=False,
        )

    @torch.no_grad()
    def __call__(
        self,
        init_pc: torch.Tensor,
        init_vel: torch.Tensor,
        force: torch.Tensor,
        drag_point: torch.Tensor,
        drag_mask: torch.Tensor,
        physics_E: torch.Tensor,
        physics_nu: torch.Tensor,
        physics_mat_type: torch.Tensor,
        physics_floor_height: torch.Tensor,
        physics_gravity: torch.Tensor,
        physics_coeff: torch.Tensor,
        physics_volume: torch.Tensor,
        phase_presence: torch.Tensor | None = None,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
        n_frames: int = 24,
        num_inference_steps: int = 25,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        device = torch.device(device)
        init_pc = init_pc.to(device)
        init_vel = init_vel.to(device)
        force = force.to(device)
        drag_point = drag_point.to(device)
        drag_mask = drag_mask.to(device)
        physics_E = physics_E.to(device)
        physics_nu = physics_nu.to(device)
        physics_mat_type = physics_mat_type.to(device)
        physics_floor_height = physics_floor_height.to(device)
        physics_gravity = physics_gravity.to(device)
        physics_coeff = physics_coeff.to(device)
        physics_volume = physics_volume.to(device)

        if init_pc.ndim != 4 or init_pc.shape[1] != 2 or init_pc.shape[-1] != 3:
            raise ValueError(f"init_pc must be [B,2,N,3], got {tuple(init_pc.shape)}")
        batch_size, _, point_num, _ = init_pc.shape
        if phase_presence is None:
            phase_presence = torch.ones((batch_size, 2), device=device, dtype=torch.bool)
        else:
            phase_presence = phase_presence.to(device=device, dtype=torch.bool)
            if phase_presence.shape != (batch_size, 2):
                raise ValueError(
                    f"phase_presence must be [{batch_size},2], got {tuple(phase_presence.shape)}"
                )
            if not bool(phase_presence.any(dim=1).all()):
                raise ValueError("Each sample must contain at least one active phase")
        dtype = init_pc.dtype
        active_sample = phase_presence[:, None, :, None, None]
        noise_shape = (batch_size, n_frames, 2, point_num, 3)
        if generator is None:
            sample = torch.randn(noise_shape, device=device, dtype=dtype)
        else:
            generator_device = str(getattr(generator, "device", "cpu"))
            if generator_device.startswith(str(device).split(":")[0]):
                sample = torch.randn(noise_shape, generator=generator, device=device, dtype=dtype)
            else:
                sample = torch.randn(noise_shape, generator=generator, device="cpu").to(
                    device=device, dtype=dtype
                )
        sample = torch.where(active_sample, sample, torch.zeros_like(sample))

        self.model.to(device)
        self.model.eval()
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        use_cfg = float(guidance_scale) > 1.0

        if use_cfg:
            init_pc_model = _repeat_for_cfg(init_pc)
            init_vel_model = _repeat_for_cfg(init_vel)
            force_model = _repeat_for_cfg(force)
            drag_point_model = _repeat_for_cfg(drag_point)
            drag_mask_model = _repeat_for_cfg(drag_mask)
            phase_presence_model = _repeat_for_cfg(phase_presence)
            physics_values = [
                _repeat_for_cfg(value)
                for value in (
                    physics_E,
                    physics_nu,
                    physics_mat_type,
                    physics_floor_height,
                    physics_gravity,
                    physics_coeff,
                    physics_volume,
                )
            ]
            null_emb = torch.cat(
                [
                    torch.zeros((batch_size, 1, 1), device=device, dtype=dtype),
                    torch.ones((batch_size, 1, 1), device=device, dtype=dtype),
                ],
                dim=0,
            )
        else:
            init_pc_model = init_pc
            init_vel_model = init_vel
            force_model = force
            drag_point_model = drag_point
            drag_mask_model = drag_mask
            phase_presence_model = phase_presence
            physics_values = [
                physics_E,
                physics_nu,
                physics_mat_type,
                physics_floor_height,
                physics_gravity,
                physics_coeff,
                physics_volume,
            ]
            null_emb = None

        for timestep in self.scheduler.timesteps:
            timestep_value = int(timestep.item())
            model_sample = torch.cat([sample, sample], dim=0) if use_cfg else sample
            model_timesteps = torch.full(
                (model_sample.shape[0],), timestep_value, device=device, dtype=torch.long
            )
            model_output = self.model(
                model_sample,
                model_timesteps,
                init_pc_model,
                init_vel_model,
                force_model,
                drag_point_model,
                drag_mask_model,
                physics_E=physics_values[0],
                physics_nu=physics_values[1],
                physics_mat_type=physics_values[2],
                physics_floor_height=physics_values[3],
                physics_gravity=physics_values[4],
                physics_coeff=physics_values[5],
                physics_volume=physics_values[6],
                null_emb=null_emb,
                phase_presence=phase_presence_model,
            )
            if use_cfg:
                uncond, cond = model_output.chunk(2, dim=0)
                model_output = uncond + float(guidance_scale) * (cond - uncond)
            sample = self.scheduler.step(
                model_output.float(), timestep, sample
            ).prev_sample
            sample = torch.where(active_sample, sample, torch.zeros_like(sample))

        return sample
