from __future__ import annotations

import torch
import torch.nn as nn

from .spacetime import PointEmbed, SpaitalTemporalTransformer


def _zero_initialize(module: nn.Module) -> None:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)


class PhaseConditionedMDM_ST(nn.Module):
    """Single-phase PhysCtrl branch with the exact Stage 2 branch interface."""

    def __init__(self, n_points, n_frame, n_feats, model_config):
        super().__init__()
        if not bool(model_config.get("point_embed", True)):
            raise ValueError("PhaseConditionedMDM_ST requires point_embed=true")
        if bool(model_config.get("force_as_token", False)):
            raise ValueError("PhaseConditionedMDM_ST requires force_as_token=false")
        if bool(model_config.get("force_as_latent", False)):
            raise ValueError("PhaseConditionedMDM_ST requires force_as_latent=false")
        if int(model_config.get("max_num_forces", 1)) != 1:
            raise ValueError("PhaseConditionedMDM_ST currently requires max_num_forces=1")
        if bool(model_config.get("class_token", False)):
            raise ValueError("Material conditioning is additive; class_token must be false")

        self.n_points = int(n_points)
        self.n_feats = int(n_feats)
        self.latent_dim = int(model_config.latent_dim)
        self.cond_frame = 1 if bool(model_config.frame_cond) else 0
        self.frame_cond = bool(model_config.frame_cond)
        self.pred_offset = bool(model_config.get("pred_offset", True))
        self.physics_condition_enabled = bool(model_config.get("physics_condition", True))
        self.physics_num_materials = int(model_config.get("physics_num_materials", 5))

        self.input_encoder = PointEmbed(dim=self.latent_dim)
        self.E_cond_encoder = nn.Linear(1, self.latent_dim)
        self.nu_cond_encoder = nn.Linear(1, self.latent_dim)
        self.force_cond_encoder = nn.Linear(3, self.latent_dim)
        self.drag_point_encoder = nn.Linear(3, self.latent_dim)

        self.velocity_adapter = nn.Linear(3, self.latent_dim)
        self.mask_adapter = nn.Linear(1, self.latent_dim)
        self.gravity_adapter = nn.Embedding(2, self.latent_dim)
        self.floor_adapter = nn.Linear(1, self.latent_dim)
        self.coeff_adapter = nn.Linear(1, self.latent_dim)
        self.material_adapter = nn.Embedding(self.physics_num_materials, self.latent_dim)
        self.volume_adapter = nn.Linear(1, self.latent_dim)
        for name in (
            "velocity_adapter",
            "mask_adapter",
            "gravity_adapter",
            "floor_adapter",
            "coeff_adapter",
            "material_adapter",
            "volume_adapter",
        ):
            _zero_initialize(getattr(self, name))

        self.cond_seq_length = 4
        self.dit = SpaitalTemporalTransformer(
            sample_points=self.n_points,
            sample_frames=int(n_frame) + self.cond_frame,
            in_channels=self.n_feats,
            num_layers=int(model_config.n_layers),
            num_attention_heads=self.latent_dim // 64,
            time_embed_dim=self.latent_dim,
            cond_seq_length=self.cond_seq_length,
            cond_seq_length_t=self.cond_frame,
            transformer_block=str(model_config.transformer_block),
            num_classes=0,
        )

    def enable_gradient_checkpointing(self):
        self.dit._set_gradient_checkpointing(self.dit, value=True)

    @staticmethod
    def _first_scalar(value, batch, device, dtype, default):
        if value is None:
            return torch.full((batch, 1), default, device=device, dtype=dtype)
        return value.to(device=device, dtype=dtype).reshape(batch, -1)[:, :1]

    def _condition_tokens(
        self,
        force,
        E,
        nu,
        drag_point,
        floor_height,
        gravity_label,
        coeff,
        material_type,
    ):
        batch = force.shape[0]
        device, dtype = force.device, force.dtype
        force = force.reshape(batch, -1, 3)[:, 0]
        drag_point = drag_point.reshape(batch, -1, drag_point.shape[-1])[:, 0, :3]
        E = self._first_scalar(E, batch, device, dtype, 0.0)
        nu = self._first_scalar(nu, batch, device, dtype, 0.0)
        floor_height = self._first_scalar(floor_height, batch, device, dtype, -2.4)
        coeff = self._first_scalar(coeff, batch, device, dtype, 1.0)
        gravity = self._first_scalar(
            gravity_label, batch, device, torch.float32, 0.0
        ).long().squeeze(-1).clamp(0, 1)
        material = self._first_scalar(
            material_type, batch, device, torch.float32, 0.0
        ).long().squeeze(-1).clamp(0, self.physics_num_materials - 1)

        force_token = self.force_cond_encoder(force)
        E_token = self.E_cond_encoder(E)
        nu_token = self.nu_cond_encoder(nu)
        point_token = self.drag_point_encoder(drag_point)
        if self.physics_condition_enabled:
            force_token = force_token + self.gravity_adapter(gravity)
            E_token = E_token + self.material_adapter(material)
            nu_token = nu_token + self.floor_adapter(floor_height)
            point_token = point_token + self.coeff_adapter(
                torch.log10(coeff.clamp_min(1e-8))
            )
        return torch.stack([force_token, E_token, nu_token, point_token], dim=1)

    def _encode(self, x, init_pc, init_vel, drag_mask, physics_volume):
        if self.frame_cond:
            x = torch.cat([init_pc.unsqueeze(1), x], dim=1)
        batch, frames, points, channels = x.shape
        hidden = self.input_encoder(
            x.reshape(batch * frames, points, channels)
        ).reshape(batch, frames, points, self.latent_dim)
        if self.physics_condition_enabled:
            hidden = hidden + self.velocity_adapter(init_vel)[:, None]
            hidden = hidden + self.mask_adapter(drag_mask)[:, None]
            hidden = hidden + self.volume_adapter(
                torch.log10(physics_volume.clamp_min(1e-8))
            )[:, None]
        return hidden

    def forward(
        self,
        x,
        timesteps,
        init_pc,
        force,
        E,
        nu,
        drag_mask,
        drag_point,
        floor_height,
        gravity_label=None,
        coeff=None,
        y=None,
        null_emb=None,
        init_vel=None,
        physics_volume=None,
    ):
        if x.ndim != 4 or x.shape[2:] != (self.n_points, self.n_feats):
            raise ValueError(
                f"Expected x=[B,T,{self.n_points},{self.n_feats}], got {tuple(x.shape)}"
            )
        batch, frames, points, channels = x.shape
        init_pc = init_pc.reshape(batch, points, channels)
        if init_vel is None:
            init_vel = torch.zeros_like(init_pc)
        else:
            init_vel = init_vel.reshape(batch, points, channels).to(init_pc.dtype)
        if physics_volume is None:
            physics_volume = torch.ones(
                batch, points, 1, device=init_pc.device, dtype=init_pc.dtype
            )
        else:
            physics_volume = physics_volume.reshape(batch, points, 1).to(init_pc.dtype)
        if drag_mask.ndim == 4:
            drag_mask = drag_mask.to(init_pc.dtype).amax(dim=1)
        else:
            drag_mask = drag_mask.reshape(batch, points, 1).to(init_pc.dtype)

        condition = self._condition_tokens(
            force,
            E,
            nu,
            drag_point,
            floor_height,
            gravity_label,
            coeff,
            y,
        )
        if null_emb is not None:
            condition = condition * null_emb
        hidden = self._encode(x, init_pc, init_vel, drag_mask, physics_volume)
        output = self.dit(
            hidden,
            condition,
            timesteps,
            class_labels=None,
        ).reshape(batch, -1, points, 3)[:, self.cond_frame :]
        if output.shape[1] != frames:
            raise RuntimeError(
                f"Unexpected output frames={output.shape[1]} expected={frames}"
            )
        return output + init_pc.unsqueeze(1) if self.pred_offset else output
