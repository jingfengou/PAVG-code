"""Official PhysCtrl solid model exposed to the shared phase-aware trainer."""

import torch
from torch import nn


class SolidStage1(nn.Module):
    def __init__(self, points, frames, config, upstream):
        super().__init__()
        self.solid = upstream.MDM_ST(points, frames, 3, config)

    def enable_gradient_checkpointing(self):
        self.solid.enable_gradient_checkpointing()

    def forward(self, x, timesteps, init_pc, init_vel, force, drag_point, drag_mask,
                physics_E, physics_nu, physics_mat_type, physics_floor_height,
                physics_gravity, physics_coeff, physics_volume,
                phase_presence=None, null_emb=None):
        if phase_presence is not None and (
            phase_presence[:, 0].any() or not phase_presence[:, 1].all()
        ):
            raise ValueError("Solid Stage1 accepts solid-only samples")
        solid = self.solid(
            x[:, :, 1], timesteps, init_pc[:, 1],
            force[:, 1], physics_E[:, 1], physics_nu[:, 1],
            drag_mask[:, 1], drag_point[:, 1], physics_floor_height[:, 1],
            gravity_label=physics_gravity[:, 1, None],
            y=physics_mat_type[:, 1], null_emb=null_emb,
        )
        return torch.stack([torch.zeros_like(solid), solid], dim=2)
