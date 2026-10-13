from __future__ import annotations

import torch
import torch.nn.functional as F


class DeformLoss(torch.nn.Module):
    """PhysCtrl's kinematic MPM deformation constraint for a solid branch."""

    def __init__(
        self,
        num_points: int = 2048,
        grid_size: int = 125,
        grid_lim: float = 10.0,
        density: float = 1000.0,
        frame_dt: float = 0.0417,
    ):
        super().__init__()
        self.num_points = int(num_points)
        self.grid_size = int(grid_size)
        self.grid_lim = float(grid_lim)
        self.dx = self.grid_lim / self.grid_size
        self.inv_dx = 1.0 / self.dx
        self.density = float(density)
        self.frame_dt = float(frame_dt)
        self.register_buffer("identity", torch.eye(3).unsqueeze(0).repeat(self.num_points, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        vol: torch.Tensor,
        deformation: torch.Tensor,
        affine: torch.Tensor,
        frame_interval: int = 2,
        norm_fac: float = 5.0,
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[-1] != 3:
            raise ValueError(f"Expected x [B,T,N,3], got {tuple(x.shape)}")
        if x.shape[2] != self.num_points:
            raise ValueError(
                f"Expected {self.num_points} points for deformation loss, got {x.shape[2]}"
            )
        if deformation.shape[:3] != x.shape[:3] or affine.shape[:3] != x.shape[:3]:
            raise ValueError(
                "Deformation-state shape mismatch: "
                f"x={tuple(x.shape)}, F={tuple(deformation.shape)}, C={tuple(affine.shape)}"
            )

        x = x.float() * 2.0 + float(norm_fac)
        deformation = deformation.float().reshape(*deformation.shape[:3], 3, 3)
        affine = affine.float().reshape(*affine.shape[:3], 3, 3)
        vol = vol.float().reshape(x.shape[0], self.num_points)
        dt = self.frame_dt * int(frame_interval)
        start_t = 1 if int(frame_interval) == 1 else 0
        end_t = x.shape[1] - 2
        if end_t <= start_t:
            return x.sum() * 0.0

        batch_size = x.shape[0]
        time_count = end_t - start_t
        merged = batch_size * time_count
        device = x.device
        identity = self.identity.to(device=device, dtype=x.dtype)

        particle_x = x[:, start_t:end_t].reshape(merged, self.num_points, 3)
        particle_v = (
            x[:, start_t + 2:end_t + 2] - x[:, start_t:end_t]
        ) / (2.0 * dt)
        particle_v = particle_v.reshape(merged, self.num_points, 3)
        particle_f = deformation[:, start_t:end_t].reshape(
            merged, self.num_points, 3, 3
        )
        particle_f_next = deformation[:, start_t + 1:end_t + 1].reshape(
            merged, self.num_points, 3, 3
        )
        particle_c = affine[:, start_t:end_t].reshape(
            merged, self.num_points, 3, 3
        )
        particle_mass = (
            self.density
            * vol.unsqueeze(1).expand(-1, time_count, -1).reshape(merged, self.num_points)
        ).unsqueeze(-1).expand(-1, -1, 3)

        grid_m = torch.zeros(
            merged, self.grid_size, self.grid_size, self.grid_size,
            device=device, dtype=x.dtype,
        )
        grid_v = torch.zeros(
            merged, self.grid_size, self.grid_size, self.grid_size, 3,
            device=device, dtype=x.dtype,
        )

        # Keep the 3x3x3 stencil inside the finite MPM grid. The learned
        # trajectory can briefly leave the normalized simulation domain early
        # in training; clipping only the grid lookup coordinates prevents an
        # invalid scatter/gather index while preserving a differentiable
        # boundary value for the physical constraint.
        grid_coordinate = particle_x.clamp(
            min=0.5 * self.dx,
            max=(self.grid_size - 2.5) * self.dx,
        )
        grid_pos = grid_coordinate * self.inv_dx
        base_pos = (grid_pos - 0.5).int()
        fx = grid_pos - base_pos
        weights = torch.stack(
            [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ],
            dim=3,
        )
        gradients = torch.stack(
            [fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5],
            dim=3,
        )

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    offset = torch.tensor(
                        [i, j, k], device=device, dtype=x.dtype
                    ).reshape(1, 1, 3)
                    dpos = (offset - fx) * self.dx
                    ix = base_pos[:, :, 0] + i
                    iy = base_pos[:, :, 1] + j
                    iz = base_pos[:, :, 2] + k
                    weight = (
                        weights[:, :, 0, i]
                        * weights[:, :, 1, j]
                        * weights[:, :, 2, k]
                    )
                    dweight = torch.stack(
                        [
                            gradients[:, :, 0, i]
                            * weights[:, :, 1, j]
                            * weights[:, :, 2, k],
                            weights[:, :, 0, i]
                            * gradients[:, :, 1, j]
                            * weights[:, :, 2, k],
                            weights[:, :, 0, i]
                            * weights[:, :, 1, j]
                            * gradients[:, :, 2, k],
                        ],
                        dim=2,
                    ) * self.inv_dx
                    incoming = weight.unsqueeze(-1) * particle_mass * (
                        particle_v
                        + (particle_c @ dpos.unsqueeze(-1)).squeeze(-1)
                    )
                    flat = (
                        ix * self.grid_size * self.grid_size
                        + iy * self.grid_size
                        + iz
                    ).long()
                    flat_grid_v = grid_v.reshape(merged, -1, 3)
                    flat_grid_v.scatter_add_(
                        1, flat.unsqueeze(-1).expand(-1, -1, 3), incoming
                    )
                    flat_grid_m = grid_m.reshape(merged, -1)
                    flat_grid_m.scatter_add_(1, flat, weight * particle_mass[:, :, 0])

        grid_m = torch.where(grid_m > 1.0e-15, grid_m, torch.ones_like(grid_m))
        grid_v = grid_v / grid_m.unsqueeze(-1)
        new_f = torch.zeros_like(particle_f)

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    ix = base_pos[:, :, 0] + i
                    iy = base_pos[:, :, 1] + j
                    iz = base_pos[:, :, 2] + k
                    weight = (
                        weights[:, :, 0, i]
                        * weights[:, :, 1, j]
                        * weights[:, :, 2, k]
                    )
                    dweight = torch.stack(
                        [
                            gradients[:, :, 0, i]
                            * weights[:, :, 1, j]
                            * weights[:, :, 2, k],
                            weights[:, :, 0, i]
                            * gradients[:, :, 1, j]
                            * weights[:, :, 2, k],
                            weights[:, :, 0, i]
                            * weights[:, :, 1, j]
                            * gradients[:, :, 2, k],
                        ],
                        dim=2,
                    ) * self.inv_dx
                    flat = (
                        ix * self.grid_size * self.grid_size
                        + iy * self.grid_size
                        + iz
                    ).long()
                    grid_v_local = grid_v.reshape(merged, -1, 3).gather(
                        1, flat.unsqueeze(-1).expand(-1, -1, 3)
                    )
                    new_f = new_f + (
                        grid_v_local.unsqueeze(-1) @ dweight.unsqueeze(2)
                    )

        predicted_f = (identity.unsqueeze(0) + new_f * dt) @ particle_f
        # PhysCtrl's implementation sums the per-interval deformation
        # penalties rather than averaging them over the temporal window.
        return F.l1_loss(predicted_f, particle_f_next) * time_count
