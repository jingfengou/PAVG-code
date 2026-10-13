from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _zero_initialize(module: nn.Module) -> None:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)


class JointMDM_ST(nn.Module):
    """Two pretrained PhysCtrl branches with object-to-gas DiT interaction.

    The gas and object branches retain the exact parameterized components of a
    single-stream PhysCtrl ``MDM_ST``: point encoder, four condition encoders,
    and the complete DiT. New physical-field adapters are zero initialized, and
    every in-block cross-attention path starts behind a zero residual gate.
    """

    _SOURCE_ENCODER_MAP = {
        "input_encoder.": "{branch}_input_encoder.",
        "E_cond_encoder.": "{branch}_E_cond_encoder.",
        "nu_cond_encoder.": "{branch}_nu_cond_encoder.",
        "force_cond_encoder.": "{branch}_force_cond_encoder.",
        "drag_point_encoder.": "{branch}_drag_point_encoder.",
        "velocity_adapter.": "{branch}_velocity_adapter.",
        "mask_adapter.": "{branch}_mask_adapter.",
        "gravity_adapter.": "{branch}_gravity_adapter.",
        "floor_adapter.": "{branch}_floor_adapter.",
        "coeff_adapter.": "{branch}_coeff_adapter.",
        "material_adapter.": "{branch}_material_adapter.",
        "volume_adapter.": "{branch}_volume_adapter.",
        "dit.": "dual_dit.{branch}_dit.",
    }

    def __init__(self, point_num: int, frame_num: int, model_config, mdm_model_module):
        super().__init__()
        PointEmbed = mdm_model_module.PointEmbed
        DualTransformer = mdm_model_module.DualStreamSpaitalTemporalTransformer

        self.point_num = int(point_num)
        self.frame_num = int(frame_num)
        self.latent_dim = int(model_config.latent_dim)
        self.cond_frame = 1 if bool(model_config.frame_cond) else 0
        self.pred_offset = bool(model_config.get("pred_offset", True))
        self.physics_condition_enabled = bool(model_config.get("physics_condition", True))
        self.physics_num_materials = int(model_config.get("physics_num_materials", 5))
        self.backbone_mode = str(model_config.get("backbone_mode", "separate"))
        self.cross_attention_enabled = bool(
            model_config.get("dit_cross_attention", True)
        )
        self.stop_gradient_object_context = bool(
            model_config.get("stop_gradient_object_context", False)
        )
        if self.backbone_mode not in {"separate", "shared"}:
            raise ValueError(
                f"backbone_mode must be separate or shared, got {self.backbone_mode!r}"
            )
        if self.physics_num_materials < 1:
            raise ValueError("physics_num_materials must be positive")
        direction = str(
            model_config.get("dit_cross_attention_direction", "object_to_gas")
        )
        if direction not in {"object_to_gas", "bidirectional"}:
            raise ValueError(
                "This experiment requires object_to_gas or bidirectional "
                "cross-attention, "
                f"got {direction!r}"
            )

        # These names deliberately mirror the source MDM_ST state dict after a
        # gas_/object_ prefix is added. This makes checkpoint mapping auditable.
        self.gas_input_encoder = PointEmbed(dim=self.latent_dim)
        self.object_input_encoder = PointEmbed(dim=self.latent_dim)
        self.gas_E_cond_encoder = nn.Linear(1, self.latent_dim)
        self.object_E_cond_encoder = nn.Linear(1, self.latent_dim)
        self.gas_nu_cond_encoder = nn.Linear(1, self.latent_dim)
        self.object_nu_cond_encoder = nn.Linear(1, self.latent_dim)
        self.gas_force_cond_encoder = nn.Linear(3, self.latent_dim)
        self.object_force_cond_encoder = nn.Linear(3, self.latent_dim)
        self.gas_drag_point_encoder = nn.Linear(3, self.latent_dim)
        self.object_drag_point_encoder = nn.Linear(3, self.latent_dim)

        # The source checkpoints do not contain these fields. Keep them as
        # trainable zero-init adapters so step zero reproduces the two source
        # branches while training can learn to use the richer composite data.
        for branch in ("gas", "object"):
            setattr(self, f"{branch}_velocity_adapter", nn.Linear(3, self.latent_dim))
            setattr(self, f"{branch}_mask_adapter", nn.Linear(1, self.latent_dim))
            setattr(self, f"{branch}_gravity_adapter", nn.Embedding(2, self.latent_dim))
            setattr(self, f"{branch}_floor_adapter", nn.Linear(1, self.latent_dim))
            setattr(self, f"{branch}_coeff_adapter", nn.Linear(1, self.latent_dim))
            setattr(
                self,
                f"{branch}_material_adapter",
                nn.Embedding(self.physics_num_materials, self.latent_dim),
            )
            setattr(self, f"{branch}_volume_adapter", nn.Linear(1, self.latent_dim))
            for suffix in (
                "velocity_adapter",
                "mask_adapter",
                "gravity_adapter",
                "floor_adapter",
                "coeff_adapter",
                "material_adapter",
                "volume_adapter",
            ):
                _zero_initialize(getattr(self, f"{branch}_{suffix}"))

        # Both pretrained models use force_as_token=false, yielding the exact
        # token order [force, E, nu, drag_point]. Extra scalar conditions are
        # zero-init residuals on these four tokens, not extra sequence entries.
        self.cond_seq_length = 4
        self.dual_dit = DualTransformer(
            sample_points=self.point_num,
            sample_frames=self.frame_num + self.cond_frame,
            in_channels=3,
            num_layers=int(model_config.n_layers),
            num_attention_heads=self.latent_dim // 64,
            time_embed_dim=self.latent_dim,
            cond_seq_length=self.cond_seq_length,
            cond_seq_length_t=self.cond_frame,
            transformer_block=str(model_config.transformer_block),
            num_classes=0,
            cross_attention_heads=int(
                model_config.get("dit_cross_attention_heads", self.latent_dim // 64)
            ),
            cross_attention_spatial=bool(
                model_config.get("dit_cross_attention_spatial", True)
            ),
            cross_attention_temporal=bool(
                model_config.get("dit_cross_attention_temporal", True)
            ),
            cross_attention_direction=direction,
            cross_attention_enabled=self.cross_attention_enabled,
            backbone_mode=self.backbone_mode,
            stop_gradient_object_context=self.stop_gradient_object_context,
        )

        if self.backbone_mode == "shared":
            if self.dual_dit.gas_dit is not self.dual_dit.object_dit:
                raise RuntimeError("Shared backbone mode did not alias the two DiT trunks")

    def enable_gradient_checkpointing(self) -> None:
        self.dual_dit.enable_gradient_checkpointing()

    def set_cross_attention_enabled(self, enabled: bool) -> None:
        """Toggle directed solid-to-gas communication for inference ablations."""
        self.cross_attention_enabled = bool(enabled)
        self.dual_dit.set_cross_attention_enabled(enabled)

    @classmethod
    def _target_key(cls, source_key: str, branch: str) -> str | None:
        source_key = source_key.removeprefix("module.")
        for source_prefix, target_pattern in cls._SOURCE_ENCODER_MAP.items():
            if source_key.startswith(source_prefix):
                return target_pattern.format(branch=branch) + source_key[len(source_prefix) :]
        return None

    def initialize_pretrained_branch(self, branch: str, checkpoint: str | Path) -> dict[str, Any]:
        """Load every tensor from one single-stream PhysCtrl checkpoint."""

        if branch not in {"gas", "object"}:
            raise ValueError(f"branch must be gas or object, got {branch!r}")
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing {branch} initialization checkpoint: {checkpoint}")
        if checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            source_state = load_file(str(checkpoint), device="cpu")
        else:
            source_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if isinstance(source_state, dict) and "state_dict" in source_state:
                source_state = source_state["state_dict"]
        if not isinstance(source_state, dict):
            raise TypeError(f"Unsupported checkpoint object in {checkpoint}")

        model_state = self.state_dict()
        mapped = {}
        unmapped = []
        mismatched = []
        for raw_key, value in source_state.items():
            source_key = raw_key.removeprefix("module.")
            target_key = self._target_key(source_key, branch)
            if target_key is None or target_key not in model_state:
                unmapped.append(source_key)
                continue
            if tuple(model_state[target_key].shape) != tuple(value.shape):
                mismatched.append(
                    (source_key, target_key, tuple(value.shape), tuple(model_state[target_key].shape))
                )
                continue
            mapped[target_key] = value

        if unmapped or mismatched or len(mapped) != len(source_state):
            raise RuntimeError(
                f"Incomplete {branch} initialization from {checkpoint}: "
                f"source={len(source_state)} mapped={len(mapped)} "
                f"unmapped={unmapped[:8]} mismatched={mismatched[:8]}"
            )
        self.load_state_dict(mapped, strict=False)
        loaded_numel = sum(int(value.numel()) for value in mapped.values())
        return {
            "branch": branch,
            "checkpoint": str(checkpoint),
            "source_tensors": len(source_state),
            "loaded_tensors": len(mapped),
            "loaded_numel": loaded_numel,
            "coverage": 1.0,
        }

    def initialize_pretrained_branches(
        self,
        *,
        object_checkpoint: str | Path,
        gas_checkpoint: str | Path,
    ) -> dict[str, Any]:
        report = {
            "object": self.initialize_pretrained_branch("object", object_checkpoint),
            "gas": self.initialize_pretrained_branch("gas", gas_checkpoint),
        }
        frozen = [name for name, parameter in self.named_parameters() if not parameter.requires_grad]
        if frozen:
            raise RuntimeError(f"All parameters must be trainable; frozen parameters: {frozen[:8]}")
        report["trainable_tensors"] = sum(1 for _ in self.parameters())
        report["trainable_numel"] = sum(
            int(parameter.numel()) for parameter in self.parameters() if parameter.requires_grad
        )
        report["frozen_tensors"] = 0
        return report

    def initialize_shared_pretrained(self, checkpoint: str | Path) -> dict[str, Any]:
        """Initialize both phase adapters and their shared DiT from one checkpoint."""
        if self.backbone_mode != "shared":
            raise RuntimeError("Shared checkpoint initialization requires backbone_mode=shared")
        report = {
            "gas": self.initialize_pretrained_branch("gas", checkpoint),
            "object": self.initialize_pretrained_branch("object", checkpoint),
            "shared_backbone": True,
        }
        report["trainable_tensors"] = sum(1 for _ in self.parameters())
        report["trainable_numel"] = sum(
            int(parameter.numel()) for parameter in self.parameters() if parameter.requires_grad
        )
        report["frozen_tensors"] = 0
        return report

    def _condition_tokens(
        self,
        branch: str,
        force: torch.Tensor,
        drag_point: torch.Tensor,
        physics_E: torch.Tensor,
        physics_nu: torch.Tensor,
        physics_mat_type: torch.Tensor,
        physics_floor_height: torch.Tensor,
        physics_gravity: torch.Tensor,
        physics_coeff: torch.Tensor,
    ) -> torch.Tensor:
        force_token = getattr(self, f"{branch}_force_cond_encoder")(force)
        E_token = getattr(self, f"{branch}_E_cond_encoder")(physics_E)
        nu_token = getattr(self, f"{branch}_nu_cond_encoder")(physics_nu)
        point_token = getattr(self, f"{branch}_drag_point_encoder")(drag_point)
        if self.physics_condition_enabled:
            gravity = physics_gravity.long().clamp(0, 1)
            material = physics_mat_type.long().clamp(0, self.physics_num_materials - 1)
            force_token = force_token + getattr(self, f"{branch}_gravity_adapter")(gravity)
            E_token = E_token + getattr(self, f"{branch}_material_adapter")(material)
            nu_token = nu_token + getattr(self, f"{branch}_floor_adapter")(
                physics_floor_height
            )
            point_token = point_token + getattr(self, f"{branch}_coeff_adapter")(
                torch.log10(physics_coeff.clamp_min(1e-8))
            )
        return torch.stack([force_token, E_token, nu_token, point_token], dim=1)

    def _encode_branch(
        self,
        branch: str,
        noisy: torch.Tensor,
        init_pc: torch.Tensor,
        init_vel: torch.Tensor,
        drag_mask: torch.Tensor,
        physics_volume: torch.Tensor,
    ) -> torch.Tensor:
        if self.cond_frame:
            noisy = torch.cat([init_pc.unsqueeze(1), noisy], dim=1)
        batch, frames, points, channels = noisy.shape
        hidden = getattr(self, f"{branch}_input_encoder")(
            noisy.reshape(batch * frames, points, channels)
        ).reshape(batch, frames, points, self.latent_dim)
        if self.physics_condition_enabled:
            hidden = hidden + getattr(self, f"{branch}_velocity_adapter")(init_vel)[:, None]
            hidden = hidden + getattr(self, f"{branch}_mask_adapter")(drag_mask)[:, None]
            hidden = hidden + getattr(self, f"{branch}_volume_adapter")(
                torch.log10(physics_volume.clamp_min(1e-8))
            )[:, None]
        return hidden

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        init_pc: torch.Tensor,
        init_vel: torch.Tensor,
        force: torch.Tensor,
        drag_point: torch.Tensor,
        drag_mask: torch.Tensor,
        physics_E: torch.Tensor | None = None,
        physics_nu: torch.Tensor | None = None,
        physics_mat_type: torch.Tensor | None = None,
        physics_floor_height: torch.Tensor | None = None,
        physics_gravity: torch.Tensor | None = None,
        physics_coeff: torch.Tensor | None = None,
        physics_volume: torch.Tensor | None = None,
        null_emb: torch.Tensor | None = None,
        phase_presence: torch.Tensor | None = None,
    ):
        if x.ndim != 5 or x.shape[2:] != (2, self.point_num, 3):
            raise ValueError(
                f"Expected x=[B,T,2,{self.point_num},3], got {tuple(x.shape)}"
            )
        batch, frames, _, points, channels = x.shape
        if phase_presence is not None:
            if phase_presence.shape != (batch, 2):
                raise ValueError(
                    f"phase_presence must have shape [{batch},2], "
                    f"got {tuple(phase_presence.shape)}"
                )
            phase_presence = phase_presence.to(device=x.device, dtype=torch.bool)
        if init_pc.shape != (batch, 2, points, channels):
            raise ValueError("init_pc must have shape [B,2,N,3]")
        if init_vel.shape != init_pc.shape:
            raise ValueError("init_vel must have the same shape as init_pc")

        required = {
            "physics_E": physics_E,
            "physics_nu": physics_nu,
            "physics_mat_type": physics_mat_type,
            "physics_floor_height": physics_floor_height,
            "physics_gravity": physics_gravity,
            "physics_coeff": physics_coeff,
            "physics_volume": physics_volume,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError("Missing joint physical fields: " + ", ".join(missing))

        gas_condition = self._condition_tokens(
            "gas",
            force[:, 0],
            drag_point[:, 0],
            physics_E[:, 0],
            physics_nu[:, 0],
            physics_mat_type[:, 0],
            physics_floor_height[:, 0],
            physics_gravity[:, 0],
            physics_coeff[:, 0],
        )
        object_condition = self._condition_tokens(
            "object",
            force[:, 1],
            drag_point[:, 1],
            physics_E[:, 1],
            physics_nu[:, 1],
            physics_mat_type[:, 1],
            physics_floor_height[:, 1],
            physics_gravity[:, 1],
            physics_coeff[:, 1],
        )
        if null_emb is not None:
            if null_emb.shape[0] != batch:
                raise ValueError(
                    f"null_emb batch must be {batch}, got {tuple(null_emb.shape)}"
                )
            gas_condition = gas_condition * null_emb
            object_condition = object_condition * null_emb

        gas_hidden = self._encode_branch(
            "gas",
            x[:, :, 0],
            init_pc[:, 0],
            init_vel[:, 0],
            drag_mask[:, 0],
            physics_volume[:, 0],
        )
        object_hidden = self._encode_branch(
            "object",
            x[:, :, 1],
            init_pc[:, 1],
            init_vel[:, 1],
            drag_mask[:, 1],
            physics_volume[:, 1],
        )
        gas_output, object_output = self.dual_dit(
            gas_hidden,
            object_hidden,
            gas_condition,
            object_condition,
            timesteps,
            phase_presence=phase_presence,
        )
        gas_output = gas_output.reshape(batch, -1, points, 3)[:, self.cond_frame :]
        object_output = object_output.reshape(batch, -1, points, 3)[:, self.cond_frame :]
        if gas_output.shape[1] != frames or object_output.shape[1] != frames:
            raise RuntimeError(
                f"Unexpected output frames gas={gas_output.shape[1]} "
                f"object={object_output.shape[1]} expected={frames}"
            )
        if self.pred_offset:
            gas_output = gas_output + init_pc[:, 0].unsqueeze(1)
            object_output = object_output + init_pc[:, 1].unsqueeze(1)
        trajectory = torch.stack([gas_output, object_output], dim=2)
        if phase_presence is not None:
            # Select inactive streams explicitly so a non-finite raw branch
            # cannot leak through NaN * 0 into the returned trajectory.
            active = phase_presence[:, None, :, None, None]
            trajectory = torch.where(active, trajectory, torch.zeros_like(trajectory))
        return trajectory
