"""Dual PhysCtrl DiT trunks with in-block cross-stream interaction.

Each branch is a complete upstream ``SpaitalTemporalTransformer`` so its
parameters can be initialized losslessly from a single-stream PhysCtrl
checkpoint. The new path is full-resolution cross-attention after the spatial
and temporal self-attention stages in every DiT block. The direction can be
object-to-gas or bidirectional.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _SpatialCrossAttention(nn.Module):
    """Attend from every query point to every context point in one frame."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("Cross-attention dimension must be divisible by heads")
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        enabled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, frames, points, dim = query.shape
        query_flat = query.reshape(batch * frames, points, dim)
        context_flat = context.reshape(batch * frames, points, dim)
        context_norm = self.context_norm(context_flat)
        attended, _ = self.attention(
            self.query_norm(query_flat),
            context_norm,
            context_norm,
            need_weights=False,
        )
        scale = torch.tanh(self.gate).to(dtype=attended.dtype)
        update = scale * attended
        if enabled is not None:
            enabled = enabled.to(device=query.device, dtype=update.dtype)
            if enabled.shape != (batch,):
                raise ValueError(
                    f"Spatial cross-attention mask must have shape [{batch}], "
                    f"got {tuple(enabled.shape)}"
                )
            update = update * enabled[:, None, None].repeat_interleave(frames, dim=0)
        return (query_flat + update).reshape(batch, frames, points, dim)


class _TemporalCrossAttention(nn.Module):
    """Attend across the paired point trajectories of two streams."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("Cross-attention dimension must be divisible by heads")
        self.query_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        enabled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, frames, points, dim = query.shape
        query_time = query.permute(0, 2, 1, 3).reshape(batch * points, frames, dim)
        context_time = context.permute(0, 2, 1, 3).reshape(batch * points, frames, dim)
        context_norm = self.context_norm(context_time)
        attended, _ = self.attention(
            self.query_norm(query_time),
            context_norm,
            context_norm,
            need_weights=False,
        )
        scale = torch.tanh(self.gate).to(dtype=attended.dtype)
        update = scale * attended
        if enabled is not None:
            enabled = enabled.to(device=query.device, dtype=update.dtype)
            if enabled.shape != (batch,):
                raise ValueError(
                    f"Temporal cross-attention mask must have shape [{batch}], "
                    f"got {tuple(enabled.shape)}"
                )
            update = update * enabled[:, None, None].repeat_interleave(points, dim=0)
        query_time = query_time + update
        return query_time.reshape(batch, points, frames, dim).permute(0, 2, 1, 3)


class _CrossStreamLayerInteraction(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        spatial: bool,
        temporal: bool,
        direction: str,
    ):
        super().__init__()
        if direction not in {"object_to_gas", "bidirectional"}:
            raise ValueError(f"Unsupported cross-attention direction: {direction!r}")
        if direction == "object_to_gas":
            # Preserve the original state-dict names so BS changes can resume
            # from checkpoints produced by the one-way experiment.
            self.spatial = _SpatialCrossAttention(dim, heads) if spatial else None
            self.temporal = _TemporalCrossAttention(dim, heads) if temporal else None
        else:
            self.gas_spatial = _SpatialCrossAttention(dim, heads) if spatial else None
            self.object_spatial = _SpatialCrossAttention(dim, heads) if spatial else None
            self.gas_temporal = _TemporalCrossAttention(dim, heads) if temporal else None
            self.object_temporal = _TemporalCrossAttention(dim, heads) if temporal else None


def configure(base_module):
    """Bind the dual-stream implementation to an upstream PhysCtrl module."""

    base_transformer = base_module.SpaitalTemporalTransformer

    class DualStreamSpaitalTemporalTransformer(nn.Module):
        """Two independent PhysCtrl trunks synchronized at every DiT layer."""

        def __init__(
            self,
            *,
            sample_points: int,
            sample_frames: int,
            in_channels: int,
            num_layers: int,
            num_attention_heads: int,
            time_embed_dim: int,
            cond_seq_length: int,
            cond_seq_length_t: int,
            transformer_block: str,
            num_classes: int = 0,
            cross_attention_heads: int = 4,
            cross_attention_spatial: bool = True,
            cross_attention_temporal: bool = True,
            cross_attention_direction: str = "object_to_gas",
            cross_attention_enabled: bool = True,
            backbone_mode: str = "separate",
            stop_gradient_object_context: bool = False,
        ):
            super().__init__()
            if transformer_block != "SpatialTemporalTransformerBlock":
                raise ValueError(
                    "Dual-stream interaction requires "
                    "transformer_block=SpatialTemporalTransformerBlock"
                )
            common = dict(
                sample_points=sample_points,
                sample_frames=sample_frames,
                in_channels=in_channels,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                time_embed_dim=time_embed_dim,
                cond_seq_length=cond_seq_length,
                cond_seq_length_t=cond_seq_length_t,
                transformer_block=transformer_block,
                num_classes=num_classes,
            )
            if backbone_mode not in {"separate", "shared"}:
                raise ValueError(
                    f"backbone_mode must be separate or shared, got {backbone_mode!r}"
                )
            self.backbone_mode = str(backbone_mode)
            self.cross_attention_enabled = bool(cross_attention_enabled)
            # In the one-way simulator, the object drives the gas, but the gas
            # objective must not update the object predictor through context.
            self.stop_gradient_object_context = bool(stop_gradient_object_context)
            self.gas_dit = base_transformer(**common)
            self.object_dit = (
                self.gas_dit
                if self.backbone_mode == "shared"
                else base_transformer(**common)
            )
            dim = int(num_attention_heads) * 64
            if dim != int(time_embed_dim):
                raise ValueError(
                    f"Expected time_embed_dim={dim} for PhysCtrl, got {time_embed_dim}"
                )
            self.interactions = nn.ModuleList(
                [
                    _CrossStreamLayerInteraction(
                        dim,
                        int(cross_attention_heads),
                        self.cross_attention_enabled and bool(cross_attention_spatial),
                        self.cross_attention_enabled and bool(cross_attention_temporal),
                        str(cross_attention_direction),
                    )
                    for _ in range(int(num_layers))
                ]
            )
            self.gradient_checkpointing = False

        def enable_gradient_checkpointing(self) -> None:
            self.gradient_checkpointing = True
            self.gas_dit.gradient_checkpointing = True
            self.object_dit.gradient_checkpointing = True

        def set_cross_attention_enabled(self, enabled: bool) -> None:
            """Toggle cross-stream communication without changing parameters."""
            self.cross_attention_enabled = bool(enabled)

        @staticmethod
        def _prepare_branch(dit, hidden: torch.Tensor, condition: torch.Tensor, timesteps):
            timestep_embedding = dit.time_proj(timesteps).to(dtype=hidden.dtype)
            timestep_embedding = dit.time_embedding(timestep_embedding, None)
            batch, frames, points, channels = hidden.shape
            full_sequence = torch.cat(
                [condition, hidden.reshape(batch, frames * points, channels)], dim=1
            )
            positional = dit.pos_embedding.to(dtype=full_sequence.dtype)
            if positional.shape[1] != full_sequence.shape[1]:
                raise ValueError(
                    "DiT positional sequence mismatch: "
                    f"pos={positional.shape[1]} input={full_sequence.shape[1]}"
                )
            full_sequence = dit.embedding_dropout(full_sequence + positional)
            condition = full_sequence[:, : dit.cond_seq_length]
            hidden = full_sequence[:, dit.cond_seq_length :].reshape(
                batch, frames, points, channels
            )
            return hidden, condition, timestep_embedding

        @staticmethod
        def _spatial_stage(block, hidden, condition, timestep_embedding):
            batch, frames, points, channels = hidden.shape
            hidden_flat = hidden.reshape(batch * frames, points, channels)
            if condition.shape[0] != batch * frames:
                condition = condition.repeat_interleave(frames, dim=0)
            frame_timestep = timestep_embedding.repeat_interleave(frames, dim=0)
            norm_hidden, norm_condition, gate_msa, condition_gate_msa = block.norm1(
                hidden_flat, condition, frame_timestep
            )
            attended_hidden, attended_condition = block.attn1(
                hidden_states=norm_hidden,
                encoder_hidden_states=norm_condition,
                image_rotary_emb=None,
            )
            hidden_flat = hidden_flat + gate_msa * attended_hidden
            condition = condition + condition_gate_msa * attended_condition
            return hidden_flat.reshape(batch, frames, points, channels), condition, frame_timestep

        @staticmethod
        def _feed_forward_stage(block, hidden, condition, frame_timestep):
            batch, frames, points, channels = hidden.shape
            hidden_flat = hidden.reshape(batch * frames, points, channels)
            text_length = condition.size(1)
            norm_hidden, norm_condition, gate_ff, condition_gate_ff = block.norm2(
                hidden_flat, condition, frame_timestep
            )
            ff_input = torch.cat([norm_condition, norm_hidden], dim=1)
            ff_output = block.ff(ff_input)
            hidden_flat = hidden_flat + gate_ff * ff_output[:, text_length:]
            condition = condition + condition_gate_ff * ff_output[:, :text_length]
            return hidden_flat.reshape(batch, frames, points, channels), condition

        @staticmethod
        def _temporal_stage(block, hidden, timestep_embedding):
            batch, frames, points, channels = hidden.shape
            hidden_time = hidden.permute(0, 2, 1, 3).reshape(
                batch * points, frames, channels
            )
            point_timestep = timestep_embedding.repeat_interleave(points, dim=0)
            norm_hidden = block.norm_temp(hidden_time, temb=point_timestep)
            hidden_time = block.attn_temp(norm_hidden) + hidden_time
            return hidden_time.reshape(batch, points, frames, channels).permute(0, 2, 1, 3)

        def _layer_forward(
            self,
            layer_index: int,
            gas_hidden: torch.Tensor,
            object_hidden: torch.Tensor,
            gas_condition: torch.Tensor,
            object_condition: torch.Tensor,
            gas_timestep: torch.Tensor,
            object_timestep: torch.Tensor,
            interaction_enabled: torch.Tensor | None = None,
        ):
            gas_block = self.gas_dit.transformer_blocks[layer_index]
            object_block = self.object_dit.transformer_blocks[layer_index]
            interaction = self.interactions[layer_index]

            gas_hidden, gas_condition, gas_frame_timestep = self._spatial_stage(
                gas_block, gas_hidden, gas_condition, gas_timestep
            )
            object_hidden, object_condition, object_frame_timestep = self._spatial_stage(
                object_block, object_hidden, object_condition, object_timestep
            )
            spatial_gas = gas_hidden
            spatial_object = object_hidden
            gas_spatial = getattr(interaction, "gas_spatial", None)
            if gas_spatial is None:
                gas_spatial = getattr(interaction, "spatial", None)
            object_spatial = getattr(interaction, "object_spatial", None)
            if self.cross_attention_enabled and gas_spatial is not None:
                spatial_gas = gas_spatial(
                    gas_hidden,
                    object_hidden.detach()
                    if self.stop_gradient_object_context
                    else object_hidden,
                    enabled=interaction_enabled,
                )
            if self.cross_attention_enabled and object_spatial is not None:
                spatial_object = object_spatial(
                    object_hidden, gas_hidden, enabled=interaction_enabled
                )
            gas_hidden, object_hidden = spatial_gas, spatial_object

            gas_hidden, gas_condition = self._feed_forward_stage(
                gas_block, gas_hidden, gas_condition, gas_frame_timestep
            )
            object_hidden, object_condition = self._feed_forward_stage(
                object_block, object_hidden, object_condition, object_frame_timestep
            )

            gas_hidden = self._temporal_stage(gas_block, gas_hidden, gas_timestep)
            object_hidden = self._temporal_stage(
                object_block, object_hidden, object_timestep
            )
            temporal_gas = gas_hidden
            temporal_object = object_hidden
            gas_temporal = getattr(interaction, "gas_temporal", None)
            if gas_temporal is None:
                gas_temporal = getattr(interaction, "temporal", None)
            object_temporal = getattr(interaction, "object_temporal", None)
            if self.cross_attention_enabled and gas_temporal is not None:
                temporal_gas = gas_temporal(
                    gas_hidden,
                    object_hidden.detach()
                    if self.stop_gradient_object_context
                    else object_hidden,
                    enabled=interaction_enabled,
                )
            if self.cross_attention_enabled and object_temporal is not None:
                temporal_object = object_temporal(
                    object_hidden, gas_hidden, enabled=interaction_enabled
                )
            gas_hidden, object_hidden = temporal_gas, temporal_object
            return gas_hidden, object_hidden, gas_condition, object_condition

        @staticmethod
        def _finish_branch(dit, hidden, timestep_embedding, return_features=False):
            batch, frames, points, channels = hidden.shape
            hidden = hidden.reshape(batch, frames * points, channels)
            hidden = dit.norm_final(hidden)
            hidden = dit.norm_out(hidden, temb=timestep_embedding)
            output = dit.proj_out(hidden)
            if return_features:
                return output, hidden
            return output

        def forward(
            self,
            gas_hidden: torch.Tensor,
            object_hidden: torch.Tensor,
            gas_condition: torch.Tensor,
            object_condition: torch.Tensor,
            timesteps: torch.Tensor,
            return_gas_features: bool = False,
            phase_presence: torch.Tensor | None = None,
        ):
            if phase_presence is not None:
                if phase_presence.shape != (gas_hidden.shape[0], 2):
                    raise ValueError(
                        "phase_presence must have shape [B,2], got "
                        f"{tuple(phase_presence.shape)}"
                    )
                phase_presence = phase_presence.to(
                    device=gas_hidden.device, dtype=torch.bool
                )
                interaction_enabled = phase_presence[:, 0] & phase_presence[:, 1]
            else:
                interaction_enabled = None

            gas_hidden, gas_condition, gas_timestep = self._prepare_branch(
                self.gas_dit, gas_hidden, gas_condition, timesteps
            )
            object_hidden, object_condition, object_timestep = self._prepare_branch(
                self.object_dit, object_hidden, object_condition, timesteps
            )

            for layer_index in range(len(self.interactions)):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    def custom_forward(
                        gas_h, object_h, gas_c, object_c, current_layer=layer_index
                    ):
                        return self._layer_forward(
                            current_layer,
                            gas_h,
                            object_h,
                            gas_c,
                            object_c,
                            gas_timestep,
                            object_timestep,
                            interaction_enabled,
                        )

                    gas_hidden, object_hidden, gas_condition, object_condition = (
                        torch.utils.checkpoint.checkpoint(
                            custom_forward,
                            gas_hidden,
                            object_hidden,
                            gas_condition,
                            object_condition,
                            use_reentrant=False,
                        )
                    )
                else:
                    gas_hidden, object_hidden, gas_condition, object_condition = (
                        self._layer_forward(
                            layer_index,
                            gas_hidden,
                            object_hidden,
                            gas_condition,
                            object_condition,
                            gas_timestep,
                            object_timestep,
                            interaction_enabled,
                        )
                    )

            gas_result = self._finish_branch(
                self.gas_dit,
                gas_hidden,
                gas_timestep,
                return_features=return_gas_features,
            )
            if return_gas_features:
                gas_output, gas_features = gas_result
            else:
                gas_output = gas_result
            object_output = self._finish_branch(
                self.object_dit, object_hidden, object_timestep
            )
            if return_gas_features:
                return gas_output, object_output, gas_features
            return gas_output, object_output

    globals()["PointEmbed"] = base_module.PointEmbed
    globals()["SpaitalTemporalTransformer"] = base_transformer
    globals()["DualStreamSpaitalTemporalTransformer"] = DualStreamSpaitalTemporalTransformer
    return globals()
