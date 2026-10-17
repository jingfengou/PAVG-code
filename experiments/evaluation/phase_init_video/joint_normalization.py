"""Shared normalization for scene-to-video trajectory inputs.

This matches learned-model input tokens, not a full SI similarity law.
Phase slots are [gas, solid]. Inputs and restored outputs retain their dtype.
"""
from __future__ import annotations

import numpy as np

KEYS = [
    'init_pc', 'init_vel', 'force', 'drag_point', 'drag_mask', 'physics_E',
    'physics_nu', 'physics_mat_type', 'physics_floor_height', 'physics_gravity',
    'physics_coeff', 'physics_volume',
]


def normalize_scene(original: dict[str, np.ndarray]):
    """Return copied model inputs, a float32 common center, and positive scale."""
    values = {k: original[k].copy() for k in [*KEYS, 'phase_presence']}
    active = values['phase_presence'].astype(bool)
    if active.shape != (2,) or not active.any():
        raise ValueError('Expected at least one active phase in [gas, solid]')
    union = values['init_pc'][active].reshape(-1, 3).astype(np.float64)
    lo, hi = union.min(0), union.max(0)
    extent = float(np.max(hi - lo))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError('Active point-cloud extent must be finite and positive')
    center = ((lo + hi) / 2).astype(np.float32)
    scale = float(1. / np.max(hi - lo))
    values['init_pc'][active] = (values['init_pc'][active] - center[None, None, :]) * scale
    values['drag_point'][active] = (values['drag_point'][active] - center[None, :]) * scale
    values['physics_floor_height'][active] = (values['physics_floor_height'][active] - center[1]) * scale
    values['init_vel'][active] *= scale
    values['force'][active] *= scale
    return values, center, scale


def restore_prediction(prediction: np.ndarray, phase_presence, center, scale: float):
    """Restore [F,2,N,3] predictions to the original input scene coordinates."""
    active = np.asarray(phase_presence, dtype=bool)
    restored = prediction.copy()
    restored[:, active] = restored[:, active] / scale + np.asarray(center, np.float32)[None, None, None, :]
    return restored


def make_diagnostic_controls(normalized: dict[str, np.ndarray]):
    """Keep the same shared transform when disabling interaction or the solid."""
    if not np.asarray(normalized['phase_presence'], dtype=bool).all():
        raise ValueError('The diagnostic controls require a two-phase scene')
    off = {k: v.copy() for k, v in normalized.items()}
    gas_only = {k: v.copy() for k, v in normalized.items()}
    gas_only['phase_presence'][1] = False
    for key in KEYS:
        gas_only[key][1] = np.zeros_like(gas_only[key][1])
    return off, gas_only
