from __future__ import annotations

import json
import os
import random
import importlib.util
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


try:
    from physctrl_aoss_h5_cache import AossH5Cache
except ImportError:
    try:
        from .aoss_h5_cache import AossH5Cache
    except ImportError:
        _aoss_path = Path(__file__).with_name("aoss_h5_cache.py")
        _aoss_spec = importlib.util.spec_from_file_location(
            "physctrl_aoss_h5_cache", _aoss_path
        )
        if _aoss_spec is None or _aoss_spec.loader is None:
            raise ImportError(f"Unable to load AOSS cache module: {_aoss_path}")
        _aoss_module = importlib.util.module_from_spec(_aoss_spec)
        _aoss_spec.loader.exec_module(_aoss_module)
        AossH5Cache = _aoss_module.AossH5Cache


def _is_aoss_uri(value: str) -> bool:
    return str(value).lower().startswith(("s3://", "aoss://", "oss://"))


def _load_manifest_entries(path: str | os.PathLike[str]) -> list[Any]:
    """Load JSON, JSONL, or one-path-per-line manifests."""
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        entries = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line.split("\t", 1)[0])
        return entries
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("files", "uris", "entries", "samples"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(
        f"Manifest must be a list, a dict containing files/uris, or line-based text: {path}"
    )


def _array(group: h5py.Group, key: str, default: Any = None):
    if key not in group:
        return default
    return np.asarray(group[key])


def _scalar(group: h5py.Group, key: str, default: float = 0.0) -> float:
    value = _array(group, key, None)
    if value is None:
        return default
    return float(np.asarray(value).reshape(-1)[0])


def _rows(value: Any, width: int) -> np.ndarray:
    if value is None:
        return np.zeros((1, width), dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    if value.size == 0:
        return np.zeros((1, width), dtype=np.float32)
    return value.reshape(-1, width)


def _norm_xyz(value: np.ndarray, norm_fac: float) -> np.ndarray:
    return (value.astype(np.float32) - norm_fac) / 2.0


def _aggregate_frame_deltas(
    values: np.ndarray, future_indices: np.ndarray
) -> np.ndarray:
    """Sum per-simulation-step deltas between consecutive sampled frames."""
    values = np.asarray(values, dtype=np.float32)
    result = []
    previous = 0
    for current in future_indices:
        result.append(values[previous + 1 : current + 1].sum(axis=0))
        previous = int(current)
    return np.stack(result, axis=0).astype(np.float32)


def _aggregate_interval_velocity_deltas(
    velocities: np.ndarray, future_indices: np.ndarray
) -> np.ndarray:
    """Convert raw velocities to changes of sampled-interval mean velocities.

    The trajectory target stores positions only at ``future_indices``. Its
    finite difference therefore represents the mean start-of-step velocity in
    each interval, rather than the endpoint velocity change stored by the
    simulator. This helper puts the target velocity changes in that same
    discrete representation.
    """
    velocities = np.asarray(velocities, dtype=np.float32)
    interval_means = []
    previous = 0
    for current in future_indices:
        current = int(current)
        if current <= previous:
            raise ValueError("future_indices must be strictly increasing")
        interval_means.append(velocities[previous:current].mean(axis=0))
        previous = current
    interval_means = np.stack(interval_means, axis=0).astype(np.float32)
    result = interval_means.copy()
    result[0] -= velocities[0]
    if result.shape[0] > 1:
        result[1:] = interval_means[1:] - interval_means[:-1]
    return result


def _aggregate_interval_transfer_deltas(
    delta_v: np.ndarray, future_indices: np.ndarray
) -> np.ndarray:
    """Convert raw transfer increments to changes of interval-mean transfer.

    ``delta_v[t]`` is the solid-induced increment applied to the raw velocity
    state at step ``t``. We integrate its cumulative contribution and average
    that contribution over each sampled interval, matching the position finite
    differences used by the model loss.
    """
    delta_v = np.asarray(delta_v, dtype=np.float32)
    cumulative = np.zeros_like(delta_v[0], dtype=np.float32)
    cumulative_series = []
    for frame in range(delta_v.shape[0]):
        if frame > 0:
            cumulative = cumulative + delta_v[frame]
        cumulative_series.append(cumulative.copy())
    cumulative_series = np.stack(cumulative_series, axis=0)

    interval_means = []
    previous = 0
    for current in future_indices:
        current = int(current)
        if current <= previous:
            raise ValueError("future_indices must be strictly increasing")
        interval_means.append(cumulative_series[previous:current].mean(axis=0))
        previous = current
    interval_means = np.stack(interval_means, axis=0).astype(np.float32)
    result = interval_means.copy()
    if result.shape[0] > 1:
        result[1:] = interval_means[1:] - interval_means[:-1]
    return result


def _integrate_frame_velocities(
    velocities: np.ndarray, future_indices: np.ndarray, dt: float
) -> np.ndarray:
    """Integrate start-of-step tracer velocities over sampled intervals."""
    velocities = np.asarray(velocities, dtype=np.float32)
    result = []
    previous = 0
    for current in future_indices:
        result.append(
            velocities[previous:current].sum(axis=0) * np.float32(dt)
        )
        previous = int(current)
    return np.stack(result, axis=0).astype(np.float32)


def _interval_alive_mask(
    alive: np.ndarray, future_indices: np.ndarray
) -> np.ndarray:
    """Require a tracer to remain alive over every raw step in an interval."""
    alive = np.asarray(alive, dtype=bool)
    result = []
    previous = 0
    for current in future_indices:
        result.append(alive[previous : current + 1].all(axis=0))
        previous = int(current)
    return np.stack(result, axis=0)


def _finite_or_zero(value: np.ndarray) -> tuple[np.ndarray, bool]:
    value = np.asarray(value, dtype=np.float32)
    finite = bool(np.isfinite(value).all())
    if finite:
        return value, True
    return np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), False


def _positive_scale(cfg, key: str, default: float) -> float:
    value = float(cfg.get(key, default))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be a positive finite number, got {value}")
    return value


def _protocol_value(h5: h5py.File, key: str, default: str) -> str:
    value = h5.attrs.get(key, default)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _attr_value(h5: h5py.File, key: str, default: Any = None):
    value = h5.attrs.get(key, default)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _composite_index(path: str) -> int:
    """Read the stable row id from a composite_000000.h5 filename."""
    stem = PurePosixPath(str(path).split("?", 1)[0]).stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Composite filename must end in an integer index: {path}") from exc


def _sidecar_row(sidecar: h5py.File, index: int) -> dict[str, np.ndarray] | None:
    """Load one row from the compact object-physics sidecar."""
    if "composite_index" not in sidecar or index < 0 or index >= len(sidecar["composite_index"]):
        return None
    if "ok" in sidecar and not bool(np.asarray(sidecar["ok"][index]).reshape(-1)[0]):
        return None
    fields = (
        "E",
        "nu",
        "mat_type",
        "floor_height",
        "gravity",
        "drag_force",
        "drag_point",
        "drag_mask",
        "vol",
    )
    missing = [name for name in fields if name not in sidecar]
    if missing:
        raise RuntimeError(f"Object-physics sidecar is missing fields: {missing}")
    return {name: np.asarray(sidecar[name][index]) for name in fields}


def _first_dataset(container: Any, names: tuple[str, ...]):
    if container is None:
        return None
    for name in names:
        if name in container:
            return np.asarray(container[name])
    return None


def _resolve_object_source(h5: h5py.File, metadata_root: str) -> Path | None:
    """Resolve the original MPM H5 referenced by a composite file."""
    raw = _attr_value(h5, "object_source_path", None)
    if not raw:
        return None
    source = Path(str(raw))
    candidates = [source]
    if source.is_file():
        candidates.append(source)
    if metadata_root:
        root = Path(metadata_root)
        if root.is_file():
            candidates.append(root)
        else:
            if not source.is_absolute():
                candidates.append(root / source)
            candidates.append(root / source.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _scalar_value(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    array = np.asarray(value)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def _object_drag_defaults(
    force: np.ndarray,
    point: np.ndarray,
    mask: np.ndarray,
    volume: np.ndarray | None,
    max_num_forces: int,
    norm_fac: float,
    position_scale: float,
    gravity: int,
    points_src: np.ndarray | None = None,
    is_mpm: bool = True,
) -> dict[str, Any]:
    """Convert solid conditions; points_src uses the same raw units as point.

    Pure gravity follows the Stage 1 body-force token: material center, full
    support and coefficient zero (there is no external drag to normalize).
    """
    force = _rows(force, 3)
    if int(gravity) == 1 and not np.allclose(force, 0.0):
        raise ValueError("Combined solid drag and gravity is not supported by Stage 1")
    if int(gravity) == 1:
        if points_src is None:
            raise ValueError("Gravity requires the initial solid points, not a placeholder")
        points = np.asarray(points_src, dtype=np.float32)
        if points.ndim == 3:
            points = points[0]
        if points.ndim != 2 or points.shape[-1] != 3 or not len(points):
            raise ValueError("Gravity points must have shape [N,3]")
        if not np.isfinite(points).all():
            raise ValueError("Gravity points must be finite")
        points = _norm_xyz(points * float(position_scale), norm_fac)
        if is_mpm:
            weights = np.asarray(volume, dtype=np.float32).reshape(-1)
            if (weights.shape != (len(points),) or not np.isfinite(weights).all()
                    or (weights < 0).any() or weights.sum() <= 0):
                raise ValueError("MPM gravity requires finite nonnegative volumes with positive sum")
            center = (points * (weights / weights.sum())[:, None]).sum(axis=0)
        else:
            center = points.mean(axis=0)
        normalized_force = np.zeros((max_num_forces, 3), dtype=np.float32)
        normalized_force[0] = [0.0, -1.0, 0.0]
        drag_point = np.zeros((max_num_forces, 4), dtype=np.float32)
        drag_point[0, :3] = center
        drag_point[0, 3] = len(points)
        all_mask = np.zeros((max_num_forces, len(points), 1), dtype=np.float32)
        all_mask[0] = 1.0
        return {"force": normalized_force, "drag_point": drag_point,
                "mask": all_mask,
                "base_drag_coeff": np.zeros((max_num_forces, 1), dtype=np.float32)}
    point = _rows(point, 3)
    mask = _rows(mask, int(mask.shape[-1]) if np.asarray(mask).size else 2048)
    mask = mask[:1]
    if mask.shape[-1] == 0:
        mask = np.zeros((1, 2048), dtype=np.float32)
    point = point[:1]
    mask_row = mask[0].astype(np.float32)

    if volume is None or np.asarray(volume).size == 0:
        volume_array = np.ones(mask_row.shape[-1], dtype=np.float32)
    else:
        volume_array = np.asarray(volume, dtype=np.float32).reshape(-1)
        if volume_array.shape[0] != mask_row.shape[0]:
            volume_array = np.ones(mask_row.shape[-1], dtype=np.float32)

    force = force[:1].astype(np.float32)
    point = point[:1].astype(np.float32)

    count = float(mask_row.sum())
    if count <= 0.0:
        mask_row = np.ones_like(mask_row, dtype=np.float32)
        count = float(mask_row.sum())

    masked_volume = float((volume_array * mask_row).sum())
    total_volume = float(volume_array.sum())
    mask_ratio = masked_volume / max(total_volume, 1e-8)
    mean_masked_volume = masked_volume / max(count, 1.0)
    base_drag_coeff = 9.8 * 1000.0 * mean_masked_volume / max(mask_ratio, 1e-8)
    if not np.isfinite(base_drag_coeff) or base_drag_coeff <= 0.0:
        base_drag_coeff = 1.0

    # Official PhysCtrl expects normalized force and stores the physical
    # coefficient separately as base_drag_coeff.
    normalized_force = force / base_drag_coeff
    normalized_force = np.pad(
        normalized_force,
        ((0, max(0, max_num_forces - normalized_force.shape[0])), (0, 0)),
    )[:max_num_forces]
    coeff = np.zeros((max_num_forces, 1), dtype=np.float32)
    coeff[0, 0] = base_drag_coeff

    if point.shape[0] == 0:
        point = np.zeros((1, 3), dtype=np.float32)
    point_norm = (point * float(position_scale) - float(norm_fac)) / 2.0
    drag_point = np.zeros((max_num_forces, 4), dtype=np.float32)
    drag_point[0, :3] = point_norm[0]
    drag_point[0, 3] = count

    all_mask = np.zeros((max_num_forces, mask_row.shape[0], 1), dtype=np.float32)
    all_mask[0, :, 0] = mask_row

    return {
        "force": normalized_force.astype(np.float32),
        "drag_point": drag_point,
        "mask": all_mask,
        "base_drag_coeff": coeff,
    }


def _load_object_physics(
    h5: h5py.File,
    metadata_root: str,
    metadata_sidecar: h5py.File | None,
    sample_index: int | None,
    norm_fac: float,
    position_scale: float,
    max_num_forces: int,
    object_E_is_log10: bool,
    required: bool,
) -> dict[str, Any] | None:
    """Load object conditions from embedded metadata or the source MPM H5.

    Composite files intentionally keep gas fields at the root, so root E/nu or
    mat_type are never interpreted as object fields. Object metadata must come
    from an explicit object_physics group, object_* datasets, or object_source_path.
    """
    physics_group = h5.get("object_physics")
    embedded_field_names = (
        "E",
        "nu",
        "mat_type",
        "floor_height",
        "gravity",
        "drag_force",
        "drag_point",
        "drag_mask",
        "vol",
    )
    embedded_fields = physics_group is not None and all(
        name in physics_group for name in embedded_field_names
    )

    sidecar_fields = None
    if not embedded_fields and metadata_sidecar is not None and sample_index is not None:
        sidecar_fields = _sidecar_row(metadata_sidecar, sample_index)

    # An embedded row is the authoritative, self-contained copy. Do not open
    # the source H5 or sidecar in that case: training can move the composite
    # files without silently switching to another metadata source.
    source_path = (
        None
        if embedded_fields or sidecar_fields is not None
        else _resolve_object_source(h5, metadata_root)
    )
    source_handle = None
    source = None
    if source_path is not None:
        source_handle = h5py.File(source_path, "r")
        source = source_handle

    object_group = h5.get("initial_conditions/object")

    def get_field(name: str):
        if physics_group is not None and name in physics_group:
            return np.asarray(physics_group[name])
        prefixed = f"object_{name}"
        if prefixed in h5:
            return np.asarray(h5[prefixed])
        if sidecar_fields is not None and name in sidecar_fields:
            return np.asarray(sidecar_fields[name])
        if source is not None and name in source:
            return np.asarray(source[name])
        return None

    try:
        required_fields = {
            "E": get_field("E"),
            "nu": get_field("nu"),
            "mat_type": get_field("mat_type"),
            "floor_height": get_field("floor_height"),
            "gravity": get_field("gravity"),
        }
        missing = [key for key, value in required_fields.items() if value is None]
        if missing:
            if required:
                source_hint = _attr_value(h5, "object_source_path", "<missing>")
                raise RuntimeError(
                    f"Object physics fields {missing} are missing; "
                    f"object_source_path={source_hint!r}, metadata_root={metadata_root!r}"
                )
            return None

        raw_E = _scalar_value(required_fields["E"])
        object_E = raw_E if object_E_is_log10 else (float(np.log10(raw_E)) if raw_E > 0.0 else 0.0)
        object_nu = _scalar_value(required_fields["nu"])
        object_mat_type = int(round(_scalar_value(required_fields["mat_type"])))
        raw_floor = _scalar_value(required_fields["floor_height"])
        floor_is_source = source is not None or sidecar_fields is not None or embedded_fields
        object_floor = (raw_floor - norm_fac) / 2.0 if floor_is_source else (
            raw_floor * position_scale - norm_fac
        ) / 2.0
        object_gravity = int(round(_scalar_value(required_fields["gravity"])))

        source_force = get_field("drag_force")
        source_point = get_field("drag_point")
        source_mask = get_field("drag_mask")
        source_volume = get_field("vol")
        if source_force is None and object_group is not None:
            source_force = _array(object_group, "drag_force", np.zeros((1, 3), dtype=np.float32))
        if source_point is None and object_group is not None:
            source_point = _array(object_group, "drag_point", np.zeros((1, 3), dtype=np.float32))
        if source_mask is None and object_group is not None:
            source_mask = _array(object_group, "drag_mask", np.zeros((1, 2048), dtype=np.float32))
        if source_volume is None:
            source_volume = _array(h5, "vol", None)

        source_position_scale = (
            1.0
            if source is not None or sidecar_fields is not None or embedded_fields
            else position_scale
        )
        if source_volume is None or np.asarray(source_volume).size == 0:
            source_volume = np.ones(2048, dtype=np.float32)
        object_points = None
        if "object/x" in h5:
            # Trajectory coordinates can have a different legacy scale from
            # embedded source metadata. Convert once into the point's units.
            object_points = np.asarray(h5["object/x"][0]) * (
                position_scale / source_position_scale
            )
        elif object_group is not None and "x" in object_group:
            object_points = np.asarray(object_group["x"][0])
        elif source is not None and "x" in source:
            object_points = np.asarray(source["x"][0])
        converted = _object_drag_defaults(
            np.zeros((1, 3), dtype=np.float32) if source_force is None else source_force,
            np.zeros((1, 3), dtype=np.float32) if source_point is None else source_point,
            np.zeros((1, 2048), dtype=np.float32) if source_mask is None else source_mask,
            source_volume,
            max_num_forces,
            norm_fac,
            source_position_scale,
            object_gravity,
            is_mpm=object_mat_type != 3,
            points_src=object_points,
        )
        return {
            "E": np.asarray([object_E], dtype=np.float32),
            "nu": np.asarray([object_nu], dtype=np.float32),
            "mat_type": np.asarray(object_mat_type, dtype=np.int64),
            "floor_height": np.asarray([object_floor], dtype=np.float32),
            "gravity": np.asarray(object_gravity, dtype=np.int64),
            "force": converted["force"],
            "drag_point": converted["drag_point"],
            "mask": converted["mask"],
            "base_drag_coeff": converted["base_drag_coeff"],
            "vol": np.asarray(source_volume, dtype=np.float32).reshape(-1),
            "source_path": (
                "sidecar"
                if sidecar_fields is not None
                else str(source_path) if source_path is not None else "embedded"
            ),
        }
    finally:
        if source_handle is not None:
            source_handle.close()


class ObjectGasDataset(Dataset):
    """Joint object/gas trajectory dataset for the composite H5 protocol."""

    def __init__(self, split: str, cfg):
        self.split = split
        self.cfg = cfg
        self.dataset_root = str(cfg.get("dataset_root", ""))
        list_path = str(cfg[f"{split}_list"])
        self.files = _load_manifest_entries(list_path)
        if not self.files:
            raise RuntimeError(f"Empty {split} manifest: {list_path}")
        self.n_training_frames = int(cfg.n_training_frames)
        self.n_frames_interval = int(cfg.n_frames_interval)
        self.pc_size = int(cfg.pc_size)
        self.norm_fac = float(cfg.norm_fac)
        # Canonical files are already in the PhysCtrl scale. Legacy files may
        # still coexist during migration, so keep their restoration factors
        # explicit and independent of the canonical config values.
        self.position_scale = _positive_scale(cfg, "position_scale", 1.0)
        self.velocity_scale = _positive_scale(cfg, "velocity_scale", 1.0)
        self.gas_force_scale = _positive_scale(cfg, "gas_force_scale", 1.0)
        self.legacy_position_scale = _positive_scale(cfg, "legacy_position_scale", 10.0)
        self.legacy_velocity_scale = _positive_scale(cfg, "legacy_velocity_scale", 10.0)
        self.legacy_gas_force_scale = _positive_scale(cfg, "legacy_gas_force_scale", 0.1)
        self.object_force_scale = _positive_scale(cfg, "object_force_scale", 1.0)
        self.gravity_override = int(cfg.get("gravity_override", 0))
        if self.gravity_override not in (0, 1):
            raise ValueError(f"gravity_override must be 0 or 1, got {self.gravity_override}")
        self.composite_protocol = str(cfg.get("composite_protocol", "physctrl_composite_v2"))
        self.seed = int(cfg.seed)
        self.return_physics = bool(cfg.get("return_physics", False))
        self.object_physics_mode = str(cfg.get("object_physics_mode", "off")).lower()
        if self.object_physics_mode not in {"off", "optional", "required"}:
            raise ValueError(
                "object_physics_mode must be one of off, optional, required; "
                f"got {self.object_physics_mode!r}"
            )
        self.object_metadata_root = str(cfg.get("object_metadata_root", ""))
        self.object_metadata_sidecar = str(cfg.get("object_metadata_sidecar", ""))
        self.aoss_metadata_sidecar_uri = str(
            cfg.get("aoss_metadata_sidecar_uri", "")
        )
        self.aoss_enabled = bool(cfg.get("aoss_enabled", False))
        self.aoss_cache_dir = str(
            cfg.get("aoss_cache_dir", "/tmp/physgen_aoss_h5_cache")
        )
        self.aoss_config_path = str(cfg.get("aoss_config_path", "~/aoss.conf"))
        self.aoss_metadata_cache_dir = str(
            cfg.get(
                "aoss_metadata_cache_dir",
                str(Path(self.aoss_cache_dir).parent / "object_metadata"),
            )
        )
        self.aoss_metadata_cache_max_bytes = int(
            cfg.get("aoss_metadata_cache_max_bytes", 2 * 1024**3)
        )
        self._object_metadata_handle = None
        self._aoss_metadata_cache_instance: AossH5Cache | None = None
        self.object_E_is_log10 = bool(cfg.get("object_E_is_log10", False))
        self.aoss_cache_max_bytes = int(
            cfg.get("aoss_cache_max_bytes", 200 * 1024**3)
        )
        configured_max_files = int(cfg.get("aoss_cache_max_files", 0))
        cache_batches = int(cfg.get("aoss_cache_batches", 4))
        cache_batch_size = int(
            cfg.get("aoss_cache_batch_size", cfg.get("train_batch_size", 1))
        )
        workers = max(1, int(cfg.get("num_workers", 0)))
        prefetch_factor = max(1, int(cfg.get("prefetch_factor", 2)))
        if configured_max_files > 0:
            self.aoss_cache_max_files = configured_max_files
        elif self.aoss_enabled and cache_batches > 0:
            # One shared directory is used by all workers.  Include the
            # prefetched worker window so eviction does not thrash the next
            # batch, while remaining bounded to a few batches.
            self.aoss_cache_max_files = max(
                1, cache_batches * max(1, cache_batch_size) * workers * prefetch_factor
            )
        else:
            self.aoss_cache_max_files = None
        self._aoss_cache_by_worker: dict[int, AossH5Cache] = {}
        # The training DataLoader performs its own epoch-wise shuffle. Keep
        # validation and test manifests in lock-file order so a fixed seed
        # maps to the same sample and prediction on every evaluation.
        if split == "train":
            random.Random(self.seed).shuffle(self.files)
        print(f"ObjectGasDataset[{split}]: {len(self.files)} files")

    def __len__(self) -> int:
        return len(self.files)

    def _future_indices(self, n_frames: int) -> np.ndarray:
        expected = self.n_training_frames * self.n_frames_interval + 1
        if n_frames < expected:
            raise ValueError(f"Need {expected} frames, got {n_frames}")
        return np.arange(1, expected, self.n_frames_interval, dtype=np.int64)

    def _get_object_metadata_sidecar(self) -> h5py.File | None:
        sidecar_ref = self.object_metadata_sidecar or self.aoss_metadata_sidecar_uri
        if not sidecar_ref:
            return None
        if self._object_metadata_handle is None:
            if _is_aoss_uri(sidecar_ref):
                if not self.aoss_enabled:
                    raise RuntimeError(
                        "An AOSS object-physics sidecar was configured but "
                        "aoss_enabled=false"
                    )
                if self._aoss_metadata_cache_instance is None:
                    self._aoss_metadata_cache_instance = AossH5Cache(
                        cache_dir=self.aoss_metadata_cache_dir,
                        config_path=self.aoss_config_path,
                        max_cache_bytes=self.aoss_metadata_cache_max_bytes,
                        max_cache_files=1,
                    )
                sidecar_path = self._aoss_metadata_cache_instance.ensure_local(sidecar_ref)
            else:
                sidecar_path = sidecar_ref
                if not os.path.isfile(sidecar_path):
                    raise FileNotFoundError(
                        f"Object-physics sidecar not found: {sidecar_path}"
                    )
            self._object_metadata_handle = h5py.File(sidecar_path, "r")
        return self._object_metadata_handle

    def _aoss_cache(self) -> AossH5Cache:
        if not self.aoss_enabled:
            raise RuntimeError(
                "The manifest contains an AOSS URI but aoss_enabled=false. "
                "Set aoss_enabled=true and provide aoss_config_path/aoss_cache_dir."
            )
        info = get_worker_info()
        worker_id = -1 if info is None else int(info.id)
        if worker_id not in self._aoss_cache_by_worker:
            self._aoss_cache_by_worker[worker_id] = AossH5Cache(
                cache_dir=self.aoss_cache_dir,
                config_path=self.aoss_config_path,
                max_cache_bytes=self.aoss_cache_max_bytes,
                max_cache_files=self.aoss_cache_max_files,
            )
        return self._aoss_cache_by_worker[worker_id]

    def _manifest_entry(self, value: Any) -> str:
        if isinstance(value, dict):
            for key in ("uri", "aoss_uri", "path", "file"):
                if value.get(key):
                    return str(value[key])
        if isinstance(value, str):
            return value
        raise TypeError(f"Unsupported manifest entry at {self.split}: {value!r}")

    def _manifest_phase(self, value: Any, entry: str) -> str:
        """Return the explicit phase label, with conservative path fallback."""
        phase = ""
        if isinstance(value, str):
            phase = ""
        elif isinstance(value, dict):
            phase = str(value.get("phase", value.get("track", ""))).lower()
        phase = phase.replace("_", "-").strip()
        aliases = {
            "single-solid": "solid",
            "solid-only": "solid",
            "single-gas": "gas",
            "gas-only": "gas",
            "solid-gas": "interaction",
            "interaction": "interaction",
            "solid": "solid",
            "gas": "gas",
        }
        if phase in aliases:
            return aliases[phase]

        lowered = str(entry).lower()
        if "composite_" in lowered or "interaction" in lowered:
            return "interaction"
        if "gas_large_force" in lowered or "gas_random_force" in lowered:
            return "gas"
        return "solid"

    def _resolve_sample_path(self, value: Any) -> tuple[str, str | None]:
        entry = self._manifest_entry(value)
        if _is_aoss_uri(entry):
            local_path = self._aoss_cache().ensure_local(entry)
            return str(local_path), entry
        path = entry
        if not os.path.isabs(path):
            path = os.path.join(self.dataset_root, path)
        return path, None

    def _load_single_phase_item(
        self,
        h5: h5py.File,
        source_entry: str,
        path: str,
        aoss_uri: str | None,
        phase: str,
    ):
        """Adapt a standalone PhysCtrl H5 into the dual-stream interface.

        Stage 2 trains on a mixture of standalone solid/gas trajectories and
        composite solid-gas trajectories. Missing streams are zero-filled and
        marked in ``phase_presence`` so the model can disable cross-attention
        and the corresponding trajectory loss for single-phase samples.
        """
        if phase not in {"solid", "gas"}:
            raise ValueError(f"Expected a single phase, got {phase!r}")
        raw_x = np.asarray(h5["x"], dtype=np.float32)
        raw_v = np.asarray(h5.get("v", np.zeros_like(raw_x)), dtype=np.float32)
        if raw_x.ndim != 3 or raw_x.shape[-1] != 3:
            raise ValueError(f"Standalone trajectory x must be [T,N,3]: {path}")
        if raw_v.shape != raw_x.shape:
            raise ValueError(f"Standalone trajectory v/x mismatch: {path}")
        if raw_x.shape[1] != self.pc_size:
            raise ValueError(
                f"Expected {self.pc_size} particles in {path}, got {raw_x.shape[1]}"
            )

        stored_protocol = _protocol_value(h5, "composite_protocol", "standalone_physctrl_v1")
        if stored_protocol == "legacy_unit10":
            position_scale = self.legacy_position_scale
            velocity_scale = self.legacy_velocity_scale
            gas_force_scale = self.legacy_gas_force_scale
        else:
            # Raw PhysCtrl solid and MantaFlow gas H5 files both use the
            # canonical coordinate range; only positions are shifted/scaled
            # by the official (x - norm_fac) / 2 transform.
            position_scale = 1.0
            velocity_scale = 1.0
            gas_force_scale = 1.0

        future_indices = self._future_indices(raw_x.shape[0])
        normalized_velocity_scale = position_scale / 2.0
        x = _norm_xyz(raw_x * position_scale, self.norm_fac)
        v = raw_v * velocity_scale
        target = x[future_indices]
        source = x[0]
        initial_velocity = v[0]
        zeros_x = np.zeros_like(source, dtype=np.float32)
        zeros_target = np.zeros_like(target, dtype=np.float32)
        zeros_velocity = np.zeros_like(initial_velocity, dtype=np.float32)

        gas_x = source if phase == "gas" else zeros_x
        object_x = source if phase == "solid" else zeros_x
        gas_v = initial_velocity if phase == "gas" else zeros_velocity
        object_v = initial_velocity if phase == "solid" else zeros_velocity
        gas_target = target if phase == "gas" else zeros_target
        object_target = target if phase == "solid" else zeros_target
        points_src = np.stack([gas_x, object_x], axis=0)
        init_vel = np.stack([gas_v, object_v], axis=0)
        points_tgt = np.stack([gas_target, object_target], axis=1)
        phase_presence = np.asarray([phase == "gas", phase == "solid"], dtype=bool)
        n_points = self.pc_size
        phase_deformation = np.zeros(
            (2, self.n_training_frames, n_points, 9),
            dtype=np.float32,
        )
        phase_affine = np.zeros_like(phase_deformation)
        if phase == "solid":
            if "F" not in h5 or "C" not in h5:
                raise RuntimeError(
                    f"Solid trajectory is missing MPM deformation state F/C: {path}"
                )
            raw_F = np.asarray(h5["F"], dtype=np.float32)
            raw_C = np.asarray(h5["C"], dtype=np.float32)
            expected_state_shape = (raw_x.shape[0], self.pc_size, 9)
            if raw_F.shape != expected_state_shape or raw_C.shape != expected_state_shape:
                raise ValueError(
                    f"Solid F/C shape mismatch in {path}: "
                    f"F={raw_F.shape}, C={raw_C.shape}, expected={expected_state_shape}"
                )
            phase_deformation[1] = raw_F[future_indices]
            phase_affine[1] = raw_C[future_indices]

        zero_force = np.zeros(3, dtype=np.float32)
        zero_point = np.zeros(3, dtype=np.float32)
        zero_mask = np.zeros(n_points, dtype=np.float32)
        zero_volume = np.ones(n_points, dtype=np.float32)
        zero_physics = {
            "E": 0.0,
            "nu": 0.0,
            "mat_type": 0,
            "floor_height": 0.0,
            "gravity": 0,
            "coeff": 1.0,
        }
        gas_force = zero_force.copy()
        gas_point = zero_point.copy()
        gas_mask = zero_mask.copy()
        gas_volume = zero_volume.copy()
        gas_E = gas_nu = gas_floor = 0.0
        gas_coeff = 1.0
        gas_mat_type = gas_gravity = 0
        object_force = zero_force.copy()
        object_point = zero_point.copy()
        object_mask = zero_mask.copy()
        object_volume = zero_volume.copy()
        object_E = object_nu = object_floor = object_coeff = 0.0
        object_mat_type = object_gravity = 0

        raw_gravity = int(round(_scalar(h5, "gravity", 0.0)))
        raw_floor = _scalar(h5, "floor_height", 0.0)
        volume = np.asarray(_array(h5, "vol", zero_volume), dtype=np.float32).reshape(-1)
        if volume.shape[0] != n_points:
            volume = zero_volume.copy()

        if phase == "gas":
            gas_force = _rows(_array(h5, "drag_force", zero_force), 3)[0]
            gas_force = gas_force * gas_force_scale
            gas_point = _rows(_array(h5, "drag_point", zero_point), 3)[0]
            gas_mask = _rows(_array(h5, "drag_mask", zero_mask), n_points)[0]
            gas_volume = volume
            gas_E_raw = _scalar(h5, "E", 0.0)
            gas_E = float(np.log10(gas_E_raw)) if gas_E_raw > 0.0 else 0.0
            gas_nu = _scalar(h5, "nu", 0.0)
            gas_mat_type = int(round(_scalar(h5, "mat_type", 4.0)))
            gas_floor = (raw_floor * position_scale - self.norm_fac) / 2.0
            gas_gravity = 0 if not np.allclose(gas_force, 0.0) else raw_gravity
        else:
            source_force = _array(h5, "drag_force", zero_force)
            source_point = _array(h5, "drag_point", zero_point)
            source_mask = _array(h5, "drag_mask", zero_mask)
            object_mat_type = int(round(_scalar(h5, "mat_type", 0.0)))
            converted = _object_drag_defaults(
                source_force,
                source_point,
                source_mask,
                volume,
                int(self.cfg.get("max_num_forces", 1)),
                self.norm_fac,
                position_scale,
                raw_gravity,
                points_src=raw_x,
                is_mpm=object_mat_type != 3,
            )
            object_force = converted["force"][0]
            object_point = converted["drag_point"][0, :3]
            object_mask = converted["mask"][0, :, 0]
            object_coeff = float(converted["base_drag_coeff"][0, 0])
            object_volume = volume
            raw_E = _scalar(h5, "E", 0.0)
            object_E = raw_E if self.object_E_is_log10 else (
                float(np.log10(raw_E)) if raw_E > 0.0 else 0.0
            )
            object_nu = _scalar(h5, "nu", 0.0)
            object_floor = (raw_floor * position_scale - self.norm_fac) / 2.0
            object_gravity = raw_gravity

        alive_raw = np.asarray(
            h5.get("alive_mask", np.ones(raw_x.shape[:2], dtype=np.uint8))
        ).astype(bool)
        interaction_dt = float(
            _attr_value(h5, "frame_dt", self.cfg.get("gas_frame_dt", self.cfg.get("frame_dt", 0.05)))
        )
        gas_advection_displacement = np.zeros(
            (self.n_training_frames, n_points, 3), dtype=np.float32
        )
        gas_advection_alive = np.zeros(
            (self.n_training_frames, n_points), dtype=bool
        )
        if phase == "gas":
            gas_advection_displacement = _integrate_frame_velocities(
                raw_v, future_indices, interaction_dt
            ) * normalized_velocity_scale
            gas_advection_alive = _interval_alive_mask(alive_raw, future_indices)

        sample_frame_steps = np.diff(
            np.concatenate([np.asarray([0], dtype=np.int64), future_indices])
        ).astype(np.float32)
        physics_E = np.asarray([gas_E, object_E], dtype=np.float32)[:, None]
        physics_nu = np.asarray([gas_nu, object_nu], dtype=np.float32)[:, None]
        physics_mat_type = np.asarray([gas_mat_type, object_mat_type], dtype=np.int64)
        physics_floor_height = np.asarray([gas_floor, object_floor], dtype=np.float32)[:, None]
        physics_gravity = np.asarray([gas_gravity, object_gravity], dtype=np.int64)
        physics_coeff = np.asarray([gas_coeff if phase == "gas" else 1.0, object_coeff if phase == "solid" else 1.0], dtype=np.float32)[:, None]
        physics_volume = np.stack([gas_volume, object_volume], axis=0).astype(np.float32)[..., None]
        force = np.stack([gas_force, object_force], axis=0).astype(np.float32)
        gas_drag_point = (
            _norm_xyz(gas_point * position_scale, self.norm_fac)
            if phase == "gas"
            else zero_point
        )
        object_drag_point = object_point if phase == "solid" else zero_point
        drag_point = np.stack([gas_drag_point, object_drag_point], axis=0).astype(
            np.float32
        )
        drag_mask = np.stack([gas_mask, object_mask], axis=0).astype(np.float32)[..., None]
        gas_target_velocity = (
            raw_v[future_indices] * normalized_velocity_scale
            if phase == "gas"
            else np.zeros_like(target)
        ).astype(np.float32)
        gas_initial_velocity = (
            raw_v[0] * normalized_velocity_scale
            if phase == "gas"
            else np.zeros_like(source)
        ).astype(np.float32)
        interaction_total_delta_v = np.zeros_like(gas_target_velocity)
        interaction_transfer_delta_v = np.zeros_like(gas_target_velocity)
        interaction_shape = (self.n_training_frames, n_points, 3)
        sample_valid = all(
            bool(np.isfinite(value).all())
            for value in (
                points_src,
                init_vel,
                points_tgt,
                force,
                drag_point,
                drag_mask,
                physics_E,
                physics_nu,
                physics_floor_height,
                physics_coeff,
                physics_volume,
                gas_target_velocity,
                gas_initial_velocity,
                gas_advection_displacement,
                phase_deformation,
                phase_affine,
            )
        )
        data = {
            "points_src": torch.from_numpy(points_src).float(),
            "init_vel": torch.from_numpy(init_vel).float(),
            "points_tgt": torch.from_numpy(points_tgt).float(),
            "force": torch.from_numpy(force).float(),
            "drag_point": torch.from_numpy(drag_point).float(),
            "drag_mask": torch.from_numpy(drag_mask).float(),
            "gravity": torch.from_numpy(physics_gravity).long(),
            "physics_E": torch.from_numpy(physics_E).float(),
            "physics_nu": torch.from_numpy(physics_nu).float(),
            "physics_mat_type": torch.from_numpy(physics_mat_type).long(),
            "physics_floor_height": torch.from_numpy(physics_floor_height).float(),
            "physics_gravity": torch.from_numpy(physics_gravity).long(),
            "physics_coeff": torch.from_numpy(physics_coeff).float(),
            "physics_volume": torch.from_numpy(physics_volume).float(),
            "physics_F": torch.from_numpy(phase_deformation).float(),
            "physics_C": torch.from_numpy(phase_affine).float(),
            "sample_valid": torch.tensor(sample_valid, dtype=torch.bool),
            "phase_presence": torch.from_numpy(phase_presence).bool(),
            "gas_target_velocity": torch.from_numpy(gas_target_velocity).float(),
            "gas_initial_velocity": torch.from_numpy(gas_initial_velocity).float(),
            "gas_velocity_delta": torch.zeros(interaction_shape).float(),
            "interaction_delta_v": torch.zeros(interaction_shape).float(),
            "interaction_total_delta_v": torch.zeros(interaction_shape).float(),
            "interaction_transfer_delta_v": torch.zeros(interaction_shape).float(),
            "interaction_alive": torch.zeros((self.n_training_frames, n_points)).bool(),
            "interaction_valid": torch.zeros(self.n_training_frames).bool(),
            "interaction_dt": torch.tensor(interaction_dt, dtype=torch.float32),
            "sample_frame_steps": torch.from_numpy(sample_frame_steps).float(),
            "gas_advection_displacement": torch.from_numpy(gas_advection_displacement).float(),
            "gas_advection_alive": torch.from_numpy(gas_advection_alive).bool(),
        }
        meta = {
            "file": source_entry,
            "phase": phase,
            "phase_presence": phase_presence.tolist(),
            "protocol": self.composite_protocol,
            "stored_protocol": stored_protocol,
            "position_scale": position_scale,
            "velocity_scale": velocity_scale,
            "gas_force_scale": gas_force_scale,
            "gravity": physics_gravity.tolist(),
            "sample_valid": sample_valid,
            "object_physics_source": "standalone_root" if phase == "solid" else "missing",
        }
        if aoss_uri is not None:
            meta["aoss_uri"] = aoss_uri
            meta["aoss_cache_path"] = path
        return data, meta

    def __getitem__(self, index: int):
        manifest_value = self.files[index]
        source_entry = self._manifest_entry(manifest_value)
        sample_phase = self._manifest_phase(manifest_value, source_entry)
        path, aoss_uri = self._resolve_sample_path(source_entry)

        with h5py.File(path, "r") as h5:
            if sample_phase != "interaction":
                return self._load_single_phase_item(
                    h5, source_entry, path, aoss_uri, sample_phase
                )
            stored_protocol = _protocol_value(h5, "composite_protocol", "legacy_unit10")
            if stored_protocol == "physctrl_composite_v2_canonical":
                position_scale = 1.0
                velocity_scale = 1.0
                # Canonical files already store gas drag_force in the same
                # velocity-condition scale as the trajectory velocity.
                gas_force_scale = 1.0
                gravity_value = 0
            elif stored_protocol == "legacy_unit10":
                position_scale = self.legacy_position_scale
                velocity_scale = self.legacy_velocity_scale
                gas_force_scale = self.legacy_gas_force_scale
                gravity_value = self.gravity_override
            else:
                raise ValueError(f"Unknown composite_protocol={stored_protocol!r} in {path}")
            gas_x_raw = np.asarray(h5["x"], dtype=np.float32)
            gas_v_raw = np.asarray(h5["v"], dtype=np.float32)
            object_x_raw = np.asarray(h5["object/x"], dtype=np.float32)
            object_v_raw = np.asarray(h5["object/v"], dtype=np.float32)
            if gas_x_raw.shape != object_x_raw.shape:
                raise ValueError(f"Object/gas position shape mismatch in {path}: {gas_x_raw.shape} vs {object_x_raw.shape}")
            if gas_v_raw.shape != gas_x_raw.shape or object_v_raw.shape != object_x_raw.shape:
                raise ValueError(
                    f"Initial-state velocity shape mismatch in {path}: "
                    f"gas_v={gas_v_raw.shape}, object_v={object_v_raw.shape}, x={gas_x_raw.shape}"
                )
            if gas_x_raw.shape[1] != self.pc_size:
                raise ValueError(f"Expected {self.pc_size} particles in {path}, got {gas_x_raw.shape[1]}")

            future_indices = self._future_indices(gas_x_raw.shape[0])
            gas_x = _norm_xyz(gas_x_raw * position_scale, self.norm_fac)
            object_x = _norm_xyz(object_x_raw * position_scale, self.norm_fac)
            gas_v = gas_v_raw.astype(np.float32) * velocity_scale
            object_v = object_v_raw.astype(np.float32) * velocity_scale
            object_source_path = _resolve_object_source(h5, self.object_metadata_root)
            if object_source_path is None:
                raise RuntimeError(
                    "Interaction trajectory cannot resolve its object MPM source for "
                    f"deformation state F/C: {path}"
                )
            with h5py.File(object_source_path, "r") as object_source:
                if "F" not in object_source or "C" not in object_source:
                    raise RuntimeError(
                        "Resolved object MPM source is missing F/C: "
                        f"{object_source_path}"
                    )
                raw_object_F = np.asarray(object_source["F"], dtype=np.float32)
                raw_object_C = np.asarray(object_source["C"], dtype=np.float32)
            expected_state_shape = (gas_x_raw.shape[0], self.pc_size, 9)
            if (
                raw_object_F.shape != expected_state_shape
                or raw_object_C.shape != expected_state_shape
            ):
                raise ValueError(
                    f"Interaction F/C shape mismatch in {path}: "
                    f"F={raw_object_F.shape}, C={raw_object_C.shape}, "
                    f"expected={expected_state_shape}"
                )
            phase_deformation = np.zeros(
                (2, self.n_training_frames, self.pc_size, 9), dtype=np.float32
            )
            phase_affine = np.zeros_like(phase_deformation)
            phase_deformation[1] = raw_object_F[future_indices]
            phase_affine[1] = raw_object_C[future_indices]

            interaction_group = h5.get("interaction")
            normalized_velocity_scale = position_scale / 2.0
            gas_target_velocity = (
                gas_v_raw[future_indices] * normalized_velocity_scale
            ).astype(np.float32)
            gas_initial_velocity = (
                gas_v_raw[0] * normalized_velocity_scale
            ).astype(np.float32)
            alive_raw = np.asarray(
                h5.get("alive_mask", np.ones(gas_x_raw.shape[:2], dtype=np.uint8))
            ).astype(bool)
            interaction_dt = float(
                interaction_group.attrs.get(
                    "dt", float(self.cfg.get("gas_frame_dt", 0.0417))
                )
                if interaction_group is not None
                else float(self.cfg.get("gas_frame_dt", 0.0417))
            )
            gas_advection_displacement = _integrate_frame_velocities(
                gas_v_raw, future_indices, interaction_dt
            ) * normalized_velocity_scale
            gas_advection_alive = _interval_alive_mask(alive_raw, future_indices)
            interaction_available = bool(
                interaction_group is not None
                and "gas_velocity_delta" in interaction_group
                and "interaction_delta_v" in interaction_group
            )
            if interaction_available:
                gas_velocity_delta_raw = np.asarray(
                    interaction_group["gas_velocity_delta"], dtype=np.float32
                )
                interaction_delta_v_raw = np.asarray(
                    interaction_group["interaction_delta_v"], dtype=np.float32
                )
                expected_shape = gas_v_raw.shape
                if (
                    gas_velocity_delta_raw.shape != expected_shape
                    or interaction_delta_v_raw.shape != expected_shape
                ):
                    raise ValueError(
                        "Interaction velocity fields must match gas trajectory shape in "
                        f"{path}: gas_delta={gas_velocity_delta_raw.shape}, "
                        f"interaction_delta={interaction_delta_v_raw.shape}, "
                        f"expected={expected_shape}"
                    )
                gas_velocity_delta = _aggregate_frame_deltas(
                    gas_velocity_delta_raw, future_indices
                ) * normalized_velocity_scale
                interaction_delta_v = _aggregate_frame_deltas(
                    interaction_delta_v_raw, future_indices
                ) * normalized_velocity_scale
                # Keep the raw-field aggregates above for provenance, and
                # derive interval-compatible targets for the trajectory loss.
                interaction_total_delta_v = (
                    _aggregate_interval_velocity_deltas(
                        gas_v_raw, future_indices
                    )
                    * normalized_velocity_scale
                )
                interaction_transfer_delta_v = (
                    _aggregate_interval_transfer_deltas(
                        interaction_delta_v_raw, future_indices
                    )
                    * normalized_velocity_scale
                )
                interaction_alive = alive_raw[future_indices]
                interaction_valid = np.ones(
                    (self.n_training_frames,), dtype=bool
                )
            else:
                gas_velocity_delta = np.zeros_like(gas_target_velocity)
                interaction_delta_v = np.zeros_like(gas_target_velocity)
                interaction_total_delta_v = np.zeros_like(gas_target_velocity)
                interaction_transfer_delta_v = np.zeros_like(gas_target_velocity)
                interaction_alive = np.zeros(
                    (self.n_training_frames, self.pc_size), dtype=bool
                )
                interaction_valid = np.zeros(
                    (self.n_training_frames,), dtype=bool
                )
            sample_frame_steps = np.diff(
                np.concatenate([np.asarray([0], dtype=np.int64), future_indices])
            ).astype(np.float32)

            points_src = np.stack([gas_x[0], object_x[0]], axis=0)
            init_vel = np.stack([gas_v[0], object_v[0]], axis=0)
            points_tgt = np.stack([gas_x[future_indices], object_x[future_indices]], axis=1)

            gas_group = h5["initial_conditions/gas"]
            object_group = h5["initial_conditions/object"]
            object_physics = None
            if self.object_physics_mode != "off":
                sample_index = None
                try:
                    sample_index = _composite_index(source_entry)
                except ValueError:
                    # A cache filename is a URI hash, not the original
                    # composite index. Embedded metadata does not need it;
                    # a sidecar requires a manifest with composite_XXXX.h5.
                    sample_index = None
                object_physics = _load_object_physics(
                    h5=h5,
                    metadata_root=self.object_metadata_root,
                    metadata_sidecar=self._get_object_metadata_sidecar(),
                    sample_index=sample_index,
                    norm_fac=self.norm_fac,
                    position_scale=position_scale,
                    max_num_forces=int(self.cfg.get("max_num_forces", 1)),
                    object_E_is_log10=self.object_E_is_log10,
                    required=self.object_physics_mode == "required",
                )
            gas_force = _rows(_array(gas_group, "drag_force", _array(h5, "drag_force")), 3)[0]
            gas_force = gas_force * gas_force_scale
            gas_point = _rows(_array(gas_group, "drag_point", _array(h5, "drag_point")), 3)[0]
            gas_mask = _rows(_array(gas_group, "drag_mask", _array(h5, "drag_mask")), self.pc_size)[0]

            # The object branch must use the same normalized force protocol as
            # the official PhysCtrl TrajDataset. The raw object drag fields in
            # initial_conditions/object are retained for provenance, while the
            # embedded metadata conversion provides force/base_drag_coeff.
            if object_physics is not None:
                object_force = _rows(object_physics["force"], 3)[0]
                object_point = _rows(object_physics["drag_point"], 4)[0, :3]
                object_mask = np.asarray(object_physics["mask"], dtype=np.float32)[0, :, 0]
                object_volume = np.asarray(object_physics["vol"], dtype=np.float32).reshape(-1)
                object_base_drag_coeff = np.asarray(
                    object_physics["base_drag_coeff"], dtype=np.float32
                ).reshape(-1, 1)
                object_E = float(np.asarray(object_physics["E"]).reshape(-1)[0])
                object_nu = float(np.asarray(object_physics["nu"]).reshape(-1)[0])
                object_mat_type = int(np.asarray(object_physics["mat_type"]).reshape(-1)[0])
                object_floor = float(np.asarray(object_physics["floor_height"]).reshape(-1)[0])
                object_gravity = int(np.asarray(object_physics["gravity"]).reshape(-1)[0])
            else:
                object_force = _rows(_array(object_group, "drag_force", None), 3)[0]
                object_point = _rows(_array(object_group, "drag_point", None), 3)[0]
                object_mask = _rows(_array(object_group, "drag_mask", None), self.pc_size)[0]
                object_volume = np.ones(self.pc_size, dtype=np.float32)
                object_base_drag_coeff = np.ones((1, 1), dtype=np.float32)
                object_E = object_nu = object_floor = 0.0
                object_mat_type = object_gravity = 0

            gas_E_raw = _scalar(h5, "E", 0.0)
            gas_E = float(np.log10(gas_E_raw)) if gas_E_raw > 0.0 else 0.0
            gas_nu = _scalar(h5, "nu", 0.0)
            gas_mat_type = int(round(_scalar(h5, "mat_type", 4.0)))
            gas_floor_raw = _scalar(h5, "floor_height", 0.0)
            gas_floor = (gas_floor_raw * position_scale - self.norm_fac) / 2.0
            raw_gas_gravity = int(round(_scalar(h5, "gravity", 0.0)))
            # A composite gas sample can carry a legacy gravity=1 marker while
            # also carrying a non-zero external velocity condition. The
            # official gravity branch asserts that these are mutually exclusive.
            # Preserve the raw value in metadata but expose the corrected label.
            gas_gravity = 0 if not np.allclose(gas_force, 0.0) else raw_gas_gravity
            gas_volume = _array(h5, "vol", np.ones(self.pc_size, dtype=np.float32))
            gas_volume = np.asarray(gas_volume, dtype=np.float32).reshape(-1)
            if gas_volume.shape[0] != self.pc_size:
                gas_volume = np.ones(self.pc_size, dtype=np.float32)
            if object_volume.shape[0] != self.pc_size:
                object_volume = np.ones(self.pc_size, dtype=np.float32)

            force = np.stack([gas_force, object_force], axis=0).astype(np.float32)
            gas_point_norm = _norm_xyz(gas_point * position_scale, self.norm_fac)
            if object_physics is not None:
                # _object_drag_defaults already applies the official
                # (point - norm_fac) / 2 transform to source-scale metadata.
                object_point_norm = np.asarray(object_physics["drag_point"], dtype=np.float32)[
                    0, :3
                ]
            else:
                object_point_norm = _norm_xyz(object_point * position_scale, self.norm_fac)
            drag_point = np.stack([gas_point_norm, object_point_norm], axis=0).astype(np.float32)
            drag_mask = np.stack([gas_mask, object_mask], axis=0).astype(np.float32)[..., None]
            physics_E = np.asarray([gas_E, object_E], dtype=np.float32)[:, None]
            physics_nu = np.asarray([gas_nu, object_nu], dtype=np.float32)[:, None]
            physics_mat_type = np.asarray([gas_mat_type, object_mat_type], dtype=np.int64)
            physics_floor_height = np.asarray([gas_floor, object_floor], dtype=np.float32)[:, None]
            physics_gravity = np.asarray([gas_gravity, object_gravity], dtype=np.int64)
            physics_coeff = np.concatenate(
                [np.ones((1, 1), dtype=np.float32), object_base_drag_coeff[:1]], axis=0
            )
            physics_volume = np.stack([gas_volume, object_volume], axis=0).astype(np.float32)[..., None]
            arrays = (
                gas_x,
                object_x,
                gas_v,
                object_v,
                points_tgt,
                force,
                drag_point,
                drag_mask,
                physics_E,
                physics_nu,
                physics_floor_height,
                physics_coeff,
                physics_volume,
                gas_target_velocity,
                gas_initial_velocity,
                gas_velocity_delta,
                interaction_delta_v,
                interaction_total_delta_v,
                interaction_transfer_delta_v,
                gas_advection_displacement,
                phase_deformation,
                phase_affine,
            )
            sample_valid = all(bool(np.isfinite(value).all()) for value in arrays)
            if not sample_valid:
                gas_x, _ = _finite_or_zero(gas_x)
                object_x, _ = _finite_or_zero(object_x)
                gas_v, _ = _finite_or_zero(gas_v)
                object_v, _ = _finite_or_zero(object_v)
                points_tgt, _ = _finite_or_zero(points_tgt)
                force, _ = _finite_or_zero(force)
                drag_point, _ = _finite_or_zero(drag_point)
                drag_mask, _ = _finite_or_zero(drag_mask)
                points_src = np.stack([gas_x[0], object_x[0]], axis=0)
                init_vel = np.stack([gas_v[0], object_v[0]], axis=0)
            meta = {
                "file": source_entry,
                "protocol": self.composite_protocol,
                "position_scale": position_scale,
                "velocity_scale": velocity_scale,
                "gas_force_scale": gas_force_scale,
                "gravity": physics_gravity.tolist(),
                "stored_protocol": stored_protocol,
                "sample_valid": sample_valid,
                "gas_gravity_raw": raw_gas_gravity,
                "gas_gravity_used": gas_gravity,
                "object_physics_source": (
                    object_physics.get("source_path", "missing")
                    if object_physics is not None
                    else "missing"
                ),
            }
            if aoss_uri is not None:
                meta["aoss_uri"] = aoss_uri
                meta["aoss_cache_path"] = path
            if self.return_physics and "F" in h5 and "C" in h5:
                meta["gas_F"] = np.asarray(h5["F"], dtype=np.float32)[future_indices]
                meta["gas_C"] = np.asarray(h5["C"], dtype=np.float32)[future_indices]

        data = {
            "points_src": torch.from_numpy(points_src).float(),
            "init_vel": torch.from_numpy(init_vel).float(),
            "points_tgt": torch.from_numpy(points_tgt).float(),
            "force": torch.from_numpy(force).float(),
            "drag_point": torch.from_numpy(drag_point).float(),
            "drag_mask": torch.from_numpy(drag_mask).float(),
            "gravity": torch.from_numpy(physics_gravity).long(),
            "physics_E": torch.from_numpy(physics_E).float(),
            "physics_nu": torch.from_numpy(physics_nu).float(),
            "physics_mat_type": torch.from_numpy(physics_mat_type).long(),
            "physics_floor_height": torch.from_numpy(physics_floor_height).float(),
            "physics_gravity": torch.from_numpy(physics_gravity).long(),
            "physics_coeff": torch.from_numpy(physics_coeff).float(),
            "physics_volume": torch.from_numpy(physics_volume).float(),
            "physics_F": torch.from_numpy(phase_deformation).float(),
            "physics_C": torch.from_numpy(phase_affine).float(),
            "sample_valid": torch.tensor(sample_valid, dtype=torch.bool),
            "phase_presence": torch.tensor([True, True], dtype=torch.bool),
            "gas_target_velocity": torch.from_numpy(gas_target_velocity).float(),
            "gas_initial_velocity": torch.from_numpy(gas_initial_velocity).float(),
            "gas_velocity_delta": torch.from_numpy(gas_velocity_delta).float(),
            "interaction_delta_v": torch.from_numpy(interaction_delta_v).float(),
            # Derived from existing H5 fields and raw velocities; no new H5
            # data fields are required. These match the sampled trajectory
            # interval discretization used by the loss.
            "interaction_total_delta_v": torch.from_numpy(
                interaction_total_delta_v
            ).float(),
            "interaction_transfer_delta_v": torch.from_numpy(
                interaction_transfer_delta_v
            ).float(),
            "interaction_alive": torch.from_numpy(interaction_alive).bool(),
            "interaction_valid": torch.from_numpy(interaction_valid).bool(),
            "interaction_dt": torch.tensor(interaction_dt, dtype=torch.float32),
            "sample_frame_steps": torch.from_numpy(sample_frame_steps).float(),
            "gas_advection_displacement": torch.from_numpy(
                gas_advection_displacement
            ).float(),
            "gas_advection_alive": torch.from_numpy(gas_advection_alive).bool(),
        }
        if object_physics is not None:
            data.update(
                {
                    # These names mirror the official PhysCtrl condition API,
                    # with an object_ prefix to avoid colliding with gas fields.
                    "object_E": torch.from_numpy(object_physics["E"]).float(),
                    "object_nu": torch.from_numpy(object_physics["nu"]).float(),
                    "object_mat_type": torch.from_numpy(object_physics["mat_type"]).long(),
                    "object_floor_height": torch.from_numpy(object_physics["floor_height"]).float(),
                    "object_gravity": torch.from_numpy(object_physics["gravity"]).long(),
                    "object_force": torch.from_numpy(object_physics["force"]).float(),
                    "object_drag_point": torch.from_numpy(object_physics["drag_point"]).float(),
                    "object_mask": torch.from_numpy(object_physics["mask"]).float(),
                    "object_base_drag_coeff": torch.from_numpy(object_physics["base_drag_coeff"]).float(),
                    "object_vol": torch.from_numpy(object_physics["vol"]).float(),
                    "object_physics_valid": torch.tensor(True, dtype=torch.bool),
                }
            )
        return data, meta
