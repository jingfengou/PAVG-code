"""Project restored 3D predictions using retained shared-camera geometry."""
from __future__ import annotations
import numpy as np
from native_geometry import project, to_world, upsample


def project_native(phase, prediction, slot: int, phase_name: str, device="cpu"):
    import torch
    world = np.asarray(phase[f"{phase_name}_registered_world"], dtype=np.float32)
    indices = np.asarray(phase[f"{phase_name}_fps_indices"], dtype=np.int64)
    initial = np.asarray(phase["points_src"], dtype=np.float32)[slot]
    trajectory = upsample(initial, prediction[:, slot])
    original = world[indices]
    drivers = to_world(trajectory, np.asarray(phase["shared_center"]),
        float(np.asarray(phase["shared_scale"]).item()), np.asarray(phase["shared_rotation"]))
    if np.max(np.abs(drivers[0] - original)) > 1e-4:
        raise ValueError(f"{phase_name}: registered geometry does not match frame 0")
    driver_tensor = torch.from_numpy(original).to(device)
    neighbors = []
    with torch.inference_mode():
        for start in range(0, len(world), 4096):
            query = torch.from_numpy(world[start:start + 4096]).to(device)
            neighbors.append(torch.cdist(query, driver_tensor).topk(
                8, largest=False, dim=-1).indices.cpu().numpy())
    nearest = np.concatenate(neighbors, axis=0)
    projected = []
    for frame in drivers:
        animated = world + (frame - original)[nearest].mean(axis=1)
        projected.append(project(animated, phase["shared_projection"], phase["shared_crop"]))
    return np.asarray(projected, dtype=np.float32)
