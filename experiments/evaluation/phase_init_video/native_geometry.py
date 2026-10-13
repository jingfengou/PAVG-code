"""Camera bookkeeping for separately reconstructed phases of one image."""

import numpy as np


def project(points, projection, crop):
    points = np.asarray(points, dtype=np.float64)
    clip = np.concatenate([points, np.ones((*points.shape[:-1], 1))], axis=-1)
    clip = clip @ np.asarray(projection, dtype=np.float64)
    if np.any(np.abs(clip[..., 3]) < 1e-9):
        raise ValueError("Points project onto the camera plane")
    ndc = clip[..., :3] / clip[..., 3:4]
    y, x, size = np.asarray(crop, dtype=np.float64)
    result = ndc.copy()
    result[..., 0] = ((ndc[..., 0] + 1) * size - 1) / 2 + x
    result[..., 1] = ((ndc[..., 1] + 1) * size - 1) / 2 + y
    return result


def register_crop(points, projection, source_crop, reference_crop):
    """Preserve pixel XY and virtual-camera NDC depth, not metric gas depth."""
    pixel = project(points, projection, source_crop)
    y, x, size = np.asarray(reference_crop, dtype=np.float64)
    ndc = pixel.copy()
    ndc[..., 0] = (2 * (pixel[..., 0] - x) + 1) / size - 1
    ndc[..., 1] = (2 * (pixel[..., 1] - y) + 1) / size - 1
    homogeneous = np.concatenate([ndc, np.ones((*ndc.shape[:-1], 1))], axis=-1)
    world = homogeneous @ np.linalg.inv(np.asarray(projection, dtype=np.float64))
    if np.any(np.abs(world[..., 3]) < 1e-9):
        raise ValueError("Unprojection produces a point at infinity")
    result = world[..., :3] / world[..., 3:4]
    error = np.max(np.abs(project(result, projection, reference_crop) - pixel))
    if error > 1e-6 or not np.isfinite(result).all():
        raise ValueError(f"Crop registration failed: {error}")
    return result


def floor_rotation(normal):
    normal = np.asarray(normal, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise ValueError("Invalid floor normal")
    magnitude = np.linalg.norm(normal)
    if magnitude < 1e-9:
        raise ValueError("Floor normal must be nonzero")
    normal = normal / magnitude
    target = np.asarray([0., 1., 0.])
    v = np.cross(normal, target)
    s, c = np.linalg.norm(v), float(normal @ target)
    if s < 1e-8:
        # A proper 180-degree rotation, not the upstream -I reflection.
        return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    x, y, z = v
    k = np.asarray([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + k + k @ k * ((1 - c) / s ** 2)


def to_model(world, center, scale, rotation):
    return ((np.asarray(world) - center) * scale / 2) @ rotation.T


def to_world(model, center, scale, rotation):
    return (np.asarray(model) @ rotation) * (2 / scale) + center


def model_point_on_pixel_ray(pixel, model_depth, projection, crop, center, scale, rotation):
    """Resolve a reviewed pixel on an explicit gravity-aligned depth plane."""
    pixel = np.asarray(pixel, dtype=np.float64)
    if pixel.shape != (2,) or not np.isfinite(pixel).all() or scale <= 0:
        raise ValueError("Invalid pixel or position scale")
    y, x, size = np.asarray(crop, dtype=np.float64)
    ndc = (2 * (pixel - [x, y]) + 1) / size - 1
    h = np.array([[*ndc, 0., 1.], [*ndc, .8, 1.]]) @ np.linalg.inv(
        np.asarray(projection, dtype=np.float64))
    world = h[:, :3] / h[:, 3:]
    model = to_model(world, center, scale, rotation)
    delta = model[1] - model[0]
    if abs(delta[2]) < 1e-8:
        raise ValueError("Pixel ray parallel to model-depth plane")
    point = model[0] + delta * ((model_depth - model[0, 2]) / delta[2])
    restored = project(to_world(point[None], center, scale, rotation), projection, crop)
    np.testing.assert_allclose(restored[0, :2], pixel, atol=1e-6)
    return point


def upsample(initial, future):
    if future.shape != (24, *initial.shape):
        raise ValueError("Expected 24 future frames")
    result = np.empty((49, *initial.shape), dtype=np.float32)
    keyframes = np.concatenate([initial[None], future], axis=0)
    result[::2] = keyframes
    result[1::2] = (keyframes[:-1] + keyframes[1:]) / 2
    return result


def support_fraction(pixels, mask):
    pixels = np.asarray(pixels)
    x, y = np.rint(pixels[:, :2]).astype(np.int64).T
    valid = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    supported = np.zeros(len(pixels), dtype=bool)
    supported[valid] = mask[y[valid], x[valid]] > 16
    return float(supported.mean())
