import numpy as np


def _to_numpy(x):
    return x.detach().cpu().numpy()


def to_masked_point_map(pts3d, mask):
    pts3d = _to_numpy(pts3d)
    mask = _to_numpy(mask)
    assert mask is not None, "mask is None"
    assert mask.shape[0] == pts3d.shape[0], f"{mask.shape[0]} != {pts3d.shape[0]}"
    assert mask.shape[1] == pts3d.shape[1], f"{mask.shape[1]} != {pts3d.shape[1]}"

    for p, m in zip(pts3d, mask):
        p[~m, :] = 0

    return pts3d


def _clamp_0_1(colors):
    if not isinstance(colors, np.ndarray):
        colors = colors.astype(float) / 255
    if np.issubdtype(colors.dtype, np.floating):
        pass
    assert 0 <= colors.min() and colors.max() <= 1, f"{colors.min()} {colors.max()}"
    return colors


def to_pcl_color(pts3d, color, mask):
    pts3d = _to_numpy(pts3d)
    mask = _to_numpy(mask)
    if mask is None:
        mask = [slice(None)] * len(pts3d)

    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(color, mask)])
    return pts.reshape(-1, 3), _clamp_0_1(col.reshape(-1, 3))


def to_depth_map(depth, img, mask):
    def to_numpy(x):
        return x.detach().cpu().numpy()

    depth = to_numpy(depth).reshape(img.shape[0], img.shape[1])
    mask = to_numpy(mask)

    for p, m in zip(depth, mask):
        p[~m] = 0

    return depth
