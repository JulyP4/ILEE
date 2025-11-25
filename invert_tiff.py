# -*- coding: utf-8 -*-
"""
Perform grayscale inversion for 2D/3D/4D images, and safely write as OME-TIFF.
- Prefer to detect axes order from OME/metadata;
- If missing, infer automatically: smallest dimension = channel (C), next smallest = Z,
  others = Y, X (preserving their relative order to minimize flipping);
- Final output is standardized to Z×C×Y×X and saved with metadata axes='ZCYX'
  to avoid display misalignment in viewers such as napari.
"""

from pathlib import Path
import numpy as np
from tifffile import imread, imwrite, TiffFile


# ===== Configuration =====
INPUT_PATH = Path(r"test_iamges/Polymerized_actin-1.tif")
# ==========================


def read_with_axes(path: Path):
    """
    Read image and try to extract axes metadata if available.
    Returns: (data, axes_str or None)
    """
    data = imread(path)
    axes = None
    try:
        with TiffFile(str(path)) as tf:
            omexml = tf.ome_metadata
            if omexml and "OME" in omexml:
                axes = tf.series[0].axes  # e.g. 'TCZYX', 'CZYX', 'ZYX', ...
            else:
                if tf.series and hasattr(tf.series[0], "axes"):
                    axes = tf.series[0].axes
    except Exception:
        pass
    return data, axes


def auto_axes_4d(shape):
    """
    When there is no axes metadata, infer for 4D arrays heuristically.
    Rules:
      - Smallest dimension → C (channel)
      - Second smallest → Z
      - Remaining two dimensions → Y, X (preserve order)
    Returns: (guessed_axes_str, permute_list_to_ZCYX)
    """
    dims = np.array(shape)
    order_idx = np.argsort(dims)
    axis_names = [None] * 4
    axis_names[order_idx[0]] = 'C'
    axis_names[order_idx[1]] = 'Z'

    remaining = [i for i in range(4) if axis_names[i] is None]
    remaining.sort()
    axis_names[remaining[0]] = 'Y'
    axis_names[remaining[1]] = 'X'

    guessed = ''.join(axis_names)
    permute = [guessed.index(ax) for ax in 'ZCYX']
    return guessed, permute


def axes_to_permute(axes):
    """
    Compute permutation from known axes string to ZCYX order.
    Handles strings of 3–5 dimensions (e.g. may include T).
    Returns: (core_axes_str, permute_list or None)
    """
    axes = axes.upper()
    core = ''.join([a for a in axes if a in 'ZCYX'])
    if not core:
        return None, None
    return core, None


def ensure_zcyx(data, axes_hint=None):
    """
    Convert any 2D/3D/4D image to standardized Z×C×Y×X.
    Returns: (array, axes='ZCYX')
    """
    arr = np.asarray(data)
    nd = arr.ndim

    if axes_hint:
        core, _ = axes_to_permute(axes_hint)
        if core:
            if len(core) == nd:
                for ax in 'ZCYX':
                    if ax not in core:
                        arr = np.expand_dims(arr, axis=0)
                        core = ax + core
                permute = [core.index(ax) for ax in 'ZCYX']
                try:
                    out = np.transpose(arr, permute)
                    return out, 'ZCYX'
                except Exception:
                    pass

    if nd == 2:
        out = arr[None, None, ...]
        return out, 'ZCYX'
    elif nd == 3:
        d0, d1, d2 = arr.shape
        if d0 <= 4:
            out = arr[None, ...]      # Assume C×Y×X
            return out, 'ZCYX'
        else:
            out = arr[:, None, ...]   # Assume Z×Y×X
            return out, 'ZCYX'
    elif nd == 4:
        guessed, permute = auto_axes_4d(arr.shape)
        out = np.transpose(arr, permute)
        return out, 'ZCYX'
    else:
        raise ValueError(f"Only 2–4D images are supported, got {arr.shape}")


def invert_image(data: np.ndarray) -> np.ndarray:
    """Perform grayscale inversion preserving dtype semantics."""
    if data.dtype == np.uint8:
        return 255 - data
    elif data.dtype == np.uint16:
        return 65535 - data
    elif np.issubdtype(data.dtype, np.floating):
        a_min, a_max = float(data.min()), float(data.max())
        return (a_max + a_min) - data
    else:
        arr = data.astype(np.float32, copy=False)
        a_min, a_max = float(arr.min()), float(arr.max())
        inv = (a_max + a_min) - arr
        return inv


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    print(f"[INFO] Reading: {INPUT_PATH}")
    data, axes_hint = read_with_axes(INPUT_PATH)
    print(f"[INFO] Original shape: {data.shape}, dtype: {data.dtype}, axes_hint: {axes_hint}")

    zcyx, axes = ensure_zcyx(data, axes_hint)
    print(f"[INFO] Standardized to {axes}, shape: {zcyx.shape}")

    inverted = invert_image(zcyx).astype(zcyx.dtype, copy=False)

    out_path = INPUT_PATH.with_name(INPUT_PATH.stem + "_inverted.tif")
    imwrite(
        out_path,
        inverted,
        metadata={"axes": "ZCYX"},
        ome=True
    )
    print(f"[DONE] Saved: {out_path}")
    print(f"[INFO] Output shape: {inverted.shape}, dtype: {inverted.dtype}, axes='ZCYX'")


if __name__ == "__main__":
    main()
