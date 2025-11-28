# -*- coding: utf-8 -*-
"""
Batch 3D cytoskeleton analysis using the same preprocessing and ILEE_3d
workflow as ``main.py``.

- Input: a folder of 3D/4D TIFF stacks (optionally multi-channel)
- Per-stack outputs: DIF stack TIFF, binary mask stack TIFF, and General API
  indices CSV from ``analyze_document_3D``
- K2 is auto-optimized per stack when omitted; K1 is derived from K2 when
  omitted (recommended formula)
"""

from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
from tifffile import imread, imwrite
from scipy.ndimage import median_filter, gaussian_filter
from skimage.morphology import remove_small_objects

import ILEE_CSK
from ILEE_CSK import analyze_document_3D

warnings.filterwarnings("ignore", category=FutureWarning)

# MATLAB/GPU toggles (advanced users)
USE_MATLAB = False
USE_MATLAB_GPU = False

DEFAULTS = dict(
    input_folder=r"E:\\NTU\\filament\\ILEE\\batch_3d/",  # Folder containing TIFF stacks
    out_root=r"E:\\NTU\\filament\\ILEE\\outputs\\batch\\3d",  # Root for outputs
    channel=0,  # Channel index when stacks are multi-channel (4D)
    k1=None,  # Auto-computed from K2 when None
    k2=None,  # Auto-optimized per stack when None
    median=1,  # XY median filter size (odd int; 1 disables)
    pixel_size=1.0,  # μm per pixel (XY)
    z_unit=None,  # μm per slice (defaults to pixel_size when None)
    g_thres_model="interaction",  # 'multilinear' or 'interaction'
    min_vol_um3=0.5,  # Remove 3D specks smaller than this volume (μm^3)
    single_k=False,  # Use single K2 per stack for General API
)


def align_image_dimension(img: np.ndarray) -> np.ndarray:
    """Align dims by ascending size: 3D -> (Z,Y,X); 4D -> (C,Z,Y,X)."""

    dims = np.array(img.shape)
    dim = len(dims)
    order = np.lexsort((np.arange(dim), dims))
    if dim == 3:
        return np.moveaxis(img, order, [0, 1, 2])
    if dim == 4:
        return np.moveaxis(img, order, [0, 1, 2, 3])
    raise ValueError(f"Supports only 3D/4D input, got {img.shape}")


def to_12bit(img01: np.ndarray) -> np.ndarray:
    """Convert normalized 0..1 float image to 12-bit uint16 (0..4095)."""

    return (np.clip(img01, 0, 1) * 4095.0 + 0.5).astype(np.uint16)


def next_available(path: Path) -> Path:
    """Append _1/_2/... if path exists; return an available path."""

    if not path.exists():
        return path
    stem, suf = path.stem, path.suffix
    parent = path.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suf}"
        if not cand.exists():
            return cand
        i += 1


def auto_opt_k2_single_file(tif_path: Path, target_channel: int, tmp_dir: Path) -> float:
    """Estimate K2 using ILEE_CSK.opt_k2() on a temp copy of the stack."""

    tmp = tmp_dir / "_k2tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.glob("*"):
        try:
            p.unlink()
        except Exception:
            pass
    shutil.copy2(tif_path, tmp / tif_path.name)
    print("[K2] Running ILEE_CSK.opt_k2 ...")
    k2 = ILEE_CSK.opt_k2(str(tmp), target_channel=target_channel)
    print(f"[K2] Optimal K2 = {float(k2):.3f}")
    return float(k2)


def run_general_3d_indices(stack_float: np.ndarray, tag: str, out_dir: Path, px_um: float, z_unit: float, k2: float, single_k: bool = False):
    """Save a temporary stack and run analyze_document_3D for indices."""

    tmp_dir = out_dir / "_tmp_general_3d"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tif_tmp = tmp_dir / f"{tag}_tmp_stack.tif"
    imwrite(tif_tmp, stack_float.astype(np.float32))

    print("[INFO] Running General API (analyze_document_3D) ...")
    df_general = analyze_document_3D(
        folder_path=str(tmp_dir),
        obj_channel=0,
        k2=k2,
        xy_unit=px_um,
        z_unit=z_unit,
        pixel_size=px_um,
        single_k=single_k,
        use_GPU=False,
    )

    csv_indices = next_available(out_dir / f"{tag}_indices.csv")
    df_general.to_csv(csv_indices, index=False)
    print(" -", csv_indices.name, "(general 3D indices)")


def preprocess_stack(zyx: np.ndarray, median_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply volume-wise 1–99% rescale, optional XY median, and 12-bit conversion."""

    p1, p99 = np.percentile(zyx, [1, 99])
    img01_stack = np.clip((zyx.astype(np.float32) - p1) / (p99 - p1 + 1e-8), 0, 1)
    if median_size and median_size > 1:
        img01_stack = median_filter(img01_stack, size=(1, median_size, median_size))

    img12_stack = to_12bit(img01_stack)
    img_float_stack = img12_stack.astype(np.float32)
    return img01_stack, img_float_stack


def process_stack(tif_path: Path, args) -> None:
    out_root = Path(args.out_root)
    out_dir = out_root / tif_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Reading:", tif_path)
    raw = imread(str(tif_path))
    aligned = align_image_dimension(raw)

    if aligned.ndim == 4:
        C = aligned.shape[0]
        ch = args.channel
        if ch is None or not (0 <= ch < C):
            raise ValueError(f"Image has {C} channels. Use --channel 0..{C-1}")
        zyx = aligned[ch]
        target_ch = ch
        print(f"[INFO] Using channel: {ch}/{C}")
    else:
        zyx = aligned
        target_ch = 0

    k2 = float(args.k2) if args.k2 is not None else auto_opt_k2_single_file(tif_path, target_ch, out_dir)
    if args.k1 is None:
        k1 = float(10 ** ((np.log10(2.5) + np.log10(k2)) / 2.0))
        print(f"[INFO] K1 not provided; using recommended K1 = {k1:.4f} (from K2={k2:.3f})")
    else:
        k1 = float(args.k1)
        print(f"[INFO] Using K1 = {k1}")

    px_um = float(args.pixel_size)
    z_unit = float(args.z_unit) if args.z_unit is not None else px_um

    print("[INFO] 3D preprocessing (volume-wise 1–99% + optional XY median + anisotropic Gaussian) ...")
    _, img_float_stack = preprocess_stack(zyx, args.median)

    sigma_xy = 0.6
    sigma_z = max(0.0, 0.6 * (z_unit / px_um))
    smoothed = gaussian_filter(img_float_stack, sigma=(sigma_z, sigma_xy, sigma_xy))

    print("[INFO] Running ILEE_3d ...")
    img_dif_stack = ILEE_CSK.ILEE_3d(
        smoothed,
        xy_unit=px_um,
        z_unit=z_unit,
        k1=k1,
        k2=k2,
        single_k1=False,
        use_matlab=USE_MATLAB,
        use_matlabGPU=USE_MATLAB_GPU,
        gauss_dif=True,
        g_thres_model=args.g_thres_model,
    )

    mask = (img_dif_stack > 0)
    vox_um3 = px_um * px_um * z_unit
    min_vox = max(1, int(round(float(args.min_vol_um3) / vox_um3)))
    if min_vox > 1:
        mask = remove_small_objects(mask, min_size=min_vox, connectivity=1)
    mask_stack = mask.astype(np.uint8)

    dif_path = next_available(out_dir / "3d-stack_dif_stack.tif")
    mask_path = next_available(out_dir / "3d-stack_mask_stack.tif")
    imwrite(dif_path, img_dif_stack.astype(np.float32))
    imwrite(mask_path, (mask_stack * 255).astype(np.uint8))
    print(" -", dif_path.name)
    print(" -", mask_path.name)

    try:
        run_general_3d_indices(img_float_stack, tif_path.stem, out_dir, px_um, z_unit, k2, single_k=args.single_k)
    except Exception as e:
        print("[WARN] analyze_document_3D failed:", e)

    print("[DONE] 3D stack output:", out_dir.resolve())


def iter_tiff_files(folder: Path) -> Iterable[Path]:
    exts = {".tif", ".tiff"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def run(args):
    in_dir = Path(args.input_folder)
    if not in_dir.is_dir():
        raise FileNotFoundError(in_dir)

    out_root = Path(args.out_root) if args.out_root else in_dir
    out_root.mkdir(parents=True, exist_ok=True)

    files = list(iter_tiff_files(in_dir))
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {in_dir}")

    for tif_path in files:
        process_stack(tif_path, args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser("ILEE_CSK batch 3D (main.py workflow)")
    ap.add_argument("--input-folder")
    ap.add_argument("--out-root")
    ap.add_argument("--channel", type=int)
    ap.add_argument("--k1", type=float)
    ap.add_argument("--k2", type=float)
    ap.add_argument("--median", type=int)
    ap.add_argument("--pixel-size", type=float)
    ap.add_argument("--z-unit", type=float)
    ap.add_argument("--g-thres-model", choices=["multilinear", "interaction"])
    ap.add_argument("--min-vol-um3", type=float)
    ap.add_argument("--single-k", action="store_true", help="Use a single K2 per stack for General API")
    args = ap.parse_args()

    for k, v in DEFAULTS.items():
        cur = getattr(args, k.replace("-", "_"), None)
        if cur is None:
            setattr(args, k.replace("-", "_"), v)

    run(args)
