# -*- coding: utf-8 -*-
"""
Batch 3D cytoskeleton analysis with:
1) Per-stack preprocessing (1–99% scale + Median + Anisotropic Gaussian)
2) ILEE_3d analysis per stack
3) Global K2 (opt_k2) on the WHOLE preprocessed folder
4) analyze_document_3D on the preprocessed folder
5) Export one final CSV to outputs
"""

from __future__ import annotations

import argparse
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

# MATLAB/GPU toggles
USE_MATLAB = False
USE_MATLAB_GPU = False

DEFAULTS = dict(
    input_folder=r"F:\\TIF_RAW\\DZ\\",
    out_root=r"F:\\TIF_RAW\\DZ_outputs",
    channel=None,
    k1=None,
    k2=None,
    median=1,
    pixel_size=0.1578,
    z_unit=0.12,
    g_thres_model="interaction",
    min_vol_um3=0.5,
)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def align_image_dimension(img: np.ndarray) -> np.ndarray:
    dims = np.array(img.shape)
    dim = len(dims)
    order = np.lexsort((np.arange(dim), dims))
    if dim == 3:
        return np.moveaxis(img, order, [0, 1, 2])
    if dim == 4:
        return np.moveaxis(img, order, [0, 1, 2, 3])
    raise ValueError(f"Only supports 3D/4D input, got {img.shape}")


def to_12bit(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0, 1) * 4095.0 + 0.5).astype(np.uint16)


def next_available(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suf = path.stem, path.suffix
    parent = path.parent
    i = 1
    while True:
        c = parent / f"{stem}_{i}{suf}"
        if not c.exists():
            return c
        i += 1


def preprocess_stack(zyx: np.ndarray, median_size: int) -> np.ndarray:
    # 1–99% 归一化
    p1, p99 = np.percentile(zyx, [1, 99])
    img01 = np.clip((zyx.astype(np.float32) - p1) / (p99 - p1 + 1e-8), 0, 1)

    # median 滤波（XY）
    if median_size and median_size > 1:
        img01 = median_filter(img01, size=(1, median_size, median_size))

    img12 = to_12bit(img01)
    return img12.astype(np.float32)


def iter_tiff_files(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in [".tif", ".tiff"]:
            yield p


# ------------------------------------------------------------
# Main per-stack processing
# ------------------------------------------------------------

def process_stack(tif_path: Path, args, preproc_root: Path) -> None:

    out_root = Path(args.out_root)
    out_dir = out_root / tif_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Reading:", tif_path.name)
    raw = imread(str(tif_path))
    aligned = align_image_dimension(raw)

    if aligned.ndim == 4:
        if args.channel is None:
            raise ValueError("Multi-channel image: please give --channel")
        zyx = aligned[args.channel]
    else:
        zyx = aligned

    px_um = float(args.pixel_size)
    z_um = float(args.z_unit) if args.z_unit is not None else px_um

    # ============================================================
    # AUTO Z-DOWNSAMPLING
    # ============================================================
    if z_um < px_um:
        skip = int(np.ceil(px_um / z_um))
        print(f"[AUTO] z_unit {z_um:.4f} µm < pixel_size {px_um:.4f} µm")
        print(f"[AUTO] Downsampling Z by factor = {skip}")
        print(f"[AUTO] Old shape: {zyx.shape}")

        zyx = zyx[::skip, :, :]
        z_um = z_um * skip

        print(f"[AUTO] New shape: {zyx.shape}")
        print(f"[AUTO] New z_unit: {z_um:.4f} µm")
    # ============================================================

    print("[INFO] Preprocessing: normalize + median")
    img_float = preprocess_stack(zyx, args.median)

    # Anisotropic Gaussian
    sigma_xy = 0.6
    sigma_z = max(0.0, 0.6 * (z_um / px_um))
    print(f"[INFO] Gaussian sigma = (z={sigma_z:.3f}, xy={sigma_xy:.3f})")

    smoothed = gaussian_filter(img_float,
                               sigma=(sigma_z, sigma_xy, sigma_xy))

    # Save to _preprocessed folder
    preproc_root.mkdir(parents=True, exist_ok=True)
    preproc_path = preproc_root / tif_path.name
    imwrite(preproc_path, smoothed.astype(np.float32))
    print(" - Saved preprocessed:", preproc_path)

    # Per-image K2 (only for ILEE_3d)
    k2 = float(args.k2) if args.k2 is not None else ILEE_CSK.opt_k2(
        str(preproc_path.parent),
        target_channel=args.channel,
    )

    if args.k1 is None:
        k1 = float(10 ** ((np.log10(2.5) + np.log10(k2)) / 2.0))
    else:
        k1 = float(args.k1)

    print(f"[INFO] Image K2={k2:.3f}  K1={k1:.3f}")

    # Run ILEE 3D
    print("[INFO] Running ILEE_3d ...")

    dif_stack = ILEE_CSK.ILEE_3d(
        smoothed,
        xy_unit=px_um,
        z_unit=z_um,
        k1=k1,
        k2=k2,
        use_matlab=USE_MATLAB,
        use_matlabGPU=USE_MATLAB_GPU,
        g_thres_model=args.g_thres_model,
    )

    mask = dif_stack > 0

    # Physical voxel volume
    vox_um3 = px_um * px_um * z_um
    min_vox = max(1, int(round(float(args.min_vol_um3) / vox_um3)))
    if min_vox > 1:
        mask = remove_small_objects(mask, min_size=min_vox)

    imwrite(next_available(out_dir / "3d-stack_dif_stack.tif"),
            dif_stack.astype(np.float32))
    imwrite(next_available(out_dir / "3d-stack_mask_stack.tif"),
            (mask.astype(np.uint8) * 255))

    print("[DONE] ILEE_3d finished:", out_dir)


# ------------------------------------------------------------
# Run entire folder
# ------------------------------------------------------------

def run(args):

    in_dir = Path(args.input_folder)
    out_root = Path(args.out_root)
    out_root.mkdir(exist_ok=True, parents=True)

    name = in_dir.name.rstrip("\\/")
    preproc_root = in_dir.parent / f"{name}_preprocessed"

    print("\n==============================")
    print("INPUT   :", in_dir)
    print("OUTPUT  :", out_root)
    print("PREPROC :", preproc_root)
    print("==============================")

    files = list(iter_tiff_files(in_dir))
    if not files:
        raise RuntimeError("No tif files found in input folder.")

    # --- STEP 1: Preprocess + ILEE_3d (per stack) ---
    for f in files:
        process_stack(f, args, preproc_root)

    # --- STEP 2: Global K2 from ALL PREPROCESSED ---
    print("\n[INFO] Calculating GLOBAL K2 using opt_k2 on:")
    print("       ", preproc_root)

    k2 = float(args.k2) if args.k2 is not None else ILEE_CSK.opt_k2(
        str(preproc_root),
        target_channel=args.channel,
    )

    if args.k1 is None:
        k1 = float(10 ** ((np.log10(2.5) + np.log10(k2)) / 2.0))
    else:
        k1 = float(args.k1)

    print(f"\n[GLOBAL] K2 = {k2:.4f}")
    print(f"[GLOBAL] K1 = {k1:.4f}")

    px_um = float(args.pixel_size)
    z_um_orig = float(args.z_unit) if args.z_unit is not None else px_um

    if z_um_orig < px_um:
        skip = int(np.ceil(px_um / z_um_orig))
        z_um_eff = z_um_orig * skip
        print(f"[AUTO] analyze_document_3D z_unit = {z_um_eff:.4f} µm "
              f"(skip={skip}, original z_unit={z_um_orig:.4f} µm)")
    else:
        z_um_eff = z_um_orig

    # --- STEP 3: analyze_document_3D (General CSV) ---
    print("\n[INFO] Running analyze_document_3D on preprocessed folder...")

    df = analyze_document_3D(
        folder_path=str(preproc_root),
        obj_channel=args.channel,
        k1=k1,
        k2=k2,
        xy_unit=px_um,
        z_unit=z_um_eff,
        pixel_size=px_um,
        single_k1=True,
        use_GPU=USE_MATLAB_GPU,
    )

    csv_path = next_available(out_root / f"{name}_preprocessed_indices.csv")
    df.to_csv(csv_path, index=False)

    print("\n ==============================")
    print(" ALL FINISHED SUCCESSFULLY")
    print(" Preprocessed:", preproc_root)
    print(" CSV:", csv_path)
    print(" ==============================")


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

if __name__ == "__main__":

    ap = argparse.ArgumentParser("ILEE_CSK batch 3D pipeline")

    ap.add_argument("--input-folder", type=str)
    ap.add_argument("--out-root", type=str)
    ap.add_argument("--channel", type=int)
    ap.add_argument("--k1", type=float)
    ap.add_argument("--k2", type=float)
    ap.add_argument("--median", type=int)
    ap.add_argument("--pixel-size", type=float)
    ap.add_argument("--z-unit", type=float)
    ap.add_argument("--g-thres-model", type=str)
    ap.add_argument("--min-vol-um3", type=float)

    args = ap.parse_args()

    # Apply defaults
    for k, v in DEFAULTS.items():
        if getattr(args, k.replace("-", "_")) is None:
            setattr(args, k.replace("-", "_"), v)

    run(args)
