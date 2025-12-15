# -*- coding: utf-8 -*-
"""
ILEE_CSK Minimal Script v6
- 2D (slice/MIP): ILEE_2d + official indices CSVs
- 3D (stack): ILEE_3d with robust preprocessing + official indices CSVs
- K2: if not provided, auto-optimize via ILEE_CSK.opt_k2() for ALL modes
- K1:
    * 2D default = 5 if not provided
    * 3D computed from K2 if not provided:
        K1 = 10 ** ((log10(2.5) + log10(K2)) / 2)
- 3D exports only a TIF mask (no PNGs)
"""

from pathlib import Path
import argparse
import warnings
import shutil
import numpy as np
import pandas as pd

from tifffile import imread, imwrite
from scipy.ndimage import median_filter, gaussian_filter
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.morphology import binary_dilation, disk, remove_small_objects

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa

import ILEE_CSK
from ILEE_CSK import analyze_actin_2d_standard, analyze_actin_3d_standard

warnings.filterwarnings("ignore", category=FutureWarning)

# =======================================================
# MATLAB/GPU toggle (advanced users)
# To use GPU, set USE_MATLAB=True and USE_MATLAB_GPU=True, and ensure:
#   - MATLAB is installed and callable
#   - Parallel Computing Toolbox
#   - NVIDIA GPU + drivers
USE_MATLAB = True
USE_MATLAB_GPU = True
# =======================================================

DEFAULTS = dict(
    input = r"F:\CZI_RAW\DZ\test_2\Experiment-2088-SIM Processing-1.tif",  # Input path (2D/3D/4D)
    out_root = r"F:\CZI_RAW\DZ\test_2",                                   # Output root folder
    mode = "3d-stack",     # '2d-slice' | '2d-mip' | '3d-stack'
    z_index = 14,          # Only for '2d-slice'
    channel =None,           # Channel index for 4D input
    k1 = None,             # 2D default=5; 3D auto-computed from K2 if None
    k2 = None,             # Auto-optimized via opt_k2() if None
    median = 1,            # Median kernel (odd int; 1 = disabled)
    pixel_size = 1.0,      # XY pixel size in μm (1.0 → pixel units)
    z_unit = None,         # Z-step in μm; defaults to pixel_size if None
    g_thres_model = "multilinear",  # 'multilinear' or 'interaction' (3D)
    min_vol_um3 = 0.5,     # Remove 3D specks smaller than this volume (μm^3)
)

# ---------- Helpers ----------
def align_image_dimension(img: np.ndarray) -> np.ndarray:
    """Align dims by ascending size: 3D -> (Z,Y,X); 4D -> (C,Z,Y,X)."""
    dims = np.array(img.shape)
    dim = len(dims)
    order = (
        pd.DataFrame({"size": dims, "idx": np.arange(dim)})
        .sort_values(["size", "idx"])
        ["idx"].to_numpy()
    )
    if dim == 3:
        return np.moveaxis(img, order, [0, 1, 2])
    elif dim == 4:
        return np.moveaxis(img, order, [0, 1, 2, 3])
    else:
        raise ValueError(f"Supports only 3D/4D input, got {img.shape}")

def to_12bit(img01: np.ndarray) -> np.ndarray:
    """Convert normalized 0..1 float image to 12-bit uint16 (0..4095)."""
    return (np.clip(img01, 0, 1) * 4095.0 + 0.5).astype(np.uint16)

def overlay_contour(base01, mask, width=2, color=(1.0, 0.0, 0.0)):
    """Overlay binary mask contours on grayscale image."""
    base = gray2rgb(np.clip(base01, 0, 1))
    m = mask.astype(bool)
    edge = binary_dilation(m, footprint=disk(width)) ^ m
    out = base.copy()
    c = np.array(color, dtype=np.float32)
    out[edge] = c
    return out

def auto_opt_k2_single_file(tif_path: Path, target_channel: int, tmp_dir: Path) -> float:
    """Estimate K2 using ILEE_CSK.opt_k2() on a temp copy of the file/folder."""
    tmp = tmp_dir / "_k2tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.glob("*"):
        try: p.unlink()
        except Exception: pass
    shutil.copy2(tif_path, tmp / tif_path.name)
    print("[K2] Running ILEE_CSK.opt_k2 ...")
    k2 = ILEE_CSK.opt_k2(str(tmp), target_channel=target_channel)
    print(f"[K2] Optimal K2 = {float(k2):.3f}")
    return float(k2)

def safe_rescale01(img2d: np.ndarray):
    """Rescale 1st–99th percentile to 0..1 for robust contrast (2D)."""
    p1, p99 = np.percentile(img2d, [1, 99])
    img01 = rescale_intensity(img2d.astype(np.float32), in_range=(p1, p99), out_range=(0, 1))
    return img01

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
# -----------------------------


def run(args):
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_root = Path(args.out_root)
    out_dir = out_root / in_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Reading:", in_path)
    raw = imread(str(in_path))
    print("[INFO] Shape:", raw.shape)

    aligned = align_image_dimension(raw)  # (Z,Y,X) or (C,Z,Y,X)
    if aligned.ndim == 4:
        C = aligned.shape[0]
        ch = args.channel
        if ch is None or not (0 <= ch < C):
            raise ValueError(f"Image has {C} channels. Use --channel 0..{C-1}")
        zyx = aligned[ch]  # (Z,Y,X)
        print(f"[INFO] Using channel: {ch}/{C}")
    else:
        zyx = aligned

    Z = zyx.shape[0]
    print(f"[INFO] Z depth: {Z} slices")
    mode = args.mode.lower()

    # ---- Auto-optimize K2 (ALL modes) ----
    if args.k2 is None:
        target_ch = args.channel if aligned.ndim == 4 else None
        print("[INFO] K2 not provided; auto-optimizing via opt_k2() ...")
        k2 = auto_opt_k2_single_file(in_path, target_ch, out_dir)
        print(f"[INFO] Using K2 = {k2}")
    else:
        k2 = float(args.k2)
        print(f"[INFO] Using K2 = {k2}")

    # ---- K1 handling ----
    if mode == "3d-stack":
        if args.k1 is None:
            # Recommended K1 for 3D from K2
            k1 = float(10 ** ((np.log10(2.5) + np.log10(k2)) / 2.0))
            print(f"[INFO] K1 not provided (3D); using recommended K1 = {k1:.4f} (from K2={k2:.3f})")
        else:
            k1 = float(args.k1)
            print(f"[INFO] Using K1 = {k1}")
    else:
        # 2D default
        k1 = float(args.k1) if args.k1 is not None else 5.0
        if args.k1 is None:
            print("[INFO] K1 not provided (2D); defaulting to K1 = 5.0")
        else:
            print(f"[INFO] Using K1 = {k1}")

    # Physical sizes & misc
    px_um = float(args.pixel_size)
    z_unit = float(args.z_unit) if args.z_unit is not None else px_um
    g_thres_model = args.g_thres_model
    min_vol_um3 = float(args.min_vol_um3)

    # Filename tag
    if mode == "2d-slice":
        tag = f"2d-slice_z{int(args.z_index):02d}"
    elif mode == "2d-mip":
        tag = "2d-mip"
    elif mode == "3d-stack":
        tag = "3d-stack"
    else:
        raise ValueError("Mode must be one of: 2d-slice / 2d-mip / 3d-stack")

    # =========================
    # 2D: single slice
    # =========================
    if mode == "2d-slice":
        zi = int(args.z_index)
        if not (0 <= zi < Z):
            raise ValueError(f"--z-index must be between 0 and {Z-1}")
        img2d = zyx[zi]

        img01 = safe_rescale01(img2d)
        if args.median and args.median > 1:
            img01 = median_filter(img01, size=args.median)

        img12 = to_12bit(img01)
        img_dif = ILEE_CSK.ILEE_2d(img12, k2=k2, k1=k1, pL_type="pL_8", gauss_dif=True)
        mask = (img_dif > 0)

        viz01 = equalize_adapthist(np.clip(img01, 0, 1), clip_limit=0.01)

        tif_mask = next_available(out_dir / f"{tag}_mask.tif")
        png_mask = next_available(out_dir / f"{tag}_mask.png")
        png_viz  = next_available(out_dir / f"{tag}_input_viz.png")
        png_over = next_available(out_dir / f"{tag}_overlay.png")

        imwrite(tif_mask, (mask.astype(np.uint8) * 255))
        plt.imsave(png_mask, mask, cmap="gray", vmin=0, vmax=1)
        plt.imsave(png_viz, viz01, cmap="gray", vmin=0, vmax=1)
        over = overlay_contour(viz01, mask, width=2, color=(1.0, 0.0, 0.0))
        plt.imsave(png_over, np.clip(over, 0, 1))

        # Official 2D indices
        try:
            df_general, df_dev = analyze_actin_2d_standard(img12, img_dif, pixel_size=px_um, exclude_true_blank=False)
            csv_indices = next_available(out_dir / f"{tag}_indices.csv")
            df_general.to_csv(csv_indices, index=False)
            csv_dev = next_available(out_dir / f"{tag}_indices_dev.csv")
            df_dev.to_csv(csv_dev, index=False)
            print(" -", csv_indices.name, "(cytoskeleton indices)")
            print(" -", csv_dev.name, "(non-biological indices)")
        except Exception as e:
            print("[WARN] analyze_actin_2d_standard failed:", e)

        print("[DONE] 2D slice output:", out_dir.resolve())
        print(" -", tif_mask.name, "/", png_mask.name, "/", png_viz.name, "/", png_over.name)

    # =========================
    # 2D: MIP
    # =========================
    elif mode == "2d-mip":
        img2d = np.max(zyx, axis=0)
        img01 = safe_rescale01(img2d)
        if args.median and args.median > 1:
            img01 = median_filter(img01, size=args.median)

        img12 = to_12bit(img01)
        img_dif = ILEE_CSK.ILEE_2d(img12, k2=k2, k1=k1, pL_type="pL_8", gauss_dif=True)
        mask = (img_dif > 0)

        viz01 = equalize_adapthist(np.clip(img01, 0, 1), clip_limit=0.01)

        tif_mask = next_available(out_dir / f"{tag}_mask.tif")
        png_mask = next_available(out_dir / f"{tag}_mask.png")
        png_viz  = next_available(out_dir / f"{tag}_input_viz.png")
        png_over = next_available(out_dir / f"{tag}_overlay.png")

        imwrite(tif_mask, (mask.astype(np.uint8) * 255))
        plt.imsave(png_mask, mask, cmap="gray", vmin=0, vmax=1)
        plt.imsave(png_viz, viz01, cmap="gray", vmin=0, vmax=1)
        over = overlay_contour(viz01, mask, width=2, color=(1.0, 0.0, 0.0))
        plt.imsave(png_over, np.clip(over, 0, 1))

        # Official 2D indices
        try:
            df_general, df_dev = analyze_actin_2d_standard(img12, img_dif, pixel_size=px_um, exclude_true_blank=False)
            csv_indices = next_available(out_dir / f"{tag}_indices.csv")
            df_general.to_csv(csv_indices, index=False)
            csv_dev = next_available(out_dir / f"{tag}_indices_dev.csv")
            df_dev.to_csv(csv_dev, index=False)
            print(" -", csv_indices.name, "(cytoskeleton indices)")
            print(" -", csv_dev.name, "(non-biological indices)")
        except Exception as e:
            print("[WARN] analyze_actin_2d_standard failed:", e)

        print("[DONE] 2D MIP output:", out_dir.resolve())
        print(" -", tif_mask.name, "/", png_mask.name, "/", png_viz.name, "/", png_over.name)

    # =========================
    # 3D: stack (ILEE_3d + robust preprocessing)
    # =========================
    elif mode == "3d-stack":
        print("[INFO] 3D preprocessing (volume-wise 1–99% + optional XY median + anisotropic Gaussian) ...")

        # 1) Volume-wise robust rescale
        p1, p99 = np.percentile(zyx, [1, 99])
        img01_stack = rescale_intensity(
            zyx.astype(np.float32), in_range=(p1, p99), out_range=(0, 1)
        )

        # 2) Optional XY median
        if args.median and args.median > 1:
            img01_stack = median_filter(img01_stack, size=(1, args.median, args.median))

        # 3) Convert like 2D
        img12_stack = to_12bit(img01_stack)              # uint16 [0..4095]
        img_float_stack = img12_stack.astype(np.float32) # float32 [0..4095]

        # 4) Light anisotropic Gaussian
        sigma_xy = 0.6
        sigma_z  = max(0.0, 0.6 * (z_unit / px_um))
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
            g_thres_model=g_thres_model,  # 'interaction' often less speckly
        )

        # 5) Binarize & remove small specks by physical volume
        mask = (img_dif_stack > 0)
        vox_um3 = (px_um * px_um * z_unit)
        min_vox = max(1, int(round(min_vol_um3 / vox_um3)))
        if min_vox > 1:
            mask = remove_small_objects(mask, min_size=min_vox, connectivity=1)
        mask_stack = mask.astype(np.uint8)

        tif_stack = next_available(out_dir / f"{tag}_mask_stack.tif")
        imwrite(tif_stack, (mask_stack * 255).astype(np.uint8))
        print(" -", tif_stack.name)

        # # 6) Official 3D indices CSV
        # try:
        #     df_general, df_dev = analyze_actin_3d_standard(
        #         img_float_stack,    # preprocessed 12-bit float volume (before smoothing)
        #         img_dif_stack,
        #         xy_unit=px_um,
        #         z_unit=z_unit,
        #         oversampling_for_bundle=True,
        #         pixel_size=px_um,
        #     )
        #     csv_indices = next_available(out_dir / f"{tag}_indices.csv")
        #     df_general.to_csv(csv_indices, index=False)
        #     csv_dev = next_available(out_dir / f"{tag}_indices_dev.csv")
        #     df_dev.to_csv(csv_dev, index=False)
        #     print(" -", csv_indices.name, "(cytoskeleton indices)")
        #     print(" -", csv_dev.name, "(non-biological indices)")
        # except Exception as e:
        #     print("[WARN] analyze_actin_3d_standard failed:", e)

        print("[DONE] 3D stack output:", out_dir.resolve())

    else:
        raise ValueError("Mode must be one of: 2d-slice / 2d-mip / 3d-stack")


if __name__ == "__main__":
    ap = argparse.ArgumentParser("ILEE_CSK minimal v6 (opt_k2 for all; 3D K1 auto from K2)")
    ap.add_argument("--input")
    ap.add_argument("--out-root")
    ap.add_argument("--mode", choices=["2d-slice", "2d-mip", "3d-stack"])
    ap.add_argument("--z-index", type=int)
    ap.add_argument("--channel", type=int)
    ap.add_argument("--k1", type=float)              # Optional; 2D default=5; 3D auto from K2
    ap.add_argument("--k2", type=float)              # Optional; auto via opt_k2 if omitted
    ap.add_argument("--median", type=int)
    ap.add_argument("--pixel-size", type=float)
    ap.add_argument("--z-unit", type=float, help="Z-step in μm; defaults to pixel-size")
    ap.add_argument("--g-thres-model", choices=["multilinear", "interaction"])
    ap.add_argument("--min-vol-um3", type=float, help="Remove 3D specks smaller than this volume (μm^3)")
    args = ap.parse_args()

    # Fill unspecified args from DEFAULTS
    for k, v in DEFAULTS.items():
        cur = getattr(args, k.replace("-", "_"), None)
        if cur is None:
            setattr(args, k.replace("-", "_"), v)

    run(args)
