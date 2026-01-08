# -*- coding: utf-8 -*-
"""
Robust 2D actin analysis with ILEE_CSK + scientific-grade masking.

输出文件严格限制为每个样本目录仅 5 个：
- dif.tif
- mask.tif
- overlay.png
- mask_high.png
- mask_low.png

说明：
- dif.tif：ILEE difference image（float32）
- mask.tif：最终分析用mask（uint8: 0/255）
- mask_low.png / mask_high.png：阈值调试用（uint8: 0/255）
- overlay.png：可视化叠加（仅展示用）

默认不再保存任何其它 png/tif（例如 *_mask.png, *_input_viz.png, *_img_dif.png, *_mask_selected.png 等）
默认也不保存 preprocessed_2d tif（可用 --save-preprocessed true 开启）
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from scipy.ndimage import gaussian_filter, median_filter, label as ndi_label

from skimage.exposure import equalize_adapthist
from skimage.color import gray2rgb
from skimage.morphology import (
    binary_dilation, disk,
    remove_small_objects, remove_small_holes,
    binary_opening, binary_closing,
)
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma

import ILEE_CSK
from ILEE_CSK import analyze_actin_2d_standard

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULTS = dict(
    input_folder=r"F:/20251120_3_Percent_Ficoll/test/test_10",
    out_root=None,
    mode="mip",          # 'mip' or 'slice' (for 3D stacks); ignored for pure 2D
    z_index=0,
    channel=None,        # for 4D (C,Z,Y,X) data
    k1=2.5,
    k2=50,               # if None, compute per-folder via opt_k2

    # ---------------------------
    # Preprocess controls (ILEE input)
    # ---------------------------
    disable_percentile_norm=True,  # TRUE minimal preprocess by default
    norm_p_low=3.0,                # only used if disable_percentile_norm=False
    norm_p_high=97.0,              # only used if disable_percentile_norm=False
    median=1,                      # <=1 disables median
    denoise_method="none",         # 'none'|'gaussian'|'bilateral'|'nlm'
    gaussian_sigma=0.4,
    bilateral_sigma_color=0.05,
    bilateral_sigma_spatial=2.0,
    bilateral_bins=10000,
    nlm_h=0.8,
    nlm_patch_size=5,
    nlm_patch_distance=7,
    nlm_fast=True,

    # ---------------------------
    # ILEE controls
    # ---------------------------
    gauss_dif=True,
    pL_type="pL_8",

    pixel_size=1.0,
    include_nonstandarized_index=True,

    # ---------------------------
    # Mask strategy
    # ---------------------------
    mask_strategy="two_stage",     # 'none' | 'single' | 'two_stage'
    single_pos_percentile=65.0,    # only if mask_strategy='single'

    low_pct=45.0,                  # two-stage: low threshold percentile (positive dif only)
    high_pct=95.0,                 # two-stage: seed threshold percentile (positive dif only)
    keep_isolated_min_size=0,      # keep components >= this size even if not connected to seeds (0 disables)

    connectivity=8,                # 4 or 8 for 2D connectivity
    min_object_size=6,             # remove small objects after selection
    fill_small_holes=16,            # fill small holes after selection

    # border smoothing (reduce jagged edges, small radii recommended)
    closing_radius=1,              # 0 disables
    opening_radius=0,              # 0 disables

    # ---------------------------
    # Saving controls
    # ---------------------------
    save_preprocessed=False,       # 默认不保存预处理tif（避免额外图像输出）
)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def align_image_dimension(img: np.ndarray) -> np.ndarray:
    dims = np.array(img.shape)
    dim = len(dims)
    order = np.lexsort((np.arange(dim), dims))
    if dim == 2:
        return img
    if dim == 3:
        return np.moveaxis(img, order, [0, 1, 2])
    if dim == 4:
        return np.moveaxis(img, order, [0, 1, 2, 3])
    raise ValueError(f"Only supports 2D/3D/4D input, got {img.shape}")

def ilee_safe_norm01(img2d: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    img = img2d.astype(np.float32, copy=False)
    lo, hi = np.percentile(img, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(img)), float(np.max(img))
        if hi <= lo:
            return np.zeros_like(img, dtype=np.float32)
    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-6)
    return np.clip(img, 0.0, 1.0)

def next_available(path: Path) -> Path:
    """Used for non-image outputs like Excel to avoid overwrite."""
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

def iter_tiff_files(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in [".tif", ".tiff"]:
            yield p

def overlay_contour(base01: np.ndarray, mask: np.ndarray, width: int = 2, color=(1.0, 0.0, 0.0)) -> np.ndarray:
    base = gray2rgb(np.clip(base01, 0, 1))
    m = mask.astype(bool)
    edge = binary_dilation(m, footprint=disk(width)) ^ m
    out = base.copy()
    out[edge] = np.array(color, dtype=np.float32)
    return out

def determine_preproc_root(in_dir: Path) -> Path:
    name = in_dir.name.rstrip("\\/")
    return in_dir.parent / f"{name}_preprocessed_2d"

def make_out_root(in_dir: Path, out_root: str | None) -> Path:
    if out_root is None:
        return in_dir.parent / f"{in_dir.name}_outputs"
    return Path(out_root)

def get_2d_from_any(aligned: np.ndarray, args) -> np.ndarray:
    if aligned.ndim == 4:
        if args.channel is None:
            raise ValueError("4D image: please specify --channel")
        zyx = aligned[args.channel]
    elif aligned.ndim == 3:
        zyx = aligned
    else:
        zyx = aligned

    if zyx.ndim == 3:
        if args.mode.lower() == "mip":
            return np.max(zyx, axis=0)
        elif args.mode.lower() == "slice":
            zi = int(args.z_index)
            if not (0 <= zi < zyx.shape[0]):
                raise ValueError(f"--z-index must be between 0 and {zyx.shape[0]-1}")
            return zyx[zi]
        else:
            raise ValueError("--mode must be mip or slice")
    return zyx

# ------------------------------------------------------------
# Preprocess (ILEE input)
# ------------------------------------------------------------

def preprocess_image(img2d: np.ndarray, args) -> np.ndarray:
    """
    Returns float32 image scaled to [0..4095] (12-bit), suitable for ILEE_2d.

    TRUE minimal preprocess (B) if disable_percentile_norm=True:
      - optional median / optional denoise (default none)
      - only scales by global max -> 0..4095
      - no percentile clipping, no [0,1] normalization
    """
    x = img2d.astype(np.float32, copy=False)

    # optional median (salt-pepper only)
    if args.median and int(args.median) > 1:
        x = median_filter(x, size=int(args.median))

    method = str(args.denoise_method).lower()

    if method == "gaussian":
        sig = float(args.gaussian_sigma) if args.gaussian_sigma is not None else 0.0
        if sig > 0:
            x = gaussian_filter(x, sigma=sig)

    elif method == "bilateral":
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if x_max > x_min:
            x01 = (x - x_min) / (x_max - x_min + 1e-6)
        else:
            x01 = np.zeros_like(x, dtype=np.float32)
        x01 = denoise_bilateral(
            x01,
            sigma_color=float(args.bilateral_sigma_color),
            sigma_spatial=float(args.bilateral_sigma_spatial),
            bins=int(args.bilateral_bins),
            channel_axis=None,
        ).astype(np.float32)
        x = x01 * (x_max - x_min) + x_min

    elif method == "nlm":
        x_min, x_max = float(np.min(x)), float(np.max(x))
        if x_max > x_min:
            x01 = (x - x_min) / (x_max - x_min + 1e-6)
        else:
            x01 = np.zeros_like(x, dtype=np.float32)
        sigma_est = float(np.mean(estimate_sigma(x01, channel_axis=None)))
        h = float(args.nlm_h) * sigma_est if sigma_est > 0 else 0.01
        x01 = denoise_nl_means(
            x01,
            h=h,
            patch_size=int(args.nlm_patch_size),
            patch_distance=int(args.nlm_patch_distance),
            fast_mode=bool(args.nlm_fast),
            channel_axis=None,
        ).astype(np.float32)
        x = x01 * (x_max - x_min) + x_min

    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown --denoise-method: {args.denoise_method}")

    if bool(args.disable_percentile_norm):
        mx = float(np.max(x))
        if mx <= 0 or not np.isfinite(mx):
            return np.zeros_like(x, dtype=np.float32)
        return (x / mx * 4095.0).astype(np.float32)

    # percentile mode
    img01 = ilee_safe_norm01(x, p_low=float(args.norm_p_low), p_high=float(args.norm_p_high))
    return (img01 * 4095.0).astype(np.float32)

# ------------------------------------------------------------
# Masking
# ------------------------------------------------------------

def _label_structure(connectivity: int):
    if connectivity == 4:
        return np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    return np.ones((3,3), dtype=np.uint8)  # 8-connectivity

def mask_single_threshold(img_dif: np.ndarray, pos_percentile: float) -> np.ndarray:
    dif = img_dif.astype(np.float32, copy=False)
    pos = dif[dif > 0]
    if pos.size < 64:
        return (dif > 0)
    th = float(np.percentile(pos, pos_percentile))
    return dif > th

def mask_two_stage_seeded(
    img_dif: np.ndarray,
    low_pct: float,
    high_pct: float,
    connectivity: int,
    keep_isolated_min_size: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-stage mask:
      - low = dif > P(low_pct)  (candidates)
      - high = dif > P(high_pct) (seeds)
      - Keep low-components that overlap high (seed-connected).
      - Optionally keep large isolated low-components (area >= keep_isolated_min_size)

    Returns: (low_mask, high_mask, selected_mask)
    """
    dif = img_dif.astype(np.float32, copy=False)
    pos = dif[dif > 0]
    if pos.size < 64:
        low = dif > 0
        return low, low, low

    low_th = float(np.percentile(pos, low_pct))
    high_th = float(np.percentile(pos, high_pct))
    if high_th < low_th:
        high_th = low_th

    low = dif > low_th
    high = dif > high_th

    struct = _label_structure(connectivity)
    lab, n = ndi_label(low.astype(np.uint8), structure=struct)
    if n == 0:
        return low, high, low

    touched_labels = np.unique(lab[high])
    touched_labels = touched_labels[touched_labels != 0]
    selected = np.isin(lab, touched_labels)

    if keep_isolated_min_size and keep_isolated_min_size > 0:
        counts = np.bincount(lab.ravel())
        large_labels = np.where(counts >= int(keep_isolated_min_size))[0]
        large_labels = large_labels[large_labels != 0]
        selected = selected | np.isin(lab, large_labels)

    return low, high, selected

def postprocess_mask(mask: np.ndarray, args) -> np.ndarray:
    m = mask.astype(bool)

    if args.min_object_size and int(args.min_object_size) > 0:
        m = remove_small_objects(m, min_size=int(args.min_object_size))

    if args.fill_small_holes and int(args.fill_small_holes) > 0:
        m = remove_small_holes(m, area_threshold=int(args.fill_small_holes))

    if args.closing_radius and int(args.closing_radius) > 0:
        m = binary_closing(m, footprint=disk(int(args.closing_radius)))

    if args.opening_radius and int(args.opening_radius) > 0:
        m = binary_opening(m, footprint=disk(int(args.opening_radius)))

    return m.astype(bool)

# ------------------------------------------------------------
# Saving helpers (ONLY 5 files)
# ------------------------------------------------------------

def save_dif(out_dir: Path, img_dif: np.ndarray) -> None:
    """Save only dif.tif (float32)."""
    imwrite(out_dir / "dif.tif", img_dif.astype(np.float32))

def save_low_high(out_dir: Path, low: np.ndarray, high: np.ndarray) -> None:
    """Save only mask_low.png & mask_high.png (uint8 0/255)."""
    imwrite(out_dir / "mask_low.png",  (low.astype(np.uint8) * 255))
    imwrite(out_dir / "mask_high.png", (high.astype(np.uint8) * 255))

def save_mask_and_overlay(out_dir: Path, img01_for_viz: np.ndarray, mask: np.ndarray) -> None:
    """
    Save:
    - mask.tif
    - overlay.png
    """
    imwrite(out_dir / "mask.tif", (mask.astype(np.uint8) * 255))

    viz01 = equalize_adapthist(np.clip(img01_for_viz, 0, 1), clip_limit=0.01)
    over = overlay_contour(viz01, mask, width=2, color=(1.0, 0.0, 0.0))
    imwrite(out_dir / "overlay.png", (np.clip(over, 0, 1) * 255).astype(np.uint8))

# ------------------------------------------------------------
# Pipeline per file
# ------------------------------------------------------------

def merge_analysis_rows(analysis_result, include_nsi: bool) -> pd.DataFrame:
    if isinstance(analysis_result, tuple) and len(analysis_result) >= 2:
        standard_row = analysis_result[0].reset_index(drop=True)
        if include_nsi and analysis_result[1] is not None:
            nsi_row = analysis_result[1].add_prefix("nsi_").reset_index(drop=True)
            return pd.concat([standard_row, nsi_row], axis=1)
        return standard_row
    if isinstance(analysis_result, pd.DataFrame):
        return analysis_result.reset_index(drop=True)
    raise ValueError("Unsupported analyze_actin_2d_standard return type")

def process_file(
    tif_path: Path,
    args,
    preproc_root: Path,
    out_root: Path,
    k1: float,
    k2: float,
    folder_key: str,
) -> pd.DataFrame:

    input_root = Path(args.input_folder)
    rel_to_input = tif_path.relative_to(input_root)
    rel_dir = rel_to_input.parent

    out_dir = out_root / rel_dir / tif_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Reading:", tif_path)
    raw = imread(str(tif_path))
    aligned = align_image_dimension(raw)
    img2d = get_2d_from_any(aligned, args)

    # preprocess (in-memory)
    img_float = preprocess_image(img2d, args)  # float32 0..4095

    # optional: save preprocessed copy (disabled by default)
    if bool(args.save_preprocessed):
        preproc_dir = preproc_root / rel_dir
        preproc_dir.mkdir(parents=True, exist_ok=True)
        preproc_path = preproc_dir / tif_path.name
        imwrite(preproc_path, img_float.astype(np.float32))
        print(" - Saved preprocessed_2d:", preproc_path)

    # ILEE difference
    print(f"[INFO] ILEE_2d (gauss_dif={args.gauss_dif}) ...")
    img_dif = ILEE_CSK.ILEE_2d(
        img_float.astype(np.uint16),
        k2=k2,
        k1=k1,
        pL_type=str(args.pL_type),
        gauss_dif=bool(args.gauss_dif),
    )

    # ---------- output 1: dif.tif ----------
    save_dif(out_dir, img_dif)

    # ---------- mask generation (NONE/SINGLE/TWO_STAGE) ----------
    if args.mask_strategy == "none":
        # 最接近“不处理”的对照：只取 ILEE 正响应区域
        low = high = selected = (img_dif > 0)
        raw_mask = selected
        mask = raw_mask.astype(bool)

    elif args.mask_strategy == "single":
        raw_mask = mask_single_threshold(img_dif, float(args.single_pos_percentile))
        low = high = raw_mask.copy()
        selected = raw_mask.copy()
        mask = postprocess_mask(raw_mask, args)

    else:  # two_stage
        low, high, selected = mask_two_stage_seeded(
            img_dif,
            low_pct=float(args.low_pct),
            high_pct=float(args.high_pct),
            connectivity=int(args.connectivity),
            keep_isolated_min_size=int(args.keep_isolated_min_size),
        )
        mask = postprocess_mask(selected, args)

    # ---------- output 2-3: mask_low.png, mask_high.png ----------
    save_low_high(out_dir, low, high)

    # ---------- output 4-5: mask.tif, overlay.png ----------
    img01_for_viz = np.clip(img_float / 4095.0, 0, 1)
    save_mask_and_overlay(out_dir, img01_for_viz, mask)

    # analysis
    print("[INFO] analyze_actin_2d_standard ...")
    analysis_result = analyze_actin_2d_standard(
        img=img_float,
        img_dif=img_dif,
        pixel_size=float(args.pixel_size),
        exclude_true_blank=False,
    )

    row = merge_analysis_rows(analysis_result, bool(args.include_nonstandarized_index))
    row["file_name"] = tif_path.name
    row["folder_key"] = folder_key
    row["k1"] = k1
    row["k2"] = k2
    row["file_path"] = str(tif_path)

    cols = list(row.columns)
    middle = [c for c in cols if c not in ["file_name", "folder_key", "k1", "k2", "file_path"]]
    row = row[["file_name", "folder_key", *middle, "k1", "k2", "file_path"]]

    return row

# ------------------------------------------------------------
# Run folder
# ------------------------------------------------------------

def collect_files_with_keys(in_dir: Path) -> List[Tuple[Path, str]]:
    out: List[Tuple[Path, str]] = []
    for tif_path in iter_tiff_files(in_dir):
        rel_dir = tif_path.parent.relative_to(in_dir)
        folder_key = "/".join(rel_dir.parts) if rel_dir.parts else in_dir.name
        out.append((tif_path, folder_key))
    return out

def sanitize_sheet_name(name: str, used: Set[str]) -> str:
    invalid = "[]:*?/\\"
    cleaned = "".join("_" if c in invalid else c for c in name).strip() or "sheet"
    cleaned = cleaned[:31]
    base = cleaned
    i = 1
    while cleaned in used:
        suffix = f"_{i}"
        cleaned = (base[: (31 - len(suffix))] + suffix) if len(base) + len(suffix) > 31 else (base + suffix)
        i += 1
    used.add(cleaned)
    return cleaned

def run(args):
    in_dir = Path(args.input_folder)
    out_root = make_out_root(in_dir, args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    preproc_root = determine_preproc_root(in_dir)
    preproc_root.mkdir(parents=True, exist_ok=True)

    files_with_keys = collect_files_with_keys(in_dir)
    if not files_with_keys:
        raise RuntimeError("No tif files found.")

    print("\n==============================")
    print("INPUT   :", in_dir)
    print("OUTPUT  :", out_root)
    print("PREPROC :", preproc_root, "(saved)" if args.save_preprocessed else "(not saved)")
    print("PREPROC MODE:", "B(minimal)" if args.disable_percentile_norm else "percentile")
    print("ILEE gauss_dif:", args.gauss_dif)
    print("MASK strategy:", args.mask_strategy)
    print("OUTPUT IMAGES: dif.tif, mask.tif, overlay.png, mask_low.png, mask_high.png")
    print("==============================")

    k1 = float(args.k1) if args.k1 is not None else 2.5

    # K2 per folder (or global)
    k2_by_folder: Dict[str, float] = {}
    if args.k2 is not None:
        k2_global = float(args.k2)
        for _, folder_key in files_with_keys:
            k2_by_folder[folder_key] = k2_global
        print(f"[K2] Using GLOBAL k2={k2_global:.4f}")
    else:
        print("[K2] Computing folder-level K2 via opt_k2 ...")
        files_by_folder: Dict[str, List[Path]] = {}
        for p, key in files_with_keys:
            files_by_folder.setdefault(key, []).append(p)

        tmp_base = out_root / "_tmp_k2"
        if tmp_base.exists():
            shutil.rmtree(tmp_base)
        tmp_base.mkdir(parents=True, exist_ok=True)

        try:
            for key, paths in files_by_folder.items():
                tmp_dir = tmp_base / (key.replace("/", "__") if key else "root")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                for i, p in enumerate(paths):
                    img = imread(str(p))
                    if img.ndim == 2:
                        img = img[np.newaxis, ...]
                    imwrite(tmp_dir / f"sample_{i}.tif", img)
                k2 = ILEE_CSK.opt_k2(str(tmp_dir), target_channel=None)
                k2_by_folder[key] = float(k2)
                print(f"  [FOLDER] {key or 'root'} k2={k2:.4f}")
        finally:
            shutil.rmtree(tmp_base, ignore_errors=True)

    rows_by_folder: Dict[str, List[pd.DataFrame]] = {}

    for tif_path, folder_key in files_with_keys:
        row = process_file(
            tif_path=tif_path,
            args=args,
            preproc_root=preproc_root,
            out_root=out_root,
            k1=k1,
            k2=k2_by_folder[folder_key],
            folder_key=folder_key,
        )
        rows_by_folder.setdefault(folder_key, []).append(row)

    all_rows = [r for rs in rows_by_folder.values() for r in rs]
    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    excel_path = next_available(out_root / f"{in_dir.name}_indices.xlsx")
    used: Set[str] = set()
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name=sanitize_sheet_name("all_images", used), index=False)
        for key, rs in sorted(rows_by_folder.items()):
            df = pd.concat(rs, ignore_index=True)
            df.to_excel(writer, sheet_name=sanitize_sheet_name(key or "root", used), index=False)

    print("\n[INFO] Saved workbook:", excel_path)
    print("[DONE] All finished.")

# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Robust ILEE 2D pipeline (strict 5-image outputs)")

    ap.add_argument("--input-folder", type=str)
    ap.add_argument("--out-root", type=str)
    ap.add_argument("--mode", choices=["mip", "slice"])
    ap.add_argument("--z-index", type=int)
    ap.add_argument("--channel", type=int)
    ap.add_argument("--k1", type=float)
    ap.add_argument("--k2", type=float)

    # preprocess
    ap.add_argument("--disable-percentile-norm", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--norm-p-low", type=float)
    ap.add_argument("--norm-p-high", type=float)
    ap.add_argument("--median", type=int)
    ap.add_argument("--denoise-method", choices=["none", "gaussian", "bilateral", "nlm"])
    ap.add_argument("--gaussian-sigma", type=float)
    ap.add_argument("--bilateral-sigma-color", type=float)
    ap.add_argument("--bilateral-sigma-spatial", type=float)
    ap.add_argument("--bilateral-bins", type=int)
    ap.add_argument("--nlm-h", type=float)
    ap.add_argument("--nlm-patch-size", type=int)
    ap.add_argument("--nlm-patch-distance", type=int)
    ap.add_argument("--nlm-fast", action=argparse.BooleanOptionalAction, default=None)

    # ILEE
    ap.add_argument("--gauss-dif", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--pL-type", type=str)

    ap.add_argument("--pixel-size", type=float)
    ap.add_argument("--include-nonstandarized-index", action=argparse.BooleanOptionalAction, default=None)

    # mask strategy
    ap.add_argument("--mask-strategy", choices=["none", "single", "two_stage"])
    ap.add_argument("--single-pos-percentile", type=float)

    ap.add_argument("--low-pct", type=float)
    ap.add_argument("--high-pct", type=float)
    ap.add_argument("--keep-isolated-min-size", type=int)

    ap.add_argument("--connectivity", choices=[4, 8], type=int)
    ap.add_argument("--min-object-size", type=int)
    ap.add_argument("--fill-small-holes", type=int)
    ap.add_argument("--closing-radius", type=int)
    ap.add_argument("--opening-radius", type=int)

    # save preprocessed copy
    ap.add_argument("--save-preprocessed", action=argparse.BooleanOptionalAction, default=None)

    args = ap.parse_args()

    # fill defaults
    for k, v in DEFAULTS.items():
        attr = k.replace("-", "_")
        if getattr(args, attr) is None:
            setattr(args, attr, v)

    # validation
    if args.mask_strategy == "two_stage":
        if float(args.low_pct) >= float(args.high_pct):
            raise ValueError("--low-pct must be < --high-pct for two_stage mask")

    run(args)
