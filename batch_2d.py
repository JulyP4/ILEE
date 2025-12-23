"""
Batch 2D cytoskeleton analysis (slice or MIP) with per-image preprocessing,
ILEE_2d + analyze_actin_2d_standard, and aggregated workbook outputs.

Adds support for:
- Saving generated 2D images in mode-dependent sibling folders
- Recursively mirroring any input sub-folder structure
- Collecting optional non-standardized indices into the same row
- Writing a single Excel workbook with an all-images sheet and per-folder sheets
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
from scipy.ndimage import median_filter
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.color import gray2rgb
from skimage.morphology import binary_dilation, disk

import ILEE_CSK
from ILEE_CSK import analyze_actin_2d_standard

warnings.filterwarnings("ignore", category=FutureWarning)

# MATLAB/GPU toggles
USE_MATLAB = False
USE_MATLAB_GPU = False

DEFAULTS = dict(
    input_folder=r"F:\\TIF_RAW\\DZ\\",
    out_root=r"F:\\TIF_RAW\\DZ_MIP_outputs",
    mode="mip",  # 'mip' or 'slice'
    z_index=0,
    channel=None,
    k1=None,
    k2=None,
    median=1,
    pixel_size=1.0,
    g_thres_model="multilinear",
    include_nonstandarized_index=True,
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


def to_12bit(img01: np.ndarray) -> np.ndarray:
    return (np.clip(img01, 0, 1) * 4095.0 + 0.5).astype(np.uint16)


def safe_rescale01(img2d: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(img2d, [1, 99])
    return rescale_intensity(img2d.astype(np.float32), in_range=(p1, p99), out_range=(0, 1))


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


def iter_tiff_files(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in [".tif", ".tiff"]:
            yield p


def overlay_contour(base01, mask, width=2, color=(1.0, 0.0, 0.0)):
    base = gray2rgb(np.clip(base01, 0, 1))
    m = mask.astype(bool)
    edge = binary_dilation(m, footprint=disk(width)) ^ m
    out = base.copy()
    c = np.array(color, dtype=np.float32)
    out[edge] = c
    return out


def determine_preproc_root(in_dir: Path, mode: str, z_index: int, is_2d: bool) -> Tuple[Path, str]:
    name = in_dir.name.rstrip("\\/")
    if is_2d:
        suffix = "_preprocessed_2d"
    elif mode == "mip":
        suffix = "_MIP"
    elif mode == "slice":
        suffix = f"_z{z_index}"
    else:
        suffix = "_preprocessed_2d"
    return in_dir.parent / f"{name}{suffix}", suffix


# ------------------------------------------------------------
# Main per-image processing
# ------------------------------------------------------------

def preprocess_image(img2d: np.ndarray, median_size: int) -> np.ndarray:
    img01 = safe_rescale01(img2d)
    if median_size and median_size > 1:
        img01 = median_filter(img01, size=median_size)
    img12 = to_12bit(img01)
    return img12.astype(np.float32)


def save_outputs(tag: str, out_dir: Path, img01: np.ndarray, mask: np.ndarray) -> None:
    tif_mask = next_available(out_dir / f"{tag}_mask.tif")
    png_mask = next_available(out_dir / f"{tag}_mask.png")
    png_viz = next_available(out_dir / f"{tag}_input_viz.png")
    png_over = next_available(out_dir / f"{tag}_overlay.png")

    imwrite(tif_mask, (mask.astype(np.uint8) * 255))
    imwrite(png_mask, mask.astype(np.uint8))
    viz01 = equalize_adapthist(np.clip(img01, 0, 1), clip_limit=0.01)
    imwrite(png_viz, (np.clip(viz01, 0, 1) * 255).astype(np.uint8))
    over = overlay_contour(viz01, mask, width=2, color=(1.0, 0.0, 0.0))
    imwrite(png_over, (np.clip(over, 0, 1) * 255).astype(np.uint8))

    print(" -", tif_mask.name, "/", png_mask.name, "/", png_viz.name, "/", png_over.name)


def get_processed_image(aligned: np.ndarray, args) -> np.ndarray:
    if aligned.ndim == 4:
        if args.channel is None:
            raise ValueError("Multi-channel image: please give --channel")
        zyx = aligned[args.channel]
    elif aligned.ndim == 3:
        zyx = aligned
    else:
        zyx = aligned

    if zyx.ndim == 3:
        mode = args.mode.lower()
        if mode == "mip":
            img2d = np.max(zyx, axis=0)
        elif mode == "slice":
            zi = int(args.z_index)
            if not (0 <= zi < zyx.shape[0]):
                raise ValueError(f"--z-index must be between 0 and {zyx.shape[0]-1}")
            img2d = zyx[zi]
        else:
            raise ValueError("Mode must be 'mip' or 'slice'")
    else:
        img2d = zyx
    return img2d


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
    include_nsi: bool,
) -> Tuple[pd.DataFrame, Path]:
    input_root = Path(args.input_folder)
    rel_to_input = tif_path.relative_to(input_root)
    rel_dir = rel_to_input.parent

    out_dir = out_root / rel_dir / tif_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[INFO] Reading:", tif_path)
    raw = imread(str(tif_path))
    aligned = align_image_dimension(raw)

    img2d = get_processed_image(aligned, args)
    output_tag = "preprocessed_2d" if aligned.ndim == 2 else args.mode.lower()

    print("[INFO] Preprocessing: normalize + optional median")
    preproc_dir = preproc_root / rel_dir
    preproc_dir.mkdir(parents=True, exist_ok=True)
    img_float = preprocess_image(img2d, args.median)
    preproc_path = preproc_dir / tif_path.name
    imwrite(preproc_path, img_float.astype(np.float32))
    print(" - Saved preprocessed_2d:", preproc_path)

    print("[INFO] Running ILEE_2d ...")
    img_dif = ILEE_CSK.ILEE_2d(
        img_float.astype(np.uint16),
        k2=k2,
        k1=k1,
        pL_type="pL_8",
        gauss_dif=True,
    )
    mask = img_dif > 0

    save_outputs(output_tag, out_dir, img_float / 4095.0, mask)

    print("[INFO] Running analyze_actin_2d_standard ...")
    analysis_result = analyze_actin_2d_standard(
        img=img_float,
        img_dif=img_dif,
        pixel_size=float(args.pixel_size),
        exclude_true_blank=False,
    )
    merged_row = merge_analysis_rows(analysis_result, include_nsi)
    merged_row["file_path"] = str(tif_path)
    merged_row["file_name"] = tif_path.name
    merged_row["folder_key"] = folder_key

    all_cols = list(merged_row.columns)
    middle_cols = [c for c in all_cols if c not in ["file_name", "folder_key", "file_path"]]
    merged_row = merged_row[["file_name", "folder_key", *middle_cols, "file_path"]]

    print("[DONE] Finished:", out_dir)
    return merged_row, preproc_path


# ------------------------------------------------------------
# Run entire folder
# ------------------------------------------------------------

def collect_files_with_keys(in_dir: Path) -> List[Tuple[Path, str]]:
    files_with_keys: List[Tuple[Path, str]] = []
    for tif_path in iter_tiff_files(in_dir):
        rel_dir = tif_path.parent.relative_to(in_dir)
        folder_key = "/".join(rel_dir.parts) if rel_dir.parts else in_dir.name
        files_with_keys.append((tif_path, folder_key))
    return files_with_keys


def sanitize_sheet_name(name: str, used_names: Set[str]) -> str:
    """Return a sheet name compatible with Excel constraints and unique within the workbook."""

    invalid = "[]:*?/\\"
    cleaned = "".join("_" if c in invalid else c for c in name)
    cleaned = cleaned.strip()
    if not cleaned:
        cleaned = "sheet"
    cleaned = cleaned[:31]

    base = cleaned
    i = 1
    while cleaned in used_names:
        suffix = f"_{i}"
        cleaned = (base[: (31 - len(suffix))] if len(base) + len(suffix) > 31 else base) + suffix
        i += 1

    used_names.add(cleaned)
    return cleaned


def run(args):
    in_dir = Path(args.input_folder)
    out_root = Path(args.out_root)
    out_root.mkdir(exist_ok=True, parents=True)

    files_with_keys = collect_files_with_keys(in_dir)
    if not files_with_keys:
        raise RuntimeError("No tif files found in input folder.")

    sample_img = align_image_dimension(imread(str(files_with_keys[0][0])))
    is_2d_only = sample_img.ndim == 2

    preproc_root, suffix = determine_preproc_root(in_dir, args.mode.lower(), args.z_index, is_2d_only)
    preproc_root.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print("INPUT   :", in_dir)
    print("OUTPUT  :", out_root)
    print("PREPROC :", preproc_root)
    print("==============================")

    print("\n[INFO] Calculating GLOBAL K2 using opt_k2 on all samples ...")

    if args.k2 is not None:
        k2 = float(args.k2)
    else:
        tmp_dir = out_root / "_tmp_k2"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            for idx, (f, _) in enumerate(files_with_keys):
                img = imread(str(f))
                if is_2d_only:
                    img = img[np.newaxis, ...]
                imwrite(tmp_dir / f"sample_{idx}.tif", img)
            k2 = ILEE_CSK.opt_k2(
                str(tmp_dir),
                target_channel=None if is_2d_only else args.channel,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    k1 = float(args.k1) if args.k1 is not None else 2.5

    print(f"[GLOBAL] K2 = {k2:.4f}")
    print(f"[GLOBAL] K1 = {k1:.4f}")

    folder_results: Dict[str, List[pd.DataFrame]] = {}

    for tif_path, folder_key in files_with_keys:
        merged_row, preproc_path = process_file(
            tif_path,
            args,
            preproc_root,
            out_root,
            k1,
            k2,
            folder_key,
            args.include_nonstandarized_index,
        )
        folder_results.setdefault(folder_key, []).append(merged_row)

    all_rows = [row for rows in folder_results.values() for row in rows]
    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    excel_name = f"{in_dir.name}{suffix}_indices.xlsx"
    excel_path = next_available(out_root / excel_name)
    used_sheet_names: Set[str] = set()

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name=sanitize_sheet_name("all_images", used_sheet_names), index=False)
        for folder_key, rows in sorted(folder_results.items()):
            df_folder = pd.concat(rows, ignore_index=True)
            sheet_name = sanitize_sheet_name(folder_key if folder_key else "root", used_sheet_names)
            df_folder.to_excel(writer, sheet_name=sheet_name, index=False)

    print("\n[INFO] Saved combined workbook:", excel_path)

    print("\n ==============================")
    print(" ALL FINISHED SUCCESSFULLY")
    print(" Preprocessed_2D root:", preproc_root)
    print(" Workbook:", excel_path)
    print(" ==============================")


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser("ILEE_CSK batch 2D pipeline")

    ap.add_argument("--input-folder", type=str)
    ap.add_argument("--out-root", type=str)
    ap.add_argument("--mode", choices=["mip", "slice"], help="Use Z-MIP or Z-slice")
    ap.add_argument("--z-index", type=int, help="Z index for slice mode")
    ap.add_argument("--channel", type=int)
    ap.add_argument("--k1", type=float)
    ap.add_argument("--k2", type=float)
    ap.add_argument("--median", type=int)
    ap.add_argument("--pixel-size", type=float)
    ap.add_argument("--g-thres-model", type=str)
    ap.add_argument(
        "--include-nonstandarized-index",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include non-standardized indices in the output row",
    )

    args = ap.parse_args()

    for k, v in DEFAULTS.items():
        attr = k.replace("-", "_")
        if getattr(args, attr) is None:
            setattr(args, attr, v)

    run(args)