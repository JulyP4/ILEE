# -*- coding: utf-8 -*-
"""
Batch 3D cytoskeleton analysis using the General API.

- Uses ``opt_k2`` to recommend a universal K2 for the folder when not provided
- Derives K1 from K2 when omitted (recommended formula in docs)
- Runs ``analyze_document_3D`` once on the folder to export indices
- Keeps a top-level ``DEFAULTS`` block for quick manual edits without CLI flags

Reference: https://github.com/phylars/ILEE_CSK/wiki/API#api
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import ILEE_CSK
from ILEE_CSK import analyze_document_3D


# Editable defaults for quick runs
DEFAULTS = dict(
    input_folder=r"E:\\NTU\\filament\\ILEE\\batch_3d/",  # Folder containing TIFF stacks (must end with "/")
    obj_channel=0,  # Channel index of cytoskeleton fluorescence
    k2=None,  # Universal K2; when None, use opt_k2 on the folder
    k1=None,  # Universal K1; when None, compute from K2 per docs
    xy_unit=1.0,  # μm per pixel (XY)
    z_unit=1.0,  # μm per slice (Z step)
    pixel_size=1.0,  # Should equal xy_unit when using real μm units
    single_k1=True,  # Use single-K mode to save time
    oversampling_for_bundle=False,  # Disable heavy oversampling by default
    use_GPU=False,  # MATLAB GPU acceleration (if available)
    g_thres_model="multilinear",  # Gradient threshold model
    include_nonstandarized_index=False,  # Add developmental indices
    out_root=r"E:\\NTU\\filament\\ILEE\\outputs\\batch\\3d",  # Where to save result tables
    out_csv="results_3d.csv",  # Output CSV filename
    out_excel=None,  # Optional Excel filename
)


def ensure_trailing_slash(path: Path) -> str:
    """The General API expects folder paths ending with '/'."""

    s = path.as_posix()
    return s if s.endswith("/") else s + "/"


def estimate_k2(folder: Path, target_channel: Optional[int]) -> float:
    print("[INFO] K2 not provided; running opt_k2 on folder ...")
    k2 = ILEE_CSK.opt_k2(folder_path=ensure_trailing_slash(folder), target_channel=target_channel)
    print(f"[INFO] Optimal K2 = {float(k2):.4f}")
    return float(k2)


def derive_k1_from_k2(k2: float) -> float:
    return float(10 ** ((np.log10(2.5) + np.log10(k2)) / 2.0))


def run(args):
    in_dir = Path(args.input_folder)
    if not in_dir.is_dir():
        raise FileNotFoundError(in_dir)

    obj_channel = args.obj_channel
    k2 = float(args.k2) if args.k2 is not None else estimate_k2(in_dir, obj_channel)
    k1 = float(args.k1) if args.k1 is not None else derive_k1_from_k2(k2)

    print("[INFO] Using K1 =", k1)
    print("[INFO] Running analyze_document_3D on folder ...")

    df = analyze_document_3D(
        folder_path=ensure_trailing_slash(in_dir),
        obj_channel=obj_channel,
        k1=k1,
        k2=k2,
        xy_unit=float(args.xy_unit),
        z_unit=float(args.z_unit),
        pixel_size=float(args.pixel_size),
        single_k1=bool(args.single_k1),
        oversampling_for_bundle=bool(args.oversampling_for_bundle),
        use_GPU=bool(args.use_GPU),
        g_thres_model=args.g_thres_model,
        include_nonstandarized_index=bool(args.include_nonstandarized_index),
    )

    out_dir = Path(args.out_root) if args.out_root else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.out_csv:
        csv_path = out_dir / args.out_csv
        df.to_csv(csv_path, index=False)
        print("[DONE] CSV:", csv_path)

    if args.out_excel:
        excel_path = out_dir / args.out_excel
        df.to_excel(excel_path, index=False)
        print("[DONE] Excel:", excel_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser("ILEE_CSK batch 3D (General API)")
    ap.add_argument("--input-folder")
    ap.add_argument("--obj-channel", type=int)
    ap.add_argument("--k2", type=float)
    ap.add_argument("--k1", type=float)
    ap.add_argument("--xy-unit", type=float)
    ap.add_argument("--z-unit", type=float)
    ap.add_argument("--pixel-size", type=float)
    ap.add_argument("--single-k1", action="store_true", default=None)
    ap.add_argument("--oversampling-for-bundle", action="store_true", default=None)
    ap.add_argument("--use-GPU", dest="use_GPU", action="store_true", default=None)
    ap.add_argument("--g-thres-model", choices=["multilinear", "interaction"])
    ap.add_argument("--include-nonstandarized-index", action="store_true", default=None)
    ap.add_argument("--out-root")
    ap.add_argument("--out-csv")
    ap.add_argument("--out-excel")
    args = ap.parse_args()

    # Fill CLI gaps with DEFAULTS for convenient manual editing
    for k, v in DEFAULTS.items():
        attr = k.replace("-", "_")
        cur = getattr(args, attr, None)
        if cur is None:
            setattr(args, attr, v)

    run(args)
