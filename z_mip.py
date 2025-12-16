# -*- coding: utf-8 -*-
"""Generate Z-axis maximum intensity projections (MIP) from 3D/4D TIFFs."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
from skimage.exposure import rescale_intensity


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


def z_mip(tif_path: Path, out_path: Path, channel: int | None = None) -> Path:
    raw = imread(str(tif_path))
    aligned = align_image_dimension(raw)

    if aligned.ndim == 4:
        if channel is None:
            raise ValueError("Multi-channel image: please give --channel")
        zyx = aligned[channel]
    elif aligned.ndim == 3:
        zyx = aligned
    else:
        raise ValueError("Input must be 3D/4D for Z MIP")

    mip = np.max(zyx, axis=0)
    # robust rescale for visualization
    p1, p99 = np.percentile(mip, [1, 99])
    mip_u16 = rescale_intensity(mip.astype(np.float32), in_range=(p1, p99), out_range=(0, 65535)).astype(np.uint16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(out_path, mip_u16)
    return out_path


def iter_tiff_files(root: Path) -> list[Path]:
    tiff_exts = {".tif", ".tiff"}
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in tiff_exts
    ]


def process_dir(input_dir: Path, output_dir: Path, channel: int | None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for tif_path in sorted(iter_tiff_files(input_dir)):
        rel = tif_path.relative_to(input_dir)
        out_path = (output_dir / rel).with_name(f"{rel.stem}_mip{tif_path.suffix}")
        saved_paths.append(z_mip(tif_path, out_path, channel=channel))
    return saved_paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Z-axis maximum intensity projection")
    ap.add_argument("--input", required=True, help="Input 3D/4D TIFF file or directory")
    ap.add_argument(
        "--output",
        help=(
            "Output path. For files: <input>_mip.tif by default; "
            "for directories: <input>_mip directory mirroring the input tree."
        ),
    )
    ap.add_argument("--channel", type=int, help="Channel index for 4D input")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")

    if in_path.is_dir():
        out_dir = Path(args.output) if args.output else in_path.with_name(f"{in_path.name}_mip")
        saved = process_dir(in_path, out_dir, channel=args.channel)
        if saved:
            print(f"Saved {len(saved)} Z-MIPs under {out_dir}")
        else:
            print("No TIFF files found to process.")
    else:
        out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_mip.tif")
        saved = z_mip(in_path, out_path, channel=args.channel)
        print("Saved Z-MIP:", saved)
