"""
Save a single 2D composite tiff for a given z-scan level and optical z index.
Stitches y-tiles in Y order, extracts one optical z plane, saves as uint16 tiff.
Missing tiles are skipped.
Uses zarr lazy loading — only reads the needed z slice, memory efficient.
Tiff layout: (X, Y, Z_optical) i.e. axes QYX with 23627 pages of (509, 638)
"""

import re
import os
import argparse
import numpy as np
import zarr
from pathlib import Path
import tifffile as tiff

Z_RE = re.compile(r"z(\d+)", re.IGNORECASE)
Y_RE = re.compile(r"y(\d+)", re.IGNORECASE)


def _parse_zy(folder: Path):
    s = folder.name
    mz = Z_RE.search(s)
    my = Y_RE.search(s)
    if not (mz and my):
        return None
    return int(mz.group(1)), int(my.group(1))


def read_single_plane(vol_path: Path, z_opt: int, channel: int) -> np.ndarray:
    """
    Lazily open tiff as zarr and slice only the z_opt plane.
    Shape is (X, Y, Z_optical) -> slice [:, :, z_opt] -> (X, Y)
    """
    store = tiff.imread(str(vol_path), aszarr=True)
    z = zarr.open(store, mode='r')
    print(f"    {vol_path.name}: shape={z.shape} dtype={z.dtype}")

    if z.ndim == 4:
        # (C, X, Y, Z_optical)
        plane = np.asarray(z[channel, :, :, z_opt], dtype=np.float32)
    elif z.ndim == 3:
        # (X, Y, Z_optical)
        plane = np.asarray(z[:, :, z_opt], dtype=np.float32)
    else:
        raise ValueError(f"Unexpected ndim={z.ndim} for {vol_path}")

    store.close()
    return plane


def save_composite_plane(
    root_dir: str,
    output_dir: str,
    z_scan: int,
    z_opt: int,
    channel: int = 0,
    folder_glob: str = "*-088.fli*_transformed*",
):
    os.makedirs(output_dir, exist_ok=True)

    root = Path(root_dir)
    folders = [p for p in root.glob(f"**/{folder_glob}") if p.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No folders matching {folder_glob} under {root}")

    vol_name = f"vol_colors_ch{channel}.tif"

    # Collect y-tiles for the requested z-scan level
    y_tiles = {}
    for f in folders:
        zy = _parse_zy(f)
        if zy is None:
            continue
        z_idx, y_idx = zy
        if z_idx != z_scan:
            continue
        vp = f / vol_name
        if not vp.exists():
            print(f"  Skipping {f.name} — {vol_name} not found")
            continue
        y_tiles[y_idx] = vp

    if not y_tiles:
        raise RuntimeError(f"No tiles found for z_scan={z_scan} with {vol_name}")

    y_keys = sorted(y_tiles.keys())
    print(f"z_scan={z_scan:02d}: found {len(y_keys)} y-tiles -> {y_keys}")

    # Read one plane at a time per tile
    planes = []
    for y_idx in y_keys:
        plane = read_single_plane(y_tiles[y_idx], z_opt, channel)
        print(f"  y{y_idx:03d}: plane shape={plane.shape} min={plane.min():.1f} max={plane.max():.1f}")
        planes.append(plane)

    # planes are (X, Y) -> concatenate along Y axis=1
    composite = np.concatenate(planes, axis=1)  # (X, Y_total)
    print(f"  Composite shape: {composite.shape} min={composite.min():.1f} max={composite.max():.1f}")

    out_u16 = np.clip(composite, 0, 65535).round().astype(np.uint16)
    out_path = os.path.join(output_dir, f"composite_zscan{z_scan:02d}_zopt{z_opt:04d}_ch{channel}.tif")
    tiff.imwrite(out_path, out_u16, photometric="minisblack")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a single 2D composite tiff for a given z-scan level and optical z index."
    )
    parser.add_argument("--root_dir",    required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--z_scan",      type=int, required=True)
    parser.add_argument("--z_opt",       type=int, required=True)
    parser.add_argument("--channel",     type=int, default=0)
    parser.add_argument("--folder_glob", default="*-088.fli*_transformed*")

    args = parser.parse_args()

    save_composite_plane(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        z_scan=args.z_scan,
        z_opt=args.z_opt,
        channel=args.channel,
        folder_glob=args.folder_glob,
    )