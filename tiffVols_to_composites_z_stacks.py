"""
Stitches y-tiles for each z-scan level one at a time, then saves each
optical z-plane as an individual 2D tiff. This avoids loading the full
stitched volume into memory.

Memory footprint at any time: one z-scan slab (X, Z_optical, Y_stitched)
"""

import re
import os
import argparse
import numpy as np
import dask.array as da
from dask import delayed
from pathlib import Path
import tifffile as tiff
import scipy.io as sio
from typing import Tuple

# Reuse helpers from your existing stitch module
from tiffVols_to_composites import (
    _parse_zy,
    _load_vol_np,
    write_volume_tiff,
)

Z_RE = re.compile(r"z(\d+)", re.IGNORECASE)
Y_RE = re.compile(r"y(\d+)", re.IGNORECASE)


##########################################################################################
# Core: stitch y-tiles for a single z-scan level
##########################################################################################

def stitch_y_tiles_for_z(
    z_idx: int,
    y_list,                         # list of (y_idx, folder) for this z
    vol_name: str,
    tile_shape: tuple,
    crop_top: int,
    crop_bottom: int,
    crop_right: int,
    crop_left: int,
    channel: int,
    rotate: bool,
    z_shift_y: int,
    n_slabs: int,
    #final_y: int,
    y_len: int,
    pad_value: float = 0.0,
    y_concat_axis: int = 2,
    y_plane=None,
    z_plane=None,
) -> da.Array:
    """
    Stitch all y-tiles for one z-scan level into a single dask array.
    Returns slab of shape (X, Z_optical, Y_stitched) or (X, Y_stitched) if z_plane set.
    """
    tiles_da = []
    for y_idx, folder in sorted(y_list, key=lambda t: t[0]):
        vp = folder / vol_name
        if not vp.exists():
            raise FileNotFoundError(f"Missing {vol_name} in {folder}")

        d = delayed(_load_vol_np)(
            str(vp), crop_top, crop_bottom, crop_right, crop_left,
            channel, y_plane=y_plane, z_plane=z_plane
        )
        a = da.from_delayed(d, shape=tile_shape, dtype=np.float32)
        if rotate:
            a = da.rot90(a, k=2, axes=(1, 2))
        tiles_da.append(a)

    slab = da.concatenate(tiles_da, axis=y_concat_axis)

    # Apply y-shift padding consistent with full stitch
    #shift = int(z_shift_y)
    #off = z_idx * abs(shift)
    #if shift >= 0:
    #    left_pad  = off
    #    right_pad = final_y - (off + y_len)
    #else:
    #    left_pad  = final_y - (off + y_len)
    #    right_pad = off

    #slab = da.pad(
    #    slab,
    #    pad_width=((0, 0), (0, 0), (left_pad, right_pad)),
    #    mode="constant",
    #    constant_values=pad_value,
    #)

    print(f"  z{z_idx:02d}: y-tiles={len(tiles_da)} -> slab shape={slab.shape}")
    return slab


##########################################################################################
# Main loop: iterate z-scan levels, compute slab, save optical z planes
##########################################################################################

def save_optical_planes_per_zscan(
    root_dir: str,
    output_dir: str,
    kind: str = "nuclei",
    folder_glob: str = "*-272.fli*_transformed*",
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
    channel: int = 0,
    z_shift_y: int = -24,
    pad_value: float = 0.0,
    z_only: int = None,
    y_only: int = None,
    y_plane=None,
    z_plane=None,
):
    os.makedirs(output_dir, exist_ok=True)

    # Determine volume filename
    if 'laser_corrected' in folder_glob or 'transformed' in folder_glob:
        vol_name = "vol.tif" if kind.lower() in ("nuc", "nuclei") else f"vol_colors_ch{channel}.tif"
    elif 'bg_subtracted' in folder_glob:
        vol_name = "vol.tif" if kind.lower() in ("nuc", "nuclei") else "vol_colors.tif"
    else:
        vol_name = "vol.tif" if kind.lower() in ("nuc", "nuclei") else "vol_colors.tif"

    rotate = 'ch' in vol_name and 'ch0' not in vol_name

    # Discover folders
    root = Path(root_dir)
    folders = [p for p in root.glob(f"**/{folder_glob}") if p.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No folders matching {folder_glob} under {root}")

    # Group by z-scan index
    by_z = {}
    for f in folders:
        zy = _parse_zy(f)
        if zy is None:
            continue
        z_idx, y_idx = zy
        if z_only is not None and z_idx != z_only:
            continue
        if y_only is not None and y_idx != y_only:
            continue
        by_z.setdefault(z_idx, []).append((y_idx, f))

    if not by_z:
        raise RuntimeError("Found folders but couldn't parse zNN/yNN from names.")

    z_keys = sorted(by_z.keys())
    n_slabs = len(z_keys)
    print(f"Z-scan levels found: {n_slabs} -> {z_keys}")
    print(f"Using volume file: {vol_name}")

    # Infer tile shape from first available tile
    sample_folder = sorted(by_z[z_keys[0]], key=lambda t: t[0])[0][1]
    sample_path = sample_folder / vol_name
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing {vol_name} in sample folder {sample_folder}")

    sample_vol = _load_vol_np(
        str(sample_path), crop_top, crop_bottom, crop_right, crop_left,
        channel, y_plane=y_plane, z_plane=z_plane
    )
    tile_shape = sample_vol.shape
    print(f"Per-tile shape after crop: {tile_shape}")

    # Precompute y padding geometry (same as full stitch)
    #shift = int(z_shift_y)
    #y_len = int(tile_shape[2]) * len(by_z[z_keys[0]])  
    #final_y = y_len + abs(shift) * (n_slabs - 1)

    # Process one z-scan level at a time
    for z_idx in z_keys:
        #if z_idx != 5:
        #    continue
        y_list = by_z[z_idx]
        y_len_actual = tile_shape[2] * len(y_list)  # total Y before padding for this z

        print(f"\n--- Processing z-scan {z_idx:02d} ({len(y_list)} y-tiles) ---")

        slab_da = stitch_y_tiles_for_z(
            z_idx=z_idx,
            y_list=y_list,
            vol_name=vol_name,
            tile_shape=tile_shape,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            crop_right=crop_right,
            crop_left=crop_left,
            channel=channel,
            rotate=rotate,
            z_shift_y=z_shift_y,
            n_slabs=n_slabs,
            #final_y=final_y,
            y_len=y_len_actual,
            pad_value=pad_value,
            y_plane=y_plane,
            z_plane=z_plane,
        )

        print(f"  Computing slab z{z_idx:02d} into memory...")
        slab = slab_da.compute().astype(np.float32)  # (X, Z_optical, Y_stitched)
        print(f"  Slab shape: {slab.shape} | min={slab.min():.3f} max={slab.max():.3f}")

        # Save each optical z plane
        n_zopt = slab.shape[1]
        zscan_dir = os.path.join(output_dir, f"zscan{z_idx:02d}")
        os.makedirs(zscan_dir, exist_ok=True)

        for z_opt in range(n_zopt):
            plane = slab[:, z_opt, :]   # (X, Y_stitched)
            print(f"plane shape={plane.shape} min={plane.min():.1f} max={plane.max():.1f} nonzero={np.count_nonzero(plane)}")
            out_path = os.path.join(zscan_dir, f"zopt{z_opt:04d}.tif")
            plane_u16 = np.clip(plane, 0, 65535).round().astype(np.uint16)
            tiff.imwrite(out_path, plane_u16, bigtiff=True, photometric="minisblack")

        #n_zopt = slab_da.shape[1]
        #for z_opt in range(n_zopt):
        #    plane = slab_da[:, z_opt, :].compute().astype(np.float32)
        #    out_path = os.path.join(zscan_dir, f"zopt{z_opt:04d}.tif")
        #    plane_u16 = np.clip(plane, 0, 65535).round().astype(np.uint16)
        #    tiff.imwrite(out_path, plane_u16, bigtiff=True, photometric="minisblack")

        print(f"  Saved {n_zopt} optical planes to {zscan_dir}")

        # Explicitly free memory before next z-scan level
        del slab, slab_da

    print("\nDone.")


##########################################################################################
# CLI
##########################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stitch y-tiles per z-scan level and save individual optical z planes as tiffs."
    )
    parser.add_argument("source_root", help="Root directory containing z/y tile folders")
    parser.add_argument("output_dir",  help="Directory to save per-zscan optical plane tiffs")
    parser.add_argument("--kind", choices=["nuclei", "colors"], default="nuclei")
    parser.add_argument("--folder_glob", default="*-272.fli*_transformed*")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--crop_top",    type=int, default=0)
    parser.add_argument("--crop_bottom", type=int, default=0)
    parser.add_argument("--crop_left",   type=int, default=0)
    parser.add_argument("--crop_right",  type=int, default=0)
    parser.add_argument("--z_shift_y",   type=int, default=-24)
    parser.add_argument("--pad_value",   type=float, default=0.0)
    parser.add_argument("--z_only", type=int, default=None, help="Only process this z-scan index")
    parser.add_argument("--y_only", type=int, default=None, help="Only include this y index")
    parser.add_argument("--z_plane", type=int, default=None)
    parser.add_argument("--y_plane", type=int, default=None)
    parser.add_argument("--use_fov_crop", action="store_true",
                        help="Load FOV crop from matlab transforms file")
    parser.add_argument("--transforms_mat",
                        default="/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Matlab_info/NPBB328_colorMerge_transforms.mat")
    parser.add_argument("--crop_top_added",    type=int, default=0)
    parser.add_argument("--crop_bottom_added", type=int, default=0)
    parser.add_argument("--crop_left_added",   type=int, default=0)
    parser.add_argument("--crop_right_added",  type=int, default=0)

    args = parser.parse_args()

    crop_top    = args.crop_top
    crop_bottom = args.crop_bottom
    crop_left   = args.crop_left
    crop_right  = args.crop_right

    if args.use_fov_crop:
        import scipy.io as sio
        transforms = sio.loadmat(args.transforms_mat)
        FOVcrop_raw = transforms['FOVcrop']
        Z_raw, Y_raw = FOVcrop_raw[0][0][0][0], FOVcrop_raw[0][0][1][0]
        Z = tuple(int(v) for v in np.atleast_1d(Z_raw).ravel())
        Y = tuple(int(v) for v in np.atleast_1d(Y_raw).ravel())
        z1, z2 = Z
        y1, y2 = Y
        crop_top    = z1 + args.crop_top_added
        crop_bottom = z2 + args.crop_bottom_added
        crop_left   = y1 - args.crop_left_added
        crop_right  = y2 - args.crop_right_added

    save_optical_planes_per_zscan(
        root_dir=args.source_root,
        output_dir=args.output_dir,
        kind=args.kind,
        folder_glob=args.folder_glob,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        crop_left=crop_left,
        crop_right=crop_right,
        channel=args.channel,
        z_shift_y=args.z_shift_y,
        pad_value=args.pad_value,
        z_only=args.z_only,
        y_only=args.y_only,
        z_plane=args.z_plane,
        y_plane=args.y_plane,
    )

    #python tiffVols_to_composites_z_stacks.py '/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab01/out_tiffs/transformed/' '/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab01/out_composites/composite_ch0_transformed_y01-y10_z-stack_v2/' --kind nuclei --folder_glob "*-088.fli*_transformed*" --channel 0