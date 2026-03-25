import re
from pathlib import Path
import numpy as np
import dask.array as da
from dask import delayed
import tifffile as tiff
from typing import Tuple
import scipy.io as sio

Z_RE = re.compile(r"z(\d+)", re.IGNORECASE)
Y_RE = re.compile(r"y(\d+)", re.IGNORECASE)

""" def write_volume_tiff(
    vol_da,
    output_tif: str,
    normalize_to_uint16: bool = True,
):
    print("Computing full volume into memory...")
    vol = vol_da.compute()  # <-- loads full volume in RAM

    if normalize_to_uint16:
        mn, mx = float(vol.min()), float(vol.max())
        if mx <= mn:
            out = np.zeros_like(vol, dtype=np.uint16)
        else:
            out = ((vol - mn) / (mx - mn) * 65535.0).astype(np.uint16)
    else:
        # If already uint16-ish and you want to keep it
        out = vol.astype(np.uint16) if vol.dtype != np.uint16 else vol

    print(f"Writing TIFF: {output_tif} | shape={out.shape} dtype={out.dtype}")
    tiff.imwrite(output_tif, out, bigtiff=True)  # BigTIFF avoids 4GB limit
    print("Done.") """

""" def write_volume_tiff(vol_da, output_tif, normalize_to_uint16=False):
    vol = vol_da.compute().astype(np.float32)

    if normalize_to_uint16:
        p1, p99 = np.percentile(vol, (1, 99))
        vol = np.clip((vol - p1) / (p99 - p1), 0, 1)

    print(f"[Stitched Volume] | min={vol.min():.4f}, max={vol.max():.4f}, dtype={vol.dtype}, shape={vol.shape}", flush=True)

    #tiff.imwrite(output_tif, vol, bigtiff=True)
    tiff.imwrite(output_tif, np.clip(vol, 0, 65535).astype(np.uint16), bigtiff=True) """

def write_volume_tiff(vol_da, output_tif):
    vol = vol_da.compute().astype(np.float32)

    # stats BEFORE cast
    p = np.percentile(vol, [0, 0.1, 1, 50, 99, 99.9, 100])
    print("FLOAT stats:",
          f"min={vol.min():.3f}",
          f"max={vol.max():.3f}",
          "pct=", p, sep=" ", flush=True)

    out = np.clip(vol, 0, 65535).round().astype(np.uint16)

    # stats AFTER cast
    p2 = np.percentile(out, [0, 1, 50, 99, 100])
    print("U16 stats:",
          f"min={out.min()}",
          f"max={out.max()}",
          "pct=", p2, sep=" ", flush=True)

    # overwrite safety (so you’re not staring at an old file)
    from pathlib import Path
    Path(output_tif).unlink(missing_ok=True)

    tiff.imwrite(
        output_tif,
        out,
        bigtiff=True,
        photometric="minisblack"
    )
    print("Wrote:", output_tif, flush=True)


def _parse_zy(folder: Path):
    s = folder.name
    mz = Z_RE.search(s)
    my = Y_RE.search(s)
    if not (mz and my):
        return None
    return int(mz.group(1)), int(my.group(1))

def _load_vol_np(vol_path: str, crop_top: int, crop_bottom: int, crop_right: int, crop_left: int, channel: int = None, y_plane: int = None, z_plane: int = None) -> np.ndarray:
    v = tiff.imread(vol_path).astype("float32")  # expected (X, Z, Y)
    # Accept (X, Z, Y) or (C, X, Z, Y)
    if v.ndim == 3:
        pass
    elif v.ndim == 4:
        if channel is not None:
            v = v[int(channel)]  # -> (X, Z, Y)
        # else keep all channels -> (C, X, Z, Y)
    else:
        raise ValueError(f"{vol_path} expected 3D or 4D, got {v.shape}")

    z_end = -crop_bottom if crop_bottom > 0 else None
    y_end = -crop_right  if crop_right  > 0 else None
    v = v[:, crop_top:z_end, crop_left:y_end]

    if z_plane is not None:
        v = v[:, z_plane, :]                   # (X, Y)

    if y_plane is not None:
        v = v[:, :, y_plane]                   # (X, Z)

    print(
        f"[TILE] {Path(vol_path).name} | "
        f"min={v.min():.4f}, max={v.max():.4f}, dtype={v.dtype}, shape={v.shape}",
        flush=True
    )

    return v

def stitch_zy_grid(
    root_dir: str,
    output_zarr: str,
    output_tif: str,
    kind: str = "nuclei",  # "nuclei" -> vol.tiff, "colors" -> vol_colors.tiff
    folder_glob: str = "*-272.fli*_bg_subtracted*",
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
    y_concat_axis: int = 2,   # for (X,Z,Y): stitch Y on axis=2
    z_concat_axis: int = 1,   # for (X,Z,Y): stitch Z on axis=1
    chunks: Tuple[int, int, int] = (256, 64, 256),
    overwrite: bool = True,
    no_norm: bool = True,
    channel: int = 0,
    z_shift_y: int = -24,      # shift in Y per z-slab
    pad_value: float = 0.0,   # padding fill value
    z_only: int = None,  # only stitch this z index
    y_only: int = None,  # only stitch this y index
    y_plane = None,
    z_plane = None,
):

    if 'laser_corrected' in folder_glob:
        vol_name = "vol.tif" if kind.lower() in ("nuc", "nuclei") else f"vol_colors_ch{channel}.tif"
    elif 'bg_subtracted' in folder_glob:
        vol_name = "vol.tif" if kind.lower() in ("nuc", "nuclei") else "vol_colors.tif"
    elif 'transformed' in folder_glob:
        vol_name = "vol.tif" if kind.lower() in ("nuc", "nuclei") else f"vol_colors_ch{channel}.tif"
    else:
        vol_name = "vol.tif" if kind.lower() in ("nuc", "nuclei") else "vol_colors.tif"

    # If it's a channel volue rotate by 180 later
    if 'ch0' in vol_name:
        rotate=False
    elif 'ch' in vol_name:
        rotate=True
    else:
        rotate=False

    root = Path(root_dir)
    folders = [p for p in root.glob(f"**/{folder_glob}") if p.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No folders matching {folder_glob} under {root}")

    # group folders by z index
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
        raise RuntimeError("Found folders but couldn't parse zNN / yNN from names.")

    #if z_plane is not None and z_only is None:
    #    raise ValueError("--z_plane requires --z_only (otherwise you'd be mixing multiple z-slabs in 2D mode).")

    z_keys = sorted(by_z.keys())
    print(f"Z-tiles found: {len(z_keys)} (min={z_keys[0]}, max={z_keys[-1]})")
    if z_only is not None:
        print(f"  [FILTERED] z_only={z_only}")
    if y_only is not None:
        print(f"  [FILTERED] y_only={y_only}")
    print(f"Using volume file: {vol_name}")

    # infer per-tile shape from first available tile
    sample_folder = sorted(by_z[z_keys[0]])[0][1]
    sample_path = sample_folder / vol_name
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing {vol_name} in sample folder {sample_folder}")

    sample_vol = _load_vol_np(str(sample_path), crop_top, crop_bottom, crop_right, crop_left, channel, y_plane=y_plane, z_plane=z_plane)
    tile_shape = sample_vol.shape  # (X, Ztile, Ytile_crop)
    print(f"Per-tile volume shape after crop: {tile_shape}")

    # build: for each z_idx -> da.Array (X, Ztile, Ytotal)
    """     stitched_per_z = []
    for z_idx in z_keys:
        y_list = sorted(by_z[z_idx], key=lambda t: t[0])  # sort by y index
        tiles_da = []

        for y_idx, folder in y_list:
            vp = folder / vol_name
            if not vp.exists():
                raise FileNotFoundError(f"Missing {vol_name} in {folder}")

            d = delayed(_load_vol_np)(str(vp), crop_top, crop_bottom, channel)
            a = da.from_delayed(d, shape=tile_shape, dtype=np.float32)
            if rotate:
                a = da.rot90(a, k=2, axes=(1, 2))
            tiles_da.append(a)

        # stitch all y-tiles for this z
        slab = da.concatenate(tiles_da, axis=y_concat_axis)  # (X, Ztile, Ysum)
        stitched_per_z.append(slab)
        print(f"z{z_idx:02d}: y-tiles={len(tiles_da)} -> slab shape={slab.shape}")

    # stitch slabs across z tiles
    full = da.concatenate(stitched_per_z, axis=z_concat_axis)  # (X, Zsum, Ysum)
    full = full.rechunk(chunks)
    print(f"Final stitched dask shape: {full.shape}, chunks={full.chunksize}")

    out = Path(output_zarr)
    if overwrite and out.exists():
        import shutil
        shutil.rmtree(out)

    #full.to_zarr(str(out), overwrite=overwrite)
    #write_volume_tiff(full, "/path/to/stitched.tif", normalize_to_uint16=True)
    #print(f"Saved: {out}")
    #return str(out) """

    # build: for each z_idx -> da.Array (X, Ztile, Ytotal)
    stitched_per_z = []
    slabs_raw = []

    for z_idx in z_keys:
        y_list = sorted(by_z[z_idx], key=lambda t: t[0])  # sort by y index
        tiles_da = []

        for y_idx, folder in y_list:
            vp = folder / vol_name
            if not vp.exists():
                raise FileNotFoundError(f"Missing {vol_name} in {folder}")

            d = delayed(_load_vol_np)(str(vp), crop_top, crop_bottom, crop_right, crop_left, channel, y_plane=y_plane, z_plane=z_plane)
            a = da.from_delayed(d, shape=tile_shape, dtype=np.float32)
            if rotate:
                a = da.rot90(a, k=2, axes=(1, 2))
            tiles_da.append(a)
        slab = da.concatenate(tiles_da, axis=y_concat_axis)  # (X, Ztile, Ysum)
        slabs_raw.append(slab)
        print(f"z{z_idx:02d}: y-tiles={len(tiles_da)} -> slab shape={slab.shape}")

    # --- pad slabs in Y so each successive slab is shifted right ---
    n_slabs = len(slabs_raw)
    if n_slabs == 0:
        raise RuntimeError("No slabs constructed. Check folder_glob / parsing.")

    shift = int(z_shift_y)
    y_len = int(slabs_raw[0].shape[2])
    final_y = y_len + abs(shift) * (n_slabs - 1)

    shifted_slabs = []
    for i, slab in enumerate(slabs_raw):
        off = i * abs(shift)

        if shift >= 0:
            # pad on the LEFT so content moves to higher Y (right)
            left_pad, right_pad = off, final_y - (off + y_len)
        else:
            # pad on the RIGHT so content stays left, and earlier slabs appear right
            # equivalently: content moves to lower Y (left) as i increases
            left_pad, right_pad = final_y - (off + y_len), off

        slab_shifted = da.pad(
            slab,
            pad_width=((0, 0), (0, 0), (left_pad, right_pad)),
            mode="constant",
            constant_values=pad_value,
        )
        shifted_slabs.append(slab_shifted)
    # stitch slabs across z tiles (now all have same Y)
    full = da.concatenate(shifted_slabs, axis=z_concat_axis)  # (X, Zsum, final_y)
    full = full.rechunk(chunks)
    print(f"Final stitched dask shape: {full.shape}, chunks={full.chunksize}")

    return full

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", help="Root directory containing z/y tile folders")
    parser.add_argument("--kind", choices=["nuclei", "colors"], default="nuclei", help="Which per-tile file to stitch: vol.tiff (nuclei) or vol_colors.tiff (colors)")
    parser.add_argument("--folder_glob", default="*-272.fli*_bg_subtracted*", help="Glob to find tile folders under source_root (should include zNN and yNN in folder name)")
    parser.add_argument("--output_zarr", default=None, help="If set, write stitched volume to this zarr path")
    parser.add_argument("--output_tif", default=None, help="If set, write stitched volume to this BigTIFF path")
    parser.add_argument("--crop_bottom_added", type=int, default=0)
    parser.add_argument("--crop_top_added", type=int, default=0)
    parser.add_argument("--crop_right_added", type=int, default=0)
    parser.add_argument("--crop_left_added", type=int, default=0)
    parser.add_argument("--no_norm", action="store_true", help="Disable min-max normalization to uint16 for TIFF output")
    parser.add_argument("--channel", type=int)
    parser.add_argument("--z_only", type=int, default=None, help="Only stitch tiles from this z index (e.g., --z_only 5)")
    parser.add_argument("--y_only", type=int, default=None, help="Only stitch tiles from this y index (e.g., --y_only 3)")
    parser.add_argument("--z_plane", type=int, default=None)
    parser.add_argument("--y_plane", type=int, default=None)

    args = parser.parse_args()

    if args.output_zarr is None and args.output_tif is None:
        raise SystemExit("Give at least one of --output_zarr or --output_tif")

    transforms = sio.loadmat("/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Matlab_info/NPBB328_colorMerge_transforms.mat")
    FOVcrop_raw = transforms['FOVcrop']
    Z_raw, Y_raw = FOVcrop_raw[0][0][0][0], FOVcrop_raw[0][0][1][0]
    Z = tuple(int(v) for v in np.atleast_1d(Z_raw).ravel())
    Y = tuple(int(v) for v in np.atleast_1d(Y_raw).ravel())
    FOVcrop = {"Z": Z, "Y": Y}

    z1, z2 = FOVcrop["Z"]
    y1, y2 = FOVcrop["Y"]

    vol_da = stitch_zy_grid(
        root_dir=args.source_root,
        output_zarr=args.output_zarr if args.output_zarr is not None else "/tmp/_unused.zarr",
        output_tif=args.output_tif if args.output_tif is not None else "/tmp/_unused.tif",
        kind=args.kind,
        folder_glob=args.folder_glob,
        crop_top=z1 + args.crop_top_added,
        crop_bottom=z2 + args.crop_bottom_added,
        crop_left=y1 - args.crop_left_added,
        crop_right=y2 - args.crop_right_added,
        overwrite=True,
        no_norm=True,
        channel=args.channel,
        z_only=args.z_only,      
        y_only=args.y_only,   
        z_plane=args.z_plane,
        y_plane=args.y_plane
    )

    if args.output_zarr: vol_da.to_zarr(args.output_zarr, overwrite=True)
    if args.output_tif:  write_volume_tiff(vol_da, args.output_tif) #normalize_to_uint16=not args.no_norm)


#python tiffVols_to_composites.py '/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_tiffs/bg_subtracted/' --output_zarr '/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_composites/composite_nuc_zarr_bg_subtracted.zarr' --output_tif '/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_composites/composite_nuc_tiff_bg_subtracted.tif' --kind 'nuclei' --folder_glob "*-272.fli*_bg_subtracted*" --crop_top 0 --crop_bottom 0 --channel 1

#python tiffVols_to_composites.py '/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_tiffs/transformed_noCrop/' --output_tif '/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_composites/composite_colors_tif_transformed.tif' --kind 'colors' --folder_glob "*-088.fli*_transformed*" --crop_top 0 --crop_bottom 0 --no_norm --channel 0