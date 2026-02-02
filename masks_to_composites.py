import re
from pathlib import Path
import numpy as np
import dask.array as da
from dask import delayed
import tifffile as tiff
from typing import Tuple

Z_RE = re.compile(r"z(\d+)", re.IGNORECASE)
Y_RE = re.compile(r"y(\d+)", re.IGNORECASE)

def write_volume_tiff(
    vol_da,
    output_tif: str,
    normalize_to_uint16: bool = True,
):
    """
    vol_da: dask array shaped (X, Z, Y)  (from stitch_zy_grid)
    Writes a 3D BigTIFF.
    """
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
    print("Done.")

def _parse_zy(folder: Path):
    s = folder.name
    mz = Z_RE.search(s)
    my = Y_RE.search(s)
    if not (mz and my):
        return None
    return int(mz.group(1)), int(my.group(1))

def _load_vol_np(vol_path: str, crop_top: int, crop_bottom: int, channel: int = None) -> np.ndarray:
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

    # crop along Y (last axis), matching your earlier usage
    if crop_bottom > 0:
        v = v[:, :, crop_top:-crop_bottom]
    else:
        v = v[:, :, crop_top:]
    return v

def _find_volume_file(tile_folder: Path, vol_name: str) -> Path:
    # 1) direct child
    p = tile_folder / vol_name
    if p.exists():
        return p

    # 2) anywhere under the folder (first hit)
    hits = list(tile_folder.rglob(vol_name))
    if hits:
        # pick the shallowest hit (usually what you want)
        hits.sort(key=lambda x: len(x.parts))
        return hits[0]

    raise FileNotFoundError(f"Missing {vol_name} under {tile_folder}")

def stitch_zy_grid(
    root_dir: str,
    output_zarr: str,
    output_tif: str,
    kind: str = "nuclei",  # "nuclei" -> vol.tiff, "colors" -> vol_colors.tiff
    folder_glob: str = "*-272.fli*_bg_subtracted*",
    crop_top: int = 0,
    crop_bottom: int = 0,
    y_concat_axis: int = 2,   # for (X,Z,Y): stitch Y on axis=2
    z_concat_axis: int = 1,   # for (X,Z,Y): stitch Z on axis=1
    chunks: Tuple[int, int, int] = (256, 64, 256),
    overwrite: bool = True,
    no_norm: bool = True,
    channel: int = 0
):

    if 'laser_corrected' in folder_glob:
        if kind.lower() in ("nuc", "nuclei"):
            vol_name = "vol.tif" 
        elif kind.lower() in ("masks", 'mask'):
            vol_name = 'combined_mask_raw_space.tif'
        else:  vol_name = f"vol_colors_ch{channel}.tif"
    elif 'bg_subtracted' in folder_glob:
        vol_name = "vol.tif" if kind.lower() in ("nuc", "nuclei") else "vol_colors.tif"
    elif 'transformed' in folder_glob:
        vol_name = "vol.tif" if kind.lower() in ("nuc", "nuclei") else f"vol_colors_ch{channel}.tif"

    # If it's a channel volume rotate by 180 later
    if 'ch0' in vol_name:
        rotate=False
    elif 'ch' in vol_name:
        rotate=True
    else:
        rotate=False

    root = Path(root_dir)
    folders = [p for p in root.rglob(folder_glob) if p.is_dir()]
    if not folders:
        raise FileNotFoundError(f"No folders matching {folder_glob} under {root}")

    # group folders by z index
    by_z = {}
    for f in folders:
        zy = _parse_zy(f)
        if zy is None:
            continue
        z_idx, y_idx = zy
        by_z.setdefault(z_idx, []).append((y_idx, f))

    if not by_z:
        raise RuntimeError("Found folders but couldn't parse zNN / yNN from names.")

    z_keys = sorted(by_z.keys())
    print(f"Z-tiles found: {len(z_keys)} (min={z_keys[0]}, max={z_keys[-1]})")
    print(f"Using volume file: {vol_name}")

    # infer per-tile shape from first available tile
    sample_folder = sorted(by_z[z_keys[0]])[0][1]
    sample_path = _find_volume_file(sample_folder, vol_name)
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing {vol_name} in sample folder {sample_folder}")

    sample_vol = _load_vol_np(str(sample_path), crop_top, crop_bottom, channel)
    tile_shape = sample_vol.shape  # (X, Ztile, Ytile_crop)
    print(f"Per-tile volume shape after crop: {tile_shape}")

    # build: for each z_idx -> da.Array (X, Ztile, Ytotal)
    stitched_per_z = []
    for z_idx in z_keys:
        y_list = sorted(by_z[z_idx], key=lambda t: t[0])  # sort by y index
        tiles_da = []

        for y_idx, folder in y_list:
            vp = _find_volume_file(folder, vol_name)

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
    #return str(out)

    return full

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", help="Root directory containing z/y tile folders")
    parser.add_argument("--kind", choices=["nuclei", "colors", 'masks'], default="nuclei", help="Which per-tile file to stitch: vol.tiff (nuclei) or vol_colors.tiff (colors)")
    parser.add_argument("--folder_glob", default="*-272.fli*_bg_subtracted*", help="Glob to find tile folders under source_root (should include zNN and yNN in folder name)")
    parser.add_argument("--output_zarr", default=None, help="If set, write stitched volume to this zarr path")
    parser.add_argument("--output_tif", default=None, help="If set, write stitched volume to this BigTIFF path")
    parser.add_argument("--crop_top", type=int, default=0)
    parser.add_argument("--crop_bottom", type=int, default=0)
    parser.add_argument("--no_norm", action="store_true", help="Disable min-max normalization to uint16 for TIFF output")
    parser.add_argument("--channel", type=int)

    args = parser.parse_args()

    if args.output_zarr is None and args.output_tif is None:
        raise SystemExit("Give at least one of --output_zarr or --output_tif")

    vol_da = stitch_zy_grid(
        root_dir=args.source_root,
        output_zarr=args.output_zarr if args.output_zarr is not None else "/tmp/_unused.zarr",
        output_tif=args.output_tif if args.output_tif is not None else "/tmp/_unused.tif",
        kind=args.kind,
        folder_glob=args.folder_glob,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        overwrite=True,
        no_norm=True,
        channel=args.channel
    )

    if args.output_zarr: vol_da.to_zarr(args.output_zarr, overwrite=True)
    if args.output_tif:  write_volume_tiff(vol_da, args.output_tif, normalize_to_uint16=not args.no_norm)


#python tiffVols_to_composites.py '/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_tiffs/bg_subtracted/' --output_zarr '/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_composites/composite_nuc_zarr_bg_subtracted.zarr' --output_tif '/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_composites/composite_nuc_tiff_bg_subtracted.tif' --kind 'nuclei' --folder_glob "*-272.fli*_bg_subtracted*" --crop_top 0 --crop_bottom 0 --channel 1

#python tiffVols_to_composites.py '/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_tiffs/transformed/' --output_tif '/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_composites/composite_colors_tif_transformed.tif' --kind 'colors' --folder_glob "*-088.fli*_transformed*" --crop_top 0 --crop_bottom 0 --no_norm --channel 0