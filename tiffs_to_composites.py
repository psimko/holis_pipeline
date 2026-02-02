import argparse
from pathlib import Path
import tifffile
import numpy as np

""" def load_and_crop(path, crop_top, crop_bottom):
    img = tifffile.imread(str(path)).astype('float32')
    # img = img / ff[args.z_index, :, None]
    # img = img / lc[args.z_index, :, None]
    return img[crop_top:-crop_bottom, :] if crop_bottom > 0 else img[crop_top:, :]

def stitch_z_plane(z_idx, source_dir, output_path, crop_top=0, crop_bottom=0):
    """ """Stitches all y-tiles at given z into a composite TIFF, with cropping.""" """
    paths = sorted(Path(source_dir).glob(f"ch0_z{z_idx:04d}_y*.tif"))
    print("files found:", len(paths))
    if len(paths) == 0:
        raise FileNotFoundError(f"No tiles found for z={z_idx} in {source_dir}")

    tiles = [load_and_crop(p, crop_top, crop_bottom) for p in paths]
    composite = np.concatenate(tiles, axis=0)  # Stack vertically (36000 x cropped)

    composite = (composite - composite.min()) / (composite.max() - composite.min())
    tifffile.imwrite(output_path, (composite*65535).astype('uint16'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("z_index", type=int)
    parser.add_argument("source_dir")
    parser.add_argument("output_tif")
    parser.add_argument("--crop_top", type=int, default=0)
    parser.add_argument("--crop_bottom", type=int, default=0)
    args = parser.parse_args()

    # ff = np.load('/h20/Public/holis/2025_06_15_NPBB328_surface_corrections/FF_nuc_norm.npy')
    # lc = np.load('/h20/Public/holis/2025_06_15_NPBB328_surface_corrections/Laser_correction_Nuclei_pattern.npy')

    stitch_z_plane(args.z_index, args.source_dir, args.output_tif, args.crop_top, args.crop_bottom) """

import argparse, re, os
from pathlib import Path
import numpy as np
import tifffile as tiff

Y_RE = re.compile(r'y(\d+)', re.IGNORECASE)

def _y_key(p: Path) -> int:
    m = Y_RE.search(p.as_posix())
    return int(m.group(1)) if m else 10**9

def load_and_crop(path: Path, crop_top: int, crop_bottom: int) -> np.ndarray:
    img = tiff.imread(str(path)).astype("float32")
    return img[:,crop_top:-crop_bottom] if crop_bottom > 0 else img[:,crop_top:]       #img[crop_top:-crop_bottom,:]

def find_slice_in_dir(y_dir: Path, z_idx: int, slice_template) -> Path:
    """
    Return the path to the z-th slice inside a y-folder.
    """
    if slice_template:
        pat = slice_template.format(z=z_idx)
        candidates = list(y_dir.glob(pat))
        if candidates:
            return candidates[0]

    raise FileNotFoundError(f"No slice for z={z_idx} in {y_dir}")

def stitch_z_from_yfolders(
    z_idx: int,
    root_dir: str,
    output_tif: str,
    y_dir_glob: str = "*y*-272.fli*_bg_subtracted*",
    slice_template: str = "slice{z:03d}.tif",
    crop_top: int = 0,
    crop_bottom: int = 0,
    normalize_to_uint16: bool = True,
):
    """
    For a given z, collect the z-th slice from each y-folder under root_dir,
    sort folders by yNNN, stack vertically, and write output_tif.
    """
    
    output_tif = Path(output_tif)
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    root = Path(root_dir)
    y_dirs = sorted((p for p in root.glob(f"**/{y_dir_glob}") if p.is_dir()), key=_y_key)
    print(f"y-folders found: {len(y_dirs)}")

    if not y_dirs:
        print(f"No y-folders matching '{y_dir_glob}' under {root}, setting it to root_dir")
        y_dirs = [root]

    tiles = []
    missing = []
    for yd in y_dirs:
        try:
            sl = find_slice_in_dir(yd, z_idx, slice_template)
        except FileNotFoundError:
            missing.append(yd)
            continue
        tiles.append(load_and_crop(sl, crop_top, crop_bottom))

    if missing:
        raise FileNotFoundError(
            f"Missing z={z_idx} slice in {len(missing)} folder(s). "
            f"Example missing: {missing[0]}"
        )

    # Stack vertically (y increases downwards)
    composite = np.concatenate(tiles, axis=1)

    if normalize_to_uint16:
        mn, mx = composite.min(), composite.max()
        if mx <= mn:
            out = np.zeros_like(composite, dtype=np.uint16)
        else:
            out = ((composite - mn) / (mx - mn) * 65535.0).astype(np.uint16)
    else:
        out = composite.astype(np.float32)

    tifffile.imwrite(str(output_tif), out)
    print(f"Saved composite: {output_tif}  | shape={out.shape}  dtype={out.dtype}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("z_index", type=int)
    #parser.add_argument("processing", nargs="?", default=None)
    parser.add_argument("source_root")
    parser.add_argument("output_tif")
    parser.add_argument("--y_dir_glob", default=f"*y*-272.fli*_bg_subtracted*")
    parser.add_argument("--slice_template", default="stripe_{z:04d}*.tif")
    parser.add_argument("--crop_top", type=int, default=0)
    parser.add_argument("--crop_bottom", type=int, default=0)
    parser.add_argument("--no_norm", action="store_true", help="disable 0-1 -> uint16 normalization")
    args = parser.parse_args()

    stitch_z_from_yfolders(
        z_idx=args.z_index,
        root_dir=args.source_root,
        output_tif=args.output_tif,
        y_dir_glob=args.y_dir_glob,
        slice_template=None if args.slice_template.lower() in ("none", "null") else args.slice_template,
        crop_top=args.crop_top,
        crop_bottom=args.crop_bottom,
        normalize_to_uint16=not args.no_norm,
    )
#python tiffs_to_composites.py 800 --y_dir_glob="*y*-272.fli*_bg_subtracted*" '/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/out_tiffs_peter/' '/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/out_composites_peter/composite__nuc_tiff_z800_bg_subtracted.tiff' --crop_top 387 --crop_bottom 5