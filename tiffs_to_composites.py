import argparse
from pathlib import Path
import tifffile
import numpy as np

def load_and_crop(path, crop_top, crop_bottom):
    img = tifffile.imread(str(path)).astype('float32')
    # img = img / ff[args.z_index, :, None]
    # img = img / lc[args.z_index, :, None]
    return img[crop_top:-crop_bottom, :] if crop_bottom > 0 else img[crop_top:, :]

def stitch_z_plane(z_idx, source_dir, output_path, crop_top=0, crop_bottom=0):
    """
    Stitches all y-tiles at given z into a composite TIFF, with cropping.
    """
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

    stitch_z_plane(args.z_index, args.source_dir, args.output_tif, args.crop_top, args.crop_bottom)