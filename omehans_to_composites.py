import argparse
from pathlib import Path
import tifffile
import numpy as np
import re
import zarr
import os
from stack_to_multiscale_ngff.archived_nested_store import Archived_Nested_Store
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store
#from holis_pipeline.preprocess_v2_sep2025 import _clip16
from holis_pipeline.preprocessing_functions import read_data_file
from holis_pipeline.read_data import read_fli_as_zarr
from holis_pipeline.utils.zarr_related import read_omehans, write_omehans, write_zarr

def _y_key(p: Path) -> int:
    """Numeric sort by yNNN anywhere in the path/name; non-matches last."""
    m = re.search(r'y(\d+)', p.as_posix(), flags=re.IGNORECASE)
    return int(m.group(1)) if m else 10**9

def _norm_z(z_idx):
    if z_idx in (None, "all"):
        raise ValueError("For 2D stitching, pass a single z index, not 'all'.")
    return int(z_idx)

def load_and_crop(processing, path, channel=0, z_idx='all', crop_top=0, crop_bottom=0, three_d_layout="XYZ"):
    """
    Load a 3D/4D volume and crop along Y.
    - 4D assumed (C,Z,Y,X) → pick `channel` then crop Y (axis 2 of the sliced volume)
    - 3D assumed (X,Y,Z) if three_d_layout='XYZ' (crop middle axis),
                 or (Z,Y,X) if three_d_layout='ZYX' (crop first axis)
    """
    arr = read_omehans(path)  
    arr.compute()
    print(arr.shape)

    # build end index once to avoid -0 (which would mean 0)
    y_end = None if crop_bottom <= 0 else -crop_bottom
    y_start = crop_top

    z = slice(None) if z_idx is None else int(z_idx)

    if processing in ['registered', 'unmixed']:
        # This means the file is either _registered or _unmixed which means shape is (C,Z,Y,X)
        img = arr[channel]              # -> (Z, Y, X)   
        return img[z, y_start:y_end, :]

    elif processing in [None, 'bg_subtracted', 'laser_corrected']:
        if three_d_layout.upper() == "XYZ":
            # This means th file has either no suffix or _bg_subtracted or _laser_corrected which means shape is (X, Y, Z)
            #return arr[:, y_start:y_end, z]
            return arr[:, z, y_start:y_end]
        elif three_d_layout.upper() == "ZYX":
            # (Z, Y, X) - this should not happen with the current preprocessing (Oct 2025)
            return arr[z, y_start:y_end, :]
        else:
            raise ValueError("three_d_layout must be 'XYZ' or 'ZYX'")


def stitch_z_plane(channel, z_idx, processing, source_dir, output_path, crop_top, crop_bottom):
    """
    Stitches all y-tiles at given z into a composite zarr, with cropping.
    Processing: None, bg_subtracted, laser_corrected, registered, unmixed
    """

    # Map the processing stage to the folder suffix you need to open

    nuc_file_pattern = "*-272.fli*"
    color_file_pattern = "*-088.fli*"

    if channel==0:
        suffix_map = {
            None:              nuc_file_pattern,
            "bg_subtracted":   nuc_file_pattern + "_bg_subtracted",
            "laser_corrected": nuc_file_pattern + "_laser_corrected",
            "registered":      color_file_pattern + "_registered",
            "unmixed":         color_file_pattern + "_unmixed",
        }
    else:
        suffix_map = {
            None:              color_file_pattern,
            "bg_subtracted":   color_file_pattern + "_bg_subtracted",
            "laser_corrected": color_file_pattern + "_laser_corrected",
            "registered":      color_file_pattern + "_registered",
            "unmixed":         color_file_pattern + "_unmixed",
        }
    wanted_name = suffix_map.get(processing)
    print(f'Wanted name: {wanted_name}')
    if wanted_name is None:
        raise ValueError(f"Unknown processing='{processing}'. Expected one of {list(suffix_map)}")

    paths = sorted(Path(source_dir).rglob(wanted_name), key=_y_key)

    paths_ome = [p / "omehans" for p in paths]

    print("files found:", len(paths_ome))

    num_scans = len(paths_ome)
    for path in paths_ome[:num_scans]:
        print(path)
    tiles = [load_and_crop(processing=processing, path=p, channel=channel, z_idx=z_idx, crop_top=crop_top, crop_bottom=crop_bottom) for p in paths_ome[0:num_scans]]
    composite = np.concatenate(tiles, axis=1)  # Stack accros y

    # Normalize
    composite = (composite - composite.min()) / (composite.max() - composite.min())

    def asnumpy(a):
        return a.compute() if hasattr(a, "compute") else a

    # Write output 
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    #tifffile.imwrite(os.path.join(output_path, f'composite{wanted_name}_wCrop_top{crop_top}_bottom{crop_bottom}_{num_scans}.tiff'), (composite*65535).astype('uint16').compute(), bigtiff=True)

    arr16 = (composite * 65535).astype('uint16')
    arr16 = asnumpy(arr16)   # safe for both NumPy and Dask
    tifffile.imwrite(
        os.path.join(output_path, f'Slab6_composite_ch{channel}_{wanted_name}_wCrop_top{crop_top}_bottom{crop_bottom}_{num_scans}.tiff'),
        arr16,
        bigtiff=True
)
    print('Wrote tiff composite to disk.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("channel", type=int, default=0)
    parser.add_argument("z_index", nargs="?")
    parser.add_argument("processing", nargs="?", default=None)
    parser.add_argument("source_dir")
    parser.add_argument("output_path")
    parser.add_argument("--crop_top", type=int, default=0)
    parser.add_argument("--crop_bottom", type=int, default=0)
    args = parser.parse_args()

    stitch_z_plane(args.channel, args.z_index, args.processing, args.source_dir, args.output_path, args.crop_top, args.crop_bottom)      # crop is [387,5]


    # example use: python omehans_to_composites.py 0 500 'bg_subtracted' '/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/out_processed_bg_dec1/' '/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/out_composites/dec2/' --crop_top 387 --crop_bottom 5