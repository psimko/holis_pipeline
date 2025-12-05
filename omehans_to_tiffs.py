import argparse
from pathlib import Path
import tifffile
import numpy as np
import re
import zarr
import os
from tifffile import imwrite
import dask
import dask.array as da
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

def save_omehans_as_tiffs(channel, processing, omehans_location, tiffs_location):

    if not os.path.exists(tiffs_location):
        os.makedirs(tiffs_location)

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

    paths = sorted(Path(omehans_location).glob(wanted_name), key=_y_key)

    paths_ome = [p / "omehans" for p in paths]

    print("files found:", len(paths_ome))

    """     for path_ome in paths_ome:
        image_dask = read_omehans(path_ome)  # (37000, 1024, 1280)
        print("Shape of the strip", image_dask.shape)
        print("Reading into memory")
        image = image_dask.compute()
        y_name = path_ome.parent.name
        print("done")
        for z in range(image.shape[1]):
            if not os.path.exists(f"{tiffs_location}/{y_name}/stripe_{z:04d}.tif"):
                if z % 100 == 0:
                    print("saving", z)
                plane = image[:, z, :]
                imwrite(f"{tiffs_location}/stripe_{z:04d}.tif", plane) """

    for path_ome in paths_ome:
        image_dask = read_omehans(path_ome)  # (37000, 1024, 1280)
        print("Shape of the strip", image_dask.shape)
        X, Z, Y = image_dask.shape
        y_name = path_ome.parent.name
        out_dir = os.path.join(tiffs_location, y_name)
        os.makedirs(out_dir, exist_ok=True)
        for z in range(Z):
            if not os.path.exists(f"{out_dir}/stripe_{z:04d}.tif"):
                target_z = 800
                if z == target_z: #if z % 100 == 0:
                    print(f'Saving plane={z}')
                    print("Reading into memory")
                    plane = image_dask[:, z, :].astype("float32").compute() 
                    print("done")
                    imwrite(f"{out_dir}/stripe_{z:04d}.tif", plane)
                else:
                    continue



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("channel", type=int, default=0)
    parser.add_argument("processing", nargs="?", default=None)
    parser.add_argument("source_dir")
    parser.add_argument("output_path")
    args = parser.parse_args()

    save_omehans_as_tiffs(args.channel, args.processing, args.source_dir, args.output_path)  


        # example use: python omehans_to_tiffs.py 0 'bg_subtracted' '/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/out_processed/' '/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/out_tiffs_peter/'