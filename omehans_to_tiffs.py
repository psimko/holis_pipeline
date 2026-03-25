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

def save_omehans_as_tiffs(channel, processing, asVolume, omehans_location, tiffs_location):

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
            "transformed":     nuc_file_pattern +  "_transformed", 
            "registered":      color_file_pattern + "_registered",
            "unmixed":         color_file_pattern + "_unmixed",
        }
    else:
        suffix_map = {
            None:              color_file_pattern,
            "bg_subtracted":   color_file_pattern + "_bg_subtracted",
            "laser_corrected": color_file_pattern + "_laser_corrected",
            "transformed":     color_file_pattern +  "_transformed", 
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

    target_z = 400

    for path_ome in paths_ome:
    #for path_ome in paths: # for raw files
        image_dask = read_omehans(path_ome)  # (37000, 1024, 1280) or (37000, 512, 640)
        print("Shape of the strip", image_dask.shape)
        ndim = image_dask.ndim

        y_name = path_ome.parent.name
        #y_name = path_ome.name # for raw files
        out_dir = os.path.join(tiffs_location, y_name)
        os.makedirs(out_dir, exist_ok=True)

        if ndim ==3:
            # shape: (X, Z, Y)
            X, Z, Y = image_dask.shape
            if asVolume == 'False':
                for z in range(Z):
                    if z != target_z:
                        continue
                    if channel==0:
                        out_path = os.path.join(out_dir, f"stripe_{z:04d}.tif")
                    else:
                        out_path = os.path.join(out_dir, f"stripe_z{z:04d}_colors.tif")
                    if os.path.exists(out_path):
                        continue

                    print(f'Saving plane={z}')
                    print("Reading into memory")
                    plane = image_dask[:, z, :].astype("float32").compute() 
                    print("Done")
                    imwrite(out_path, plane)
            elif asVolume == 'True':
                    if channel==0:
                        out_path = os.path.join(out_dir, f"vol.tif")
                    else:
                        out_path = os.path.join(out_dir, f"vol_colors.tif")
                    if os.path.exists(out_path):
                        continue

                    print(f'Saving volume {y_name}')
                    print("Reading into memory")
                    vol = image_dask[:, :, :].astype("float32").compute() 
                    print("Done")
                    imwrite(out_path, vol)
            else:
                print('asVolume must be set to True or False')


        elif ndim == 4:
            # shape: (C, X, Z, Y)
            C, X, Z, Y = image_dask.shape
            if asVolume == 'False':
                for z in range(Z):
                    if z != target_z:
                        continue

                    for c in range(C):
                        out_path = os.path.join(out_dir, f"stripe_c{c:02d}_z{z:04d}.tif")
                        if os.path.exists(out_path):
                            continue

                        print(f"Saving plane z={z}, channel={c}")
                        print("Reading into memory")
                        plane = image_dask[c, :, z, :].astype("float32").compute()
                        print("done")
                        imwrite(out_path, plane)
            elif asVolume == 'True':
                for c in range(C):
                    if channel==0:
                        out_path = os.path.join(out_dir, f"vol.tif")
                    else:
                        out_path = os.path.join(out_dir, f"vol_colors_ch{c}.tif")
                    if os.path.exists(out_path):
                        continue

                    print(f'Saving volume channel{c} to {y_name}')
                    print("Reading into memory")
                    vol = image_dask[c, :, :, :].astype("float32").compute() 
                    print("Done")
                    imwrite(out_path, vol)
            else:
                print('asVolume must be set to True or False')

        else:
            raise ValueError(f"Unsupported image shape {image_dask.shape}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("channel", type=int, default=0)
    #parser.add_argument("processing", nargs="?", default=None)
    parser.add_argument("processing", nargs="?", default=None, type=lambda s: None if s.lower() in ("none", "null", "") else s)
    parser.add_argument("asVolume", nargs="?", default=None)
    parser.add_argument("source_dir")
    parser.add_argument("output_path")
    args = parser.parse_args()

    save_omehans_as_tiffs(args.channel, args.processing, args.asVolume, args.source_dir, args.output_path)  


    # example use: python omehans_to_tiffs.py 0 'bg_subtracted' 'True' '/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/out_processed/' '/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/out_tiffs_peter/'

    #python omehans_to_tiffs.py 1 'transformed' 'True' '/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_processed/' '/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_tiffs/transformed/'

    #python omehans_to_tiffs.py 1 'transformed' 'True' '/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab01/out_processed_y1toy10/' '/bil/proj/rf1hillman/results//bil/proj/rf1hillman/results/NPBB328_Cortex/Slab01/out_tiffs/transformed/'