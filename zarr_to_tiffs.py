import os
import sys
import dask.array as da
from tifffile import imwrite
from dask.diagnostics import ProgressBar
from dask import config as dask_config
from concurrent.futures import ThreadPoolExecutor

from holis_pipeline.utils.zarr_related import read_omehans


def extract_tiff_planes_from_zarr(zarr_path, output_folder, strip_number, num_workers=8):
    """
    Extracts (1280, 36000) TIFF planes from a (36000, 1024, 1280) zarr array.
    One TIFF file per Y slice (axis 1).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Loading: {zarr_path}")
    data = read_omehans(zarr_path)
    # data = da.from_zarr(zarr_path)  # shape: (36000, 1024, 1280)


    z, y, x = data.shape
    print(f"Input shape: {data.shape}")

    data = data.compute()
    for i in range(y):
        if i % 100 == 0:
            print("Saving", i)
        slice_y = data[:, i, :].T
        output_path = os.path.join(output_folder, f"ch0_z{i:04d}_y{strip_number:04d}.tif")
        imwrite(output_path, slice_y)

    # # Set Dask config for threading
    # dask_config.set(scheduler='threads', num_workers=num_workers)
    #
    # def save_slice(i):
    #     slice_y = data[:, i, :].T  # Transpose to (1280, 36000)
    #     output_path = os.path.join(output_folder, f"ch0_z{i:04d}_y{strip_number:04d}.tif")
    #     # Use compute inside the thread to avoid memory buildup
    #     imwrite(output_path, slice_y.compute())
    #
    # with ProgressBar():
    #     with ThreadPoolExecutor(max_workers=num_workers) as executor:
    #         list(executor.map(save_slice, range(y)))


zarr_path = sys.argv[1]
output_folder = sys.argv[2]
strip_number = int(sys.argv[3])

extract_tiff_planes_from_zarr(zarr_path, output_folder, strip_number)
