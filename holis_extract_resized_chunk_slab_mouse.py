import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import dask.array as da
import tifffile
import zarr
from skimage.transform import resize
from stack_to_multiscale_ngff.archived_nested_store import Archived_Nested_Store
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store

from utils.settings import *


def get_origin_coords(ndim, patchify_chunks_shape, chunk_size):
    """
    Get coordinates of each chunk origin.

    TODO: only 3D now, make compatible with 2D

    :param ndim:
    :param chunk_shape:
    :param patches_shape:
    :return:
    """
    coords_shape = list(patchify_chunks_shape[:ndim]) + [ndim]
    coords = np.empty(coords_shape, dtype=np.uint16)
    print(" coords shape", coords.shape)
    for z in range(coords.shape[0]):
        for y in range(coords.shape[1]):
            for x in range(coords.shape[2]):
                coords[z, y, x, :] = np.array((
                    z * chunk_size[0],
                    y * chunk_size[1],
                    x * chunk_size[2]
                ))
    coords = np.reshape(coords, (np.prod(coords.shape[:ndim]), ndim))
    print("final coords shape", coords.shape)
    return coords


def get_chunk_indices(origin_coords, chunk_size):
    indices = []
    for origin in list(origin_coords):
        indices.append([
            slice(origin[0], origin[0] + chunk_size[0], 1),
            slice(origin[1], origin[1] + chunk_size[1], 1),
            slice(origin[2], origin[2] + chunk_size[2], 1)
        ])
    return indices


def process_chunk(ind):
    chunk = np.array(lazy_data[ind[0], ind[1], ind[2]])
    chunk = (
        resize(
            chunk,
            (int(round(chunk.shape[0] * yz_ratio)), chunk.shape[1], int(round(chunk.shape[2] * yx_ratio)))
        ) * 65535
    ).astype("uint16")
    tifffile.imwrite(chunk_file, chunk)
    # tifffile.imwrite(os.path.join(output_folder, f"chunk_{str(number).zfill(5)}.tif"), chunk)


chunk_file = sys.argv[1]
DATA_DIR = sys.argv[2]
chunks_folder = str(Path(chunk_file).parent)
if not os.path.exists(chunks_folder):
    os.makedirs(chunks_folder)
# DEEPBLINK_CHUNK_SIZE = (40, 1700, 1700)
# RESOLUTION = [1.34, 1.54, 2.0]
yx_ratio = float(RESOLUTION[-1]) / RESOLUTION[-2]
yz_ratio = float(RESOLUTION[-3]) / RESOLUTION[-2]
number = int(re.findall(r"\d+", os.path.basename(chunk_file))[-1])
location = os.path.join(DATA_DIR, 'scale0')
store = H5_Nested_Store(location)
zarray = zarr.open(store)
dask_zarray = da.array(zarray)
lazy_tiff_stack = dask_zarray[0, 0, :, :, :]
ratios = (np.array(lazy_tiff_stack.shape) / np.array(DEEPBLINK_CHUNK_SIZE)).astype('int') + 1
patchify_chunks_shape = (*list(ratios), *DEEPBLINK_CHUNK_SIZE)
origin_coords = get_origin_coords(3, patchify_chunks_shape, DEEPBLINK_CHUNK_SIZE)
chunk_indices = get_chunk_indices(origin_coords, DEEPBLINK_CHUNK_SIZE)
lazy_data = dask_zarray[0, 0, :, :, :]
ind = chunk_indices[number]
process_chunk(ind)
