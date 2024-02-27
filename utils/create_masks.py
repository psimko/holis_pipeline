"""
usage:
python holis_segment_foreground_and_cerebellum.py <path to input omezarr data> <path to pipeline output folder> <resolution level used to generate masks>
"""
import os
import re
import subprocess
import sys
from datetime import datetime
from glob import glob

import tifffile
import numpy as np
import pandas as pd
import dask
import dask.array as da
from skimage.transform import resize

from .settings import (
    CHUNK_SIZE, NUCLEI_DIR, OUTPUT_DIR, SCALE_USED_FOR_MASKS, NUCLEI_RESOLUTION,
    DENSE_REGIONS_MASK, FOREGROUND_MASK
)


def get_origin_coords_rounded(ndim, patchify_chunks_shape, chunk_size):
    coords_shape = list(patchify_chunks_shape[:ndim]) + [ndim]
    coords = np.empty(coords_shape, dtype=np.uint16)
    print(" coords shape", coords.shape)
    for z in range(coords.shape[0]):
        for y in range(coords.shape[1]):
            for x in range(coords.shape[2]):
                coords[z, y, x, :] = np.array((
                    int(round(z * chunk_size[0])),
                    int(round(y * chunk_size[1])),
                    int(round(x * chunk_size[2]))
                ))
    coords = np.reshape(coords, (np.prod(coords.shape[:ndim]), ndim))
    print("final coords shape", coords.shape)
    return coords


def get_chunk_indices_rounded(origin_coords, chunk_size):
    indices = []
    for origin in list(origin_coords):
        indices.append([
            slice(origin[0], int(round(origin[0] + chunk_size[0])), 1),
            slice(origin[1], int(round(origin[1] + chunk_size[1])), 1),
            slice(origin[2], int(round(origin[2] + chunk_size[2])), 1)
        ])
    return indices


def get_chunks_with_bright_signal():
    """
    Get chunks that have bright spots in them from low-resolution mask (0=bright, 1=normal)
    :return:
    """

    def dask_mask_to_tiffs_resize_check_zeros(lazy_tiff_stack, chunk_indices, output_folder, yx_ratio, yz_ratio, missing_chunks):

        def read_chunk(ind):
            print("Reading chunk", ind)
            return lazy_tiff_stack[tuple(ind)]

        def resize_chunk(chunk):
            print("Resizing chunk")
            if np.any(np.array(chunk.shape) == 0):
                return
            return (resize(chunk, (chunk.shape[0]*yz_ratio, chunk.shape[1], int(round(chunk.shape[2]*yx_ratio)))) * 65535).astype("uint16")

        def save_chunk(i, img):
            if img is None:
                return False
            tifffile.imwrite(os.path.join(output_folder, f"chunk_{str(i).zfill(5)}.tif"), img)
            has_zeros = not np.all(img)
            return has_zeros

        missing_chunk_indices = [chunk_indices[x] for x in missing_chunks]

        chunks = [dask.delayed(read_chunk)(i) for n, i in zip(missing_chunks, missing_chunk_indices)]
        resized = [dask.delayed(resize_chunk)(i) for i in chunks]
        saved = [dask.delayed(save_chunk)(i, img) for i, img in zip(missing_chunks, resized)]
        result = dask.compute(saved)
        # np.save(os.path.join(OUTPUT_DIR, f'scale_{scale}', 'chunks_bright1.npy'), np.array(result))
        bg_chunks = [x for x in range(len(result[0])) if result[0][x]]
        np.save(os.path.join(OUTPUT_DIR, 'bright_chunks.npy'), bg_chunks)
        with open(os.path.join(OUTPUT_DIR, 'bright_chunks.txt'), 'w') as f:
            f.write(str(bg_chunks))

    scale = SCALE_USED_FOR_MASKS
    scale_factor = 2 ** scale
    tiff_stack = tifffile.imread(DENSE_REGIONS_MASK)
    chunks_folder = os.path.join(OUTPUT_DIR, 'scale_x', 'bright_spots_mask_resized')
    if not os.path.exists(chunks_folder):
        os.makedirs(chunks_folder)
    lazy_tiff_stack = da.array(tiff_stack)
    chunk_shape = list(np.array(CHUNK_SIZE) / scale_factor)
    print(chunk_shape)
    rounded_chunk_shape = [int(round(chunk_shape[0])), int(round(chunk_shape[1])), int(round(chunk_shape[2]))]
    print(rounded_chunk_shape)
    ratios = (np.array(tiff_stack.shape) / np.array(chunk_shape)).astype('int') + 1
    patchify_chunks_shape = (*list(ratios), *chunk_shape)
    origin_coords = get_origin_coords_rounded(3, patchify_chunks_shape, chunk_shape)
    np.save(os.path.join(chunks_folder, 'origin_coords.npy'), origin_coords)
    chunk_indices = get_chunk_indices_rounded(origin_coords, chunk_shape)
    missing_chunks = [
        x for x in range(len(chunk_indices))
        if not os.path.exists(os.path.join(chunks_folder, f"chunk_{str(x).zfill(5)}.tif"))
    ]
    yx_ratio = float(NUCLEI_RESOLUTION[-1]) / NUCLEI_RESOLUTION[-2]
    yz_ratio = float(NUCLEI_RESOLUTION[-3]) / NUCLEI_RESOLUTION[-2]

    print("Creating downscaled dense region masks for chunks:\n", missing_chunks)
    dask_mask_to_tiffs_resize_check_zeros(lazy_tiff_stack, chunk_indices, chunks_folder, yx_ratio, yz_ratio, missing_chunks)


def get_chunks_with_background():
    """
    By having low resolution fg/bg mask, decide which high-res chunks belong to the bg
    """

    def dask_mask_to_tiffs_resize_check_zeros(lazy_tiff_stack, chunk_indices, output_folder, yx_ratio, yz_ratio, missing_chunks):

        def read_chunk(ind):
            print("Reading chunk", ind)
            return lazy_tiff_stack[tuple(ind)]

        def resize_chunk(chunk):
            print("Resizing chunk")
            print("Chunk shape", chunk.shape)
            if np.any(np.array(chunk.shape) == 0):
                return
            return (resize(chunk, (int(round(chunk.shape[0]*yz_ratio)), chunk.shape[1], int(round(chunk.shape[2]*yx_ratio)))) * 65535).astype("uint16")

        def save_chunk(i, img):
            if img is None:
                return True
            tifffile.imwrite(os.path.join(output_folder, f"chunk_{str(i).zfill(5)}.tif"), img)
            all_zeros = not np.any(img)
            return all_zeros

        missing_chunk_indices = [chunk_indices[x] for x in missing_chunks]

        chunks = [dask.delayed(read_chunk)(i) for n, i in zip(missing_chunks, missing_chunk_indices)]
        resized = [dask.delayed(resize_chunk)(i) for i in chunks]
        saved = [dask.delayed(save_chunk)(i, img) for i, img in zip(missing_chunks, resized)]
        result = dask.compute(saved)
        # np.save(os.path.join(OUTPUT_DIR, f'scale_{scale}', 'zero_chunks1.npy'), np.array(result))
        bg_chunks = [x for x in range(len(saved)) if result[0][x]]
        np.save(os.path.join(OUTPUT_DIR, 'zero_chunks.npy'), bg_chunks)
        with open(os.path.join(OUTPUT_DIR, 'zero_chunks.txt'), 'w') as f:
            f.write(str(bg_chunks))

    scale = SCALE_USED_FOR_MASKS
    scale_factor = 2 ** scale
    tiff_stack = tifffile.imread(FOREGROUND_MASK)
    chunks_folder = os.path.join(OUTPUT_DIR, 'scale_x', 'mask_resized')
    if not os.path.exists(chunks_folder):
        os.makedirs(chunks_folder)
    lazy_tiff_stack = da.array(tiff_stack)
    chunk_shape = list(np.array(CHUNK_SIZE) / scale_factor)
    print(chunk_shape)
    rounded_chunk_shape = [int(round(chunk_shape[0])), int(round(chunk_shape[1])), int(round(chunk_shape[2]))]
    print(rounded_chunk_shape)
    # lazy_tiff_stack = lazy_tiff_stack.rechunk(chunk_shape)
    ratios = (np.array(tiff_stack.shape) / np.array(chunk_shape)).astype('int') + 1
    patchify_chunks_shape = (*list(ratios), *chunk_shape)
    origin_coords = get_origin_coords_rounded(3, patchify_chunks_shape, chunk_shape)
    np.save(os.path.join(chunks_folder, 'origin_coords.npy'), origin_coords)
    chunk_indices = get_chunk_indices_rounded(origin_coords, chunk_shape)
    missing_chunks = [
        x for x in range(len(chunk_indices))
        if not os.path.exists(os.path.join(chunks_folder, f"chunk_{str(x).zfill(5)}.tif"))
    ]
    yx_ratio = float(NUCLEI_RESOLUTION[-1]) / NUCLEI_RESOLUTION[-2]
    yz_ratio = float(NUCLEI_RESOLUTION[-3]) / NUCLEI_RESOLUTION[-2]

    print("Creating downscaled fg/bg masks for chunks:\n", missing_chunks)
    dask_mask_to_tiffs_resize_check_zeros(lazy_tiff_stack, chunk_indices, chunks_folder, yx_ratio, yz_ratio, missing_chunks)


def extract_low_resolution():
    print("Extracting scale", SCALE_USED_FOR_MASKS)
    location = os.path.join(NUCLEI_DIR, f'scale{SCALE_USED_FOR_MASKS}')
    store = H5_Nested_Store(location)
    zarray = zarr.open(store)
    data = zarray[0, 0, :, :, :]
    tifffile.imwrite(os.path.join(OUTPUT_DIR, f"scale{SCALE_USED_FOR_MASKS}_stack.tif"), data)


if __name__ == "__main__":
    import dask.array as da
    import zarr
    from stack_to_multiscale_ngff.archived_nested_store import Archived_Nested_Store
    from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(os.path.join(OUTPUT_DIR, f"scale{SCALE_USED_FOR_MASKS}_stack.tif")):
        extract_low_resolution()

    if not os.path.exists(os.path.join(OUTPUT_DIR, "bg_fg_mask.tif")) or not os.path.exists(os.path.join(OUTPUT_DIR, "bright_mask.tif")):
        raise RuntimeError("Please provide both bg_fg_mask.tif and bright_mask.tif in your output folder")

    get_chunks_with_background()
    get_chunks_with_bright_signal()
