import logging
import os

import dask.array as da
import numpy as np
import zarr
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store

from holis_pipeline import settings
from holis_pipeline.utils.chunks import get_chunk_indices, get_origin_coords

log = logging.getLogger(__name__)


def chunk_data(nuclei_dir):
    # read the nuclei channel (not into memory)
    location = os.path.join(nuclei_dir, f'scale{settings.SCALE}')
    store = H5_Nested_Store(location)
    zarray = zarr.open(store)
    dask_zarray = da.array(zarray)
    lazy_tiff_stack = dask_zarray[0, nuclei_channel, :, :, :]
    log.info(f"3D stack shape {lazy_tiff_stack.shape}")
    print("3D stack shape", lazy_tiff_stack.shape)

    # Get coordinates and indices of each chunk
    ratios = (np.array(lazy_tiff_stack.shape) / np.array(settings.CHUNK_SIZE)).astype('int') + 1
    patchify_chunks_shape = (*list(ratios), *settings.CHUNK_SIZE)
    origin_coords = get_origin_coords(3, patchify_chunks_shape, settings.CHUNK_SIZE)
    chunk_indices = get_chunk_indices(origin_coords, settings.CHUNK_SIZE)
    np.save(os.path.join(output_folder_scale, "origin_coords.npy"), origin_coords)
    np.save(os.path.join(output_folder_scale, "chunk_indices.npy"), chunk_indices)
    return chunk_indices
