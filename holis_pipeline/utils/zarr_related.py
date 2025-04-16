import os

import dask.array as da
import zarr
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store
from numcodecs import Blosc


def read_omehans(path_to_omehans, scale=None):
    location = os.path.join(path_to_omehans, f'scale{scale}' if scale else "")
    store = H5_Nested_Store(location)
    zarray = zarr.open(store)
    dask_zarray = da.array(zarray)
    return dask_zarray


def write_omehans(path_to_omehans, numpy_data):
    if not os.path.exists(path_to_omehans):
        os.makedirs(path_to_omehans)
    store = H5_Nested_Store(path_to_omehans, "a")
    compressor_type = 'zstd'
    compressor_level = 5
    shuffle = 1
    if numpy_data.ndim == 3:
        chunks = (128, 128, 128)
    elif numpy_data.ndim == 4:
        chunks = (1, 128, 128, 128)
    compressor = Blosc(cname=compressor_type,clevel=compressor_level,shuffle=shuffle,blocksize=0)
    array = zarr.zeros(store=store, shape=numpy_data.shape, chunks=chunks, compressor=compressor, dtype=numpy_data.dtype)
    array = zarr.open(store, "a")
    if numpy_data.ndim == 3:
        array[:, :, :] = numpy_data
    elif numpy_data.ndim == 4:
        for c in range(numpy_data.shape[0]):
            array[c, :, :, :] = numpy_data[c, :, :, :]
