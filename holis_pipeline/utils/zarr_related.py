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
    store = H5_Nested_Store(path_to_omehans, "a")
    compressor_type = 'zstd'
    compressor_level = 5
    shuffle = 1
    compressor = Blosc(cname=compressor_type,clevel=compressor_level,shuffle=shuffle,blocksize=0)
    array = zarr.zeros(store=store, shape=numpy_data.shape, chunks=(128, 128, 128), compressor=compressor, dtype=numpy_data.dtype)
    array = zarr.open(store, "a")
    array[:, :, :] = numpy_data
