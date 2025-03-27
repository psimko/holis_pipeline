import os

import dask.array as da
import zarr
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store


def read_omehans(path_to_omehans, scale=None):
    location = os.path.join(path_to_omehans, f'scale{scale}' if scale else "")
    store = H5_Nested_Store(location)
    zarray = zarr.open(store)
    dask_zarray = da.array(zarray)
    return dask_zarray
