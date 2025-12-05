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
    print("Saving .omehans array")
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

def write_omehans_from_dask(path_to_omehans, data):
    print("Saving .omehans array")
    os.makedirs(path_to_omehans, exist_ok=True)

    compressor = Blosc(cname='zstd', clevel=5, shuffle=1, blocksize=0)

    if data.ndim == 3:
        chunks = (128, 128, 128)
    elif data.ndim == 4:
        chunks = (1, 128, 128, 128)
    else:
        raise ValueError(f"Unsupported ndim={data.ndim}")

    store = H5_Nested_Store(path_to_omehans, "a")
    z = zarr.zeros(store=store, shape=data.shape, chunks=chunks,
                   compressor=compressor, dtype=data.dtype)

    if isinstance(data, da.Array):
        if data.chunks != z.chunks:
            data = data.rechunk(z.chunks)
        da.store(data, z, lock=False)   # streams without full materialization
    else:
        z[...] = data    


def write_zarr(path_to_zarr, numpy_data):
    print("Saving .zarr array")
    if not os.path.exists(path_to_zarr):
        os.makedirs(path_to_zarr)
    if numpy_data.ndim == 2:
        #numpy_data = numpy_data[None, ...]  # (1, Y, X)
        chunks = (128, 128)
    elif numpy_data.ndim == 3:
        chunks = (128, 128, 128)
    elif numpy_data.ndim == 4:
        chunks = (1, 128, 128, 128)
    compressor = zarr.Blosc(cname='zstd', clevel=3)
    z = zarr.open(os.path.join(path_to_zarr, 'array.zarr'), mode='w', shape=numpy_data.shape, dtype=numpy_data.dtype, chunks=chunks, compressor=compressor)
    z[:] = numpy_data


def write_dask_zarr_compatible_with_napari(output_folder, dask_array):
    zarr_path = os.path.join(output_folder, 'array.zarr')

    # Use same chunking and compression as your existing method
    if dask_array.ndim == 3:
        chunks = (128, 128, 128)
    elif dask_array.ndim == 4:
        chunks = (1, 128, 128, 128)

    compressor = Blosc(cname='zstd', clevel=3)

    # Rechunk if necessary to match target
    dask_array = dask_array.rechunk(chunks)

    # Save directly to nested folder (just like your function)
    dask_array.to_zarr(zarr_path, component=None, overwrite=False,
                       compressor=compressor)
