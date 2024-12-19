def chunk_data():
    # read the nuclei channel (not into memory)
    location = os.path.join(NUCLEI_DIR, f'scale{settings.SCALE}')
    store = H5_Nested_Store(location)
    zarray = zarr.open(store)
    dask_zarray = da.array(zarray)
    lazy_tiff_stack = dask_zarray[0, nuclei_channel, :, :, :]
    log.info(f"3D stack shape {lazy_tiff_stack.shape}")
    print("3D stack shape", lazy_tiff_stack.shape)

    # Get coordinates and indices of each chunk
    ratios = (np.array(lazy_tiff_stack.shape) / np.array(CHUNK_SIZE)).astype('int') + 1
    patchify_chunks_shape = (*list(ratios), *CHUNK_SIZE)
    origin_coords = get_origin_coords(3, patchify_chunks_shape, CHUNK_SIZE)
    chunk_indices = get_chunk_indices(origin_coords, CHUNK_SIZE)
    np.save(os.path.join(output_folder_scale, "origin_coords.npy"), origin_coords)
    np.save(os.path.join(output_folder_scale, "chunk_indices.npy"), chunk_indices)
