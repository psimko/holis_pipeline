import numpy as np


def get_origin_coords(ndim, patchify_chunks_shape, chunk_size):
    """
    Get coordinates of each chunk origin.
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
