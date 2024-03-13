import logging
import os
import re
import subprocess
import time
from datetime import datetime
from glob import glob

import tifffile
import numpy as np
import pandas as pd
import dask
import dask.array as da
import zarr
from skimage.transform import resize
from stack_to_multiscale_ngff.archived_nested_store import Archived_Nested_Store
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store

from utils.chunks import get_chunk_indices, get_origin_coords
from utils.create_masks import get_chunks_with_background, get_chunks_with_bright_signal
from utils.settings import *

log = logging.getLogger(__name__)


tstart = datetime.now()
log.info(f"START TIME: {tstart}")

nuclei_channel = 0

output_folder_scale = os.path.join(OUTPUT_DIR, f'scale_{SCALE}')
if not os.path.exists(output_folder_scale):
    os.makedirs(output_folder_scale)

# extract low-resolution masks for foreground and bright spots
if not os.path.exists(os.path.join(output_folder_scale, "zero_chunks.npy")):
    get_chunks_with_background()
if not os.path.exists(os.path.join(output_folder_scale, "bright_chunks.npy")):
    get_chunks_with_bright_signal()

# read the nuclei channel (not into memory)
location = os.path.join(NUCLEI_DIR, f'scale{SCALE}')
store = H5_Nested_Store(location)
zarray = zarr.open(store)
dask_zarray = da.array(zarray)
lazy_tiff_stack = dask_zarray[0, nuclei_channel, :, :, :]
log.info(f"3D stack shape {lazy_tiff_stack.shape}")
print("3D stack shape", lazy_tiff_stack.shape)

# create folders for chunks and for SLURM jobs
detection_folder = os.path.join(output_folder_scale, 'detection')
if not os.path.exists(detection_folder):
    os.makedirs(detection_folder)
jobs_folder = os.path.join(output_folder_scale, 'slurm_jobs')
if not os.path.exists(jobs_folder):
    os.makedirs(jobs_folder)

# Get coordinates and indices of each chunk
ratios = (np.array(lazy_tiff_stack.shape) / np.array(CHUNK_SIZE)).astype('int') + 1
patchify_chunks_shape = (*list(ratios), *CHUNK_SIZE)
origin_coords = get_origin_coords(3, patchify_chunks_shape, CHUNK_SIZE)
chunk_indices = get_chunk_indices(origin_coords, CHUNK_SIZE)
np.save(os.path.join(output_folder_scale, "origin_coords.npy"), origin_coords)
np.save(os.path.join(output_folder_scale, "chunk_indices.npy"), chunk_indices)

bg_chunks = set(np.load(os.path.join(output_folder_scale, "zero_chunks.npy")))

fg_chunks = set([x for x in range(len(chunk_indices)) if x not in bg_chunks])
log.info(f"Total foreground chunks: {len(fg_chunks)}")
with open(os.path.join(output_folder_scale, "foreground_chunks.txt"), "w") as f:
    f.write(" ".join(map(str, fg_chunks)))
