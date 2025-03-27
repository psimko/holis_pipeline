"""
Pipeline for the human hemibrain.

This is a CPU task for SLURM. It spins off several GPU tasks.

GPU tasks run Pytorch UNet with a model trained on human data.

Takes in one spool file for nuclei and corresponding spool file for colors.
Outputs detected nuclei and extracted spectral information.

inputs:
- 1 hicam (.fli) file (nuclei - camera the suffix is 272)
- 1 hicam (.fli) file (colors - camera the suffix is 088)

1) read fli files to zarr (nuclei, colors)
2) subtract background (nuclei, colors)
3) split color channels (colors)
4) do laser correction (nuclei, colors)
4) unmixing (nuclei, colors)
5) chunk (nuclei)
6) run pytorch unet on all chunks
7) fix chunking artifacts
8) unchunk (nuclei, colors)
9) save coordinates (nuclei)
10) color_registration (colors)
11) get spectral information on all chunks (nuclei, colors)
12) delete intermediate files (zarr, preprocessed zarr, any chunks)

outputs:
- 1 combined mask of nuclei
- 1 csv file with coordinates
- 1 csv file with spectral information
"""

import logging
import os
import re
import subprocess
import sys
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

# from utils.create_masks import get_chunks_with_background, get_chunks_with_bright_signal
# from utils.settings import *
from holis_pipeline import settings
from holis_pipeline.chunk_data import chunk_data
from holis_pipeline.read_data import read_fli_as_zarr
from holis_pipeline.detect_nuclei import detect_cells_deepblink_slurm
from holis_pipeline.utils.create_masks import get_chunks_with_bright_signal, get_chunks_with_background
from holis_pipeline.preprocess import preprocess_nuclei, preprocess_colors
from holis_pipeline.utils.create_folders import create_folders
from holis_pipeline.unchunk_data import combine_masks, remove_chunking_artifacts, extract_coords
from holis_pipeline.extract_spectral_info import get_spectral_info

"""
TODO:
command-line arguments - Inputs (colors, nuclei) and output
"""

NUCLEI_FLI = sys.argv[1]
# COLORS_FLI = sys.argv[2]
OUTPUT_DIR = sys.argv[2]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

COLORS_FLI = NUCLEI_FLI.replace('272.fli', '088.fli')

work_dir = os.getcwd()
print("Working directory: ", work_dir)

file_handler = logging.FileHandler(
    os.path.join(
        OUTPUT_DIR,
        "holis_pipeline_log_{}.txt".format(
            datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        )
    )
)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler]
)
log = logging.getLogger(__name__)


def main():
    # Log the start time
    tstart = datetime.now()
    log.info(f"START TIME: {tstart}")
    create_folders(OUTPUT_DIR)

    # Read data to zarr
    print("Reading data")
    NUCLEI_DIR = read_fli_as_zarr(NUCLEI_FLI, os.path.join(OUTPUT_DIR, os.path.basename(NUCLEI_FLI)))
    print("Read nuclei channel")
    COLORS_DIR = read_fli_as_zarr(COLORS_FLI, os.path.join(OUTPUT_DIR, os.path.basename(COLORS_FLI)))
    print("Read color channels")

    # Preprocess data
    print("Preprocessing data")
    NUCLEI_DIR = preprocess_nuclei(NUCLEI_FLI, NUCLEI_DIR)
    print("Preprocessed nuclei")
    COLORS_DIR = preprocess_colors(COLORS_FLI, COLORS_DIR)  # TODO separate task
    print("Preprocessed colors")

    # output_folder_scale = os.path.join(OUTPUT_DIR, f'scale_{SCALE}')  # TODO: do we need scale? Will it always be full resolution?
    #
    # chunk_indices = chunk_data(NUCLEI_DIR)
    #
    # if FOREGROUND_MASKS_ENABLED:
    #     # extract low-resolution masks for foreground
    #     if not os.path.exists(os.path.join(output_folder_scale, "zero_chunks.npy")):
    #         get_chunks_with_background()
    #     bg_chunks = set(np.load(os.path.join(output_folder_scale, "zero_chunks.npy")))
    # else:
    #     bg_chunks = set()
    #
    # if DENSE_REGION_MASK_ENABLED:
    #     # extract low-resolution masks for bright spots
    #     if not os.path.exists(os.path.join(output_folder_scale, "bright_chunks.npy")):
    #         get_chunks_with_bright_signal()
    #     bright_chunks = set(np.load(os.path.join(output_folder_scale, "bright_chunks.npy")))
    # else:
    #     bright_chunks = set()
    #
    # # ================= Create nuclei detection jobs =================
    #
    # # check what chunks got extracted - save this info (compare with fg chunks list)
    # # for extracted chunks generate and submit nuclei detection jobs
    # fg_chunks = set([x for x in range(len(chunk_indices)) if x not in bg_chunks and x not in bright_chunks])
    # log.info(f"Total foreground chunks: {len(fg_chunks)}")
    # detect_cells_deepblink_slurm(list(fg_chunks), jobs_folder)
    # log.info("All nuclei detection tasks were submitted")
    #
    # # check which csv files have been generated
    # spectral_info_folder = os.path.join(output_folder_scale, 'spectral_info')
    # sent_tasks = set()
    # remaining_chunks = fg_chunks.copy()
    # while len(remaining_chunks):
    #     print("Chunks remaining to do nuclei detection", len(remaining_chunks))
    #     detection_done = set([
    #         int(re.findall(r"\d+", os.path.basename(x))[-1]) for x in glob(os.path.join(detection_folder, "*.csv"))
    #     ])
    #     chunk_numbers_set = detection_done - sent_tasks
    #     sent_tasks.update(chunk_numbers_set)
    #     remaining_chunks = fg_chunks - sent_tasks
    #     time.sleep(2)
    #
    # log.info("All nuclei detection jobs finished")
    # tfin_detection = datetime.now()
    # log.info(f"Time spent on extraction + nuclei detection: {tfin_detection - tstart}")
    #
    # # ================= Merge masks and df with coordinates =================
    #
    # combined_mask_location = combine_masks(NUCLEI_DIR)
    # no_artifact_mask_location = remove_chunking_artifacts(combined_mask_location)
    # coords_file = extract_coords(no_artifact_mask_location, OUTPUT_DIR)
    #
    # spectral_info_df = get_spectral_info(coords_file, COLORS_DIR)  # single task? No chunks?
    # # TODO: should spectral info be extracted after all neighbor slabs are finished?
    # print("Spectral information saved at", spectral_info_df)
    # log.info("All Done!")


if __name__ == "__main__":
    main()
