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
from holis_pipeline.settings import *
from holis_pipeline.chunk_data import chunk_data
from holis_pipeline.read_data import read_fli_as_zarr
from holis_pipeline.detect_nuclei import detect_cells_slurm
from holis_pipeline.utils.create_masks import get_chunks_with_bright_signal, get_chunks_with_background
from holis_pipeline.preprocess_v2_sep2025 import preprocess_nuclei, preprocess_colors # , unmix_data, nuclei_color_registration
from holis_pipeline.utils.create_folders import create_folders
from holis_pipeline.unchunk_data import combine_masks, remove_chunking_artifacts, combine_centroids_csv, extract_coords, combine_spectral_info_csv
from holis_pipeline.extract_spectral_info import get_spectral_info_slurm


"""
TODO:
command-line arguments - Inputs (colors, nuclei) and output
"""

os.umask(0o006)

NUCLEI_FLI = sys.argv[1]
# COLORS_FLI = sys.argv[2]
OUTPUT_DIR = sys.argv[2]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

output_folder_scale = os.path.join(OUTPUT_DIR, f'scale_{SCALE}')  # TODO: do we need scale? Will it always be full resolution?

COLORS_FLI = NUCLEI_FLI.replace('272.fli', '088.fli')

JOBS_DIR = os.path.join(OUTPUT_DIR, f'scale_{SCALE}', 'slurm_jobs')
DETECTION_DIR = os.path.join(OUTPUT_DIR, f'scale_{SCALE}', 'detection')
SPECTRAL_INFO_DIR = os.path.join(OUTPUT_DIR, f'scale_{SCALE}', 'spectral_info')

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
    log.info("Reading data")
    NUCLEI_DIR = read_fli_as_zarr(NUCLEI_FLI, os.path.join(OUTPUT_DIR, os.path.basename(NUCLEI_FLI)))
    log.info("Read nuclei channel")
    COLORS_DIR = read_fli_as_zarr(COLORS_FLI, os.path.join(OUTPUT_DIR, os.path.basename(COLORS_FLI)))
    log.info("Read color channels")

    # === Preprocess ===
    print("Preprocessing nuclei… (BG → laser-correct)")
    NUCLEI_PREP = preprocess_nuclei(NUCLEI_FLI, NUCLEI_DIR)
    print("Nuclei preprocessed at:", NUCLEI_PREP)

    print("Preprocessing colors… (BG → laser-correct → split → register → unmix)")
    COLORS_UNMIXED = preprocess_colors(COLORS_FLI, COLORS_DIR, NUCLEI_PREP)
    print("Colors unmixed at:", COLORS_UNMIXED)

    log.info("Preprocessing stages complete.")
    log.info(f"NUCLEI_PREP:  {NUCLEI_PREP}")
    log.info(f"COLORS_UNMIXED: {COLORS_UNMIXED}")
    log.info(f"TOTAL TIME: {datetime.now() - tstart}")

    
    chunk_indices = chunk_data(COLORS_UNMIXED, output_folder_scale)
    print(f'Chunk indices: {chunk_indices}')
    
    if FOREGROUND_MASKS_ENABLED:
        # extract low-resolution masks for foreground
        if not os.path.exists(os.path.join(output_folder_scale, "zero_chunks.npy")):
            get_chunks_with_background()
        bg_chunks = set(np.load(os.path.join(output_folder_scale, "zero_chunks.npy")))
    else:
        bg_chunks = set()
    
    if DENSE_REGION_MASK_ENABLED:
        # extract low-resolution masks for bright spots
        if not os.path.exists(os.path.join(output_folder_scale, "bright_chunks.npy")):
            get_chunks_with_bright_signal()
        bright_chunks = set(np.load(os.path.join(output_folder_scale, "bright_chunks.npy")))
    else:
        bright_chunks = set()
    
    # ================= Create nuclei detection jobs =================
    
    # check what chunks got extracted - save this info (compare with fg chunks list)
    # for extracted chunks generate and submit nuclei detection jobs
    fg_chunks = set([x for x in range(len(chunk_indices)) if x not in bg_chunks and x not in bright_chunks])
    log.info(f"Total foreground chunks: {len(fg_chunks)}")
    detect_cells_slurm(list(fg_chunks), COLORS_UNMIXED, JOBS_DIR, DETECTION_DIR)
    log.info("All nuclei detection tasks were submitted")
    
    # check which csv files have been generated
    sent_tasks = set()
    remaining_chunks = fg_chunks.copy()

    centroids_folder = os.path.join(DETECTION_DIR, 'centroids')
    try:
        os.makedirs(centroids_folder)
    except FileExistsError:
        pass

    while len(remaining_chunks):
        print("Chunks remaining to do nuclei detection", len(remaining_chunks))
        detection_done = set([
            int(re.findall(r"\d+", os.path.basename(x))[-1]) for x in glob(os.path.join(centroids_folder, "*.csv"))
        ])
        chunk_numbers_set = detection_done - sent_tasks
        sent_tasks.update(chunk_numbers_set)
        remaining_chunks = fg_chunks - sent_tasks
        time.sleep(2)
    
    log.info("All nuclei detection jobs finished")
    tfin_detection = datetime.now()
    log.info(f"Time spent on extraction + nuclei detection: {tfin_detection - tstart}")
    
    # # ================= Merge masks and df with coordinates =================

    detection_masks_folder = os.path.join(DETECTION_DIR, 'detection_masks')
    try:
        os.makedirs(detection_masks_folder)
    except FileExistsError:
        pass
    
    combined_mask_location = combine_masks(detection_masks_folder, COLORS_UNMIXED, output_folder_scale, output_folder_scale)   #DETECTION_DIR is where the chunk indices are stored
    # no_artifact_mask_location = remove_chunking_artifacts(combined_mask_location)
    #coords_file = extract_coords(combined_mask_location, output_folder_scale, coord_order="zyx", connectivity=1, min_size=4, float_dtype=np.float32)


    
    combined_centroids_location = combine_centroids_csv(
        centroids_folder,
        detection_masks_folder,
        COLORS_UNMIXED,
        output_folder_scale,
        chunk_indices_name="chunk_indices.npy",
        csv_pattern="*.csv",
        mask_pattern="*.tif",
        centroid_prefix="centroids",   # anchor for parsing chunk id from CSV names
        mask_prefix="centroids",            # anchor for parsing chunk id from mask names
        coord_order="zyx",             # your CSVs appear to be (z,y,x)
        dedupe=True,
        output_file_name="combined_centroids.csv"
    )
    
    get_spectral_info_slurm(list(fg_chunks), output_folder_scale, centroids_folder, detection_masks_folder, COLORS_UNMIXED, JOBS_DIR, SPECTRAL_INFO_DIR)
    #spectral_info_location = get_spectral_info_slurm(list(fg_chunks), output_folder_scale, centroids_folder, detection_masks_folder, COLORS_UNMIXED, JOBS_DIR, SPECTRAL_INFO_DIR)  # single task? No chunks? 
    #spectral_info_location = # single task

    # check which csv files have been generated
    sent_tasks = set()
    remaining_chunks = fg_chunks.copy()

    while len(remaining_chunks):
        print("Chunks remaining to do spectral extraction", len(remaining_chunks))
        detection_done = set([
            int(re.findall(r"\d+", os.path.basename(x))[-1]) for x in glob(os.path.join(SPECTRAL_INFO_DIR, "*.csv"))
        ])
        chunk_numbers_set = detection_done - sent_tasks
        sent_tasks.update(chunk_numbers_set)
        remaining_chunks = fg_chunks - sent_tasks
        time.sleep(2)

    combined_spectral_info_location = combine_spectral_info_csv(
    SPECTRAL_INFO_DIR,
    os.path.join(output_folder_scale, "combined_spectral_info.csv"),
    csv_pattern="spectral_chunk_*.csv",
    spectral_prefix="spectral_chunk",
    coord_cols=("axis-0","axis-1","axis-2"),
    round_coords=True,       
    dedupe=True,
    dedupe_strategy="max_vol_l1",  # 'first' | 'max_vol_l1' | 'mean'
    )
    

    print("Spectral information saved at", combined_spectral_info_location)
    log.info("All Done!")


if __name__ == "__main__":
    main()
