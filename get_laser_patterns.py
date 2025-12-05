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
from get_correction_masks_hicam_hemibrain1 import get_laser_pattern
from holis_pipeline.preprocessing_functions import setup_logging, infer_z, infer_laser_nm, EXC_RE, infer_dark_frames

os.umask(0o007)

NUCLEI_FLI = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#output_folder_scale = os.path.join(OUTPUT_DIR, f'scale_{SCALE}')  # TODO: do we need scale? Will it always be full resolution?

COLORS_FLI = NUCLEI_FLI.replace('272.fli.zst', '088.fli.zst')

#Get z and define mask file names
base_nuclei = os.path.basename(NUCLEI_FLI)  
base_colors = os.path.basename(COLORS_FLI)                                # e.g. NPBB328-...-272.fli.zst
root = os.path.splitext(os.path.splitext(base_nuclei)[0])[0] 


z_str = infer_z(NUCLEI_FLI)
laser = infer_laser_nm(NUCLEI_FLI) 
bg = infer_dark_frames(NUCLEI_FLI)

NUCLEI_OMEHANS_PATH = os.path.join(OUTPUT_DIR, base_nuclei)
COLORS_OMEHANS_PATH = os.path.join(OUTPUT_DIR, base_colors)

# Define path for laser correction masks
if bg:
    laserPattern_nuclei_fileName = f'laser_pattern_nuclei_BG_z{z_str}.npy'
    laserPattern_colors_fileName = f'laser_pattern_colors_BG_z{z_str}.npy'
else:
    laserPattern_nuclei_fileName = f'laser_pattern_nuclei_mask_z{z_str}_{laser}.npy'
    laserPattern_colors_fileName = f'laser_pattern_colors_mask_z{z_str}_{laser}.npy'
NUCLEI_LASER_PATTERN_MASK_PATH = os.path.join(OUTPUT_DIR, laserPattern_nuclei_fileName)
COLORS_LASER_PATTERN_MASK_PATH = os.path.join(OUTPUT_DIR, laserPattern_colors_fileName)


def main():
    # Log the start time
    tstart = datetime.now()
    log.info(f"START TIME: {tstart}")
    #create_folders(OUTPUT_DIR)

    #################################################################################################
    # Convert fli to omehans
    #################################################################################################

    if os.path.isdir(NUCLEI_OMEHANS_PATH):
        log.info(f"Nuclei omehans already present at {NUCLEI_OMEHANS_PATH}, skipping read.")
        nuclei_dir = NUCLEI_OMEHANS_PATH
    else:
        nuclei_dir = read_fli_as_zarr(NUCLEI_FLI, os.path.join(OUTPUT_DIR, os.path.basename(NUCLEI_FLI)))
        log.info(f"Read nuclei channel to {nuclei_dir}")

    if os.path.isdir(COLORS_OMEHANS_PATH):
        log.info(f"Colors omehans already present at {COLORS_OMEHANS_PATH}, skipping read.")
        colors_dir = COLORS_OMEHANS_PATH
    else:
        colors_dir = read_fli_as_zarr(COLORS_FLI, os.path.join(OUTPUT_DIR, os.path.basename(COLORS_FLI)))
        log.info(f"Read color channels to {colors_dir}")

    #################################################################################################



    #################################################################################################
    # Get raw laser correction masks and bg
    #################################################################################################

    log.info("Generating laser correction masks")

    if os.path.exists(NUCLEI_LASER_PATTERN_MASK_PATH):
        log.info(f"Nuclei laser pattern mask already present at {NUCLEI_LASER_PATTERN_MASK_PATH}, skipping read.")
        nuclei_laser_pattern_mask_path = NUCLEI_LASER_PATTERN_MASK_PATH
    else:
        nuclei_laser_pattern_mask_path = get_laser_pattern(nuclei_dir, NUCLEI_LASER_PATTERN_MASK_PATH)
        log.info(f"Saved nuclei laser pattern mask to {nuclei_laser_pattern_mask_path}")

    if os.path.exists(COLORS_LASER_PATTERN_MASK_PATH):
        log.info(f"Colors laser pattern mask already present at {COLORS_LASER_PATTERN_MASK_PATH}, skipping read.")
    else:
        colors_laser_pattern_mask_path = get_laser_pattern(colors_dir, COLORS_LASER_PATTERN_MASK_PATH)
        log.info(f"Saved colors laser pattern mask to {colors_laser_pattern_mask_path}")
    
    #################################################################################################



if __name__ == "__main__":
    log = setup_logging(OUTPUT_DIR)
    try:
        main()
    except Exception:
        log.exception("Unhandled exception")
        raise