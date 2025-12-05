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
#import argparse

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
#from holis_pipeline.utils.create_folders import create_folders
from holis_pipeline.unchunk_data import combine_masks, remove_chunking_artifacts, combine_centroids_csv, extract_coords, combine_spectral_info_csv
from holis_pipeline.extract_spectral_info import get_spectral_info_slurm
from holis_pipeline.preprocessing_functions import setup_logging, infer_z, infer_laser_nm, EXC_RE


#def run_pipeline(nuclei_fli: str, output_dir: str):

os.umask(0o007)

NUCLEI_FLI = sys.argv[1] #nuclei_fli                  #/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/NPBB328-Cortex-Slab06-run002-z01-y001-Exc-488nm-561nm-594nm-660nm_HiCAM FLUO_1875-ST-272.fli.zst
OUTPUT_DIR = sys.argv[2] #output_dir                  #/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/out_processed/NPBB328-Cortex-Slab06-run002-z01-y001-Exc-488nm-561nm-594nm-660nm_HiCAM FLUO_1875-ST-272.fli


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#output_folder_scale = os.path.join(OUTPUT_DIR, f'scale_{SCALE}')  # TODO: do we need scale? Will it always be full resolution?

COLORS_FLI = NUCLEI_FLI.replace('272.fli.zst', '088.fli.zst')


def main():
    # Log the start time
    tstart = datetime.now()
    log.info(f"START TIME: {tstart}")
    #create_folders(OUTPUT_DIR)

    # Read data to zarr - this might have already been done
    #log.info("Reading data")
    #NUCLEI_DIR = read_fli_as_zarr(NUCLEI_FLI, os.path.join(OUTPUT_DIR, os.path.basename(NUCLEI_FLI)))
    #log.info("Read nuclei channel")
    #COLORS_DIR = read_fli_as_zarr(COLORS_FLI, os.path.join(OUTPUT_DIR, os.path.basename(COLORS_FLI)))
    #log.info("Read color channels")

    NUCLEI_OUT = os.path.join(OUTPUT_DIR, os.path.basename(NUCLEI_FLI)) # path to the scan output directory named using its name
    COLORS_OUT = os.path.join(OUTPUT_DIR, os.path.basename(COLORS_FLI))

    #################################################################################################
    # Run file conversion
    #################################################################################################

    # Read data to zarr
    log.info("Reading data")
    if os.path.isdir(NUCLEI_OUT):
        log.info(f"Nuclei omehans already present at {NUCLEI_OUT}, skipping read.")
    else:
        NUCLEI_OUT = read_fli_as_zarr(NUCLEI_FLI, NUCLEI_OUT)
        log.info("Read nuclei channel")
    if os.path.isdir(COLORS_OUT):
        log.info(f"Colors omehans already present at {COLORS_OUT}, skipping read.")   
    else:
        COLORS_OUT = read_fli_as_zarr(COLORS_FLI, COLORS_OUT)
        log.info("Read color channels")


    log.info("Conversion stage complete.")
    log.info(f"NUCLEI_OUT:  {NUCLEI_OUT}")
    log.info(f"COLORS_OUT: {COLORS_OUT}")
    log.info(f"TOTAL TIME: {datetime.now() - tstart}")

if __name__ == "__main__":
    log = setup_logging(OUTPUT_DIR)
    try:
        main()
    except Exception:
        log.exception("Unhandled exception")
        raise
