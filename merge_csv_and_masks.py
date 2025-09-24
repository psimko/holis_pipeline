import logging
import os
import re
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import tifffile
import zarr
from skimage.transform import resize
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store

from utils.settings import *

log = logging.getLogger(__name__)


def merge_spectral_info_df():
    bg_chunks = np.load(os.path.join(OUTPUT_DIR, f"scale_{SCALE}", "zero_chunks.npy"))
    spectral_info_folder = os.path.join(OUTPUT_DIR, f"scale_{SCALE}", "spectral_info")
    # get all csv in spectral info folder, check their number
    dfs = sorted(glob(os.path.join(spectral_info_folder, "spectral*.csv")))
    print("total dfs", len(dfs))
    # check they do not belong to bg
    dfs = [x for x in dfs if int(re.findall(r"\d+", os.path.basename(x))[-1]) not in bg_chunks]
    print("Dfs to merge", len(dfs))
    # read them (without dask)
    print("reading")
    to_merge = []
    labels_max = 0
    for df_file in dfs:
        current_chunk = int(re.findall(r"\d+", os.path.basename(df_file))[-1])
        print("processing", current_chunk)
        df = pd.read_csv(df_file)
        df['label'] += labels_max
        labels_max = df['label'].max()
        to_merge.append(df)

    print("done reading. Concatenating")
    # concatenate them (without rescaling)
    df = pd.concat(to_merge, ignore_index=True)
    print("Concatenated. Saving")
    # save new df
    df.to_csv(os.path.join(OUTPUT_DIR, f"scale_{SCALE}", "detected_cells_pytorch_unet_bg_removed_px_w_color_info.csv"))
    return df


def combine_masks(chunk_indices_folder, detection_masks_folder, vol_unmixed_folder):
    print("Merging masks")
    chunk_indices = np.load(chunk_indices_folder, allow_pickle=True)
    store_nuclei = H5_Nested_Store(vol_unmixed_folder) # This is just to get the shape for the combined mask
    zarray_nuclei = zarr.open(store_nuclei)
    raw_img_shape = zarray_nuclei.shape[-3:]
    combined_mask = np.zeros(raw_img_shape, dtype=np.uint8)
    
    masks = sorted(glob(os.path.join(detection_masks_folder, "mask*.tif")))
    for mask in masks:
        chunk_number = int(re.findall(r"\d+", os.path.basename(mask))[-1])
        print("Adding chunk", chunk_number)
        chunk_slices = chunk_indices[chunk_number]
        edge_flag = False
        edge_chunk_shape = CHUNK_SIZE
        if chunk_slices[0].stop >= raw_img_shape[0]:
            edge_flag = True
            chunk_slices[0] = slice(chunk_slices[0].start, raw_img_shape[0], chunk_slices[0].step)
            edge_chunk_shape = (raw_img_shape[0] - chunk_slices[0].start, edge_chunk_shape[1], edge_chunk_shape[2])
        if chunk_slices[1].stop >= raw_img_shape[1]:
            edge_flag = True
            chunk_slices[1] = slice(chunk_slices[1].start, raw_img_shape[1], chunk_slices[1].step)
            edge_chunk_shape = (edge_chunk_shape[0], raw_img_shape[1] - chunk_slices[1].start, edge_chunk_shape[2])
        if chunk_slices[2].stop >= raw_img_shape[2]:
            edge_flag = True
            chunk_slices[2] = slice(chunk_slices[2].start, raw_img_shape[2], chunk_slices[2].step)
            edge_chunk_shape = (edge_chunk_shape[0], edge_chunk_shape[1], raw_img_shape[2] - chunk_slices[2].start)
        mask_resized_space = tifffile.imread(mask)
        print("Read mask of shape", mask_resized_space.shape)
        if edge_flag:
            print("Edge chunk")
            mask_raw_space = resize(mask_resized_space, edge_chunk_shape) > 0
        else:
            mask_raw_space = resize(mask_resized_space, CHUNK_SIZE) > 0
        combined_mask[chunk_slices[0], chunk_slices[1], chunk_slices[2]] = mask_raw_space
        tifffile.imwrite(os.path.join(chunk_indices_folder, "combined_mask_raw_space.tif"), combined_mask)


log.info("Merging spectral info df")
df = merge_spectral_info_df()
# log time when finished
tfin_save_df = datetime.now()

# drop columns with spectral info
df = df[['axis-0', 'axis-1', 'axis-2']]
# save df with just the coordinates
df.to_csv(os.path.join(OUTPUT_DIR, f"scale_{SCALE}", "detected_cells_pytorch_unet_bg_removed_new_px.csv"))

log.info("Combining masks")
combine_masks()
log.info("All Done!")
