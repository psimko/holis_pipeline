"""
Pipeline for the combinatorial mouse slab.

Uses Pytorch UNet with a model trained on mouse data.

1) napari_apoc segmentation with existing model (done at CBI)
2) Compute bg chunks, save their numbers (done at CBI)
3) compute chunks with bright spots, save their numbers (done at CBI)
4) extract good fg chunks, resize them (with slurm)
5) extract and inpaint fg chunks that have bright spots (with slurm)
6) run pytorch unet on all chunks
7) get spectral information
8) save giant dataframe

requirements to run @ BIL:
put the .npy files for background chunks and bright chunks in the output folder
extract scale 4 mask of bright spots (in chunks)
"""

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

from utils.settings import *


signal_channel = 0
resolution_level = 0
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


def write_detection_task_for_slurm(img_path, output_path):
    with open(output_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('module load miniconda3\n')
        f.write(f'source activate {GPU_ENV_NAME}')
        f.write('\n')
        f.write(f'python {work_dir}/predict_w_patchify_2.py ')  # TODO
        f.write(MODEL_PATH)
        f.write(' ')
        f.write(img_path)
        f.write('\n')


def submit_slurm_task_gpu(path_to_task):
    command = ['sbatch', '-p', 'gpu', '--gres=gpu:1', '--mem=64Gb', '-n8', path_to_task]
    subprocess.run(command)


def detect_cells_deepblink_slurm(chunk_numbers, chunks_folder, jobs_folder):
    for chunk_number in chunk_numbers:
        chunk_file = os.path.join(chunks_folder, f"chunk_{str(chunk_number).zfill(5)}.tif")
        detections_file_name = os.path.join(os.path.dirname(chunk_file), f"napari_{os.path.basename(chunk_file).replace('.tif', '.csv')}")
        if os.path.exists(detections_file_name):
            print(f"Skipping chunk {chunk_number}")
            continue
        print(f"Submitting gpu task for chunk {chunk_number}")
        task_path = os.path.join(jobs_folder, f"detect_chunk_{str(chunk_number).zfill(5)}.sh")
        write_detection_task_for_slurm(chunk_file, task_path)
        submit_slurm_task_gpu(task_path)


def get_origin_coords(ndim, patchify_chunks_shape, chunk_size):
    """
    Get coordinates of each chunk origin.
    TODO: only 3D now, make compatible with ND
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


def dask_chunks_to_tiffs_resize(lazy_tiff_stack, chunk_indices, output_folder, yx_ratio, missing_chunks):

    def read_chunk(ind):
        print("Reading chunk", ind)
        return lazy_tiff_stack[tuple(ind)]

    def resize_chunk(chunk):
        print("Resizing chunk")
        return (resize(chunk, (chunk.shape[0], chunk.shape[1], int(round(chunk.shape[2]*yx_ratio)))) * 65535).astype("uint16")

    def save_chunk(i, img):
        tifffile.imwrite(os.path.join(output_folder, f"chunk_{str(i).zfill(5)}.tif"), img)
        return True

    missing_chunk_indices = [chunk_indices[x] for x in missing_chunks]

    chunks = [dask.delayed(read_chunk)(i) for n, i in zip(missing_chunks, missing_chunk_indices)]
    resized = [dask.delayed(resize_chunk)(i) for i in chunks]
    saved = [dask.delayed(save_chunk)(i, img) for i, img in zip(missing_chunks, resized)]
    saved = dask.compute(saved)


def convert_to_napari_format(chunks_folder):
    """
    Change columns in csv file to make it readable with napari.

    :param chunks_folder:
    :return:
    """
    csv_files = sorted(glob(os.path.join(chunks_folder, '*.csv')))
    csv_files = [
        f for f in csv_files if (
        not os.path.basename(f).startswith('napari') and
        not os.path.exists(os.path.join(os.path.dirname(f), f"napari_{os.path.basename(f)}"))
    )]
    print("CSV files to convert:", len(csv_files))

    def convert(csv_file):
        print("Converting", csv_file)
        napari_csv_file_path = os.path.join(os.path.dirname(csv_file), f"napari_{os.path.basename(csv_file)}")
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError as e:
            print("Warning: ", e)
            return
        df2 = pd.DataFrame()
        df2['index'] = list(range(df.shape[0]))  # TODO
        try:
            zvals = df['z'].tolist()
            yvals = df['y [px]'].tolist()
            xvals = df['x [px]'].tolist()
        except KeyError as e:
            print(e)
            return

        df2['axis-0'] = zvals
        df2['axis-1'] = xvals
        df2['axis-2'] = yvals
        df2.to_csv(napari_csv_file_path)

    converted = [dask.delayed(convert)(f) for f in csv_files]
    converted = dask.compute(converted)


def merge_df_fix_wrong_scaling_no_bg(chunks_folder, origin_coords, xy_factor):
    df_column_names = ['index', 'axis-0', 'axis-1', 'axis-2']  # TODO: ndim
    df = pd.DataFrame(columns=df_column_names)
    csv_files = sorted(glob(os.path.join(chunks_folder, 'napari*.csv')))
    bg_chunks = set(np.load(os.path.join(OUTPUT_DIR, 'zero_chunks.npy')))
    csv_files = [x for x in csv_files if int(re.findall(r"\d+", os.path.basename(x))[-1]) not in bg_chunks]
    print("CSV files", len(csv_files))

    def read_df(chunk_file):
        print("reading", chunk_file)
        return pd.read_csv(chunk_file)

    def process_df(chunk_file, chunk_df):
        current_chunk = int(re.findall(r"\d+", os.path.basename(chunk_file))[-1])
        print("processing", current_chunk)
        chunk_df_corrected = pd.DataFrame()
        z_values = chunk_df[['axis-0']].to_numpy()
        y_values = chunk_df[['axis-1']].to_numpy()
        x_values = chunk_df[['axis-2']].to_numpy()
        x_values = x_values / xy_factor
        z_values += origin_coords[current_chunk, 0]
        y_values += origin_coords[current_chunk, 1]
        x_values += origin_coords[current_chunk, 2]
        chunk_df_corrected['index'] = list(range(chunk_df.shape[0]))
        chunk_df_corrected['axis-0'] = z_values
        chunk_df_corrected['axis-1'] = y_values
        chunk_df_corrected['axis-2'] = x_values
        return chunk_df_corrected

    dfs = [dask.delayed(read_df)(f) for f in csv_files]
    processed = [dask.delayed(process_df)(f, d_f) for f, d_f in zip(csv_files, dfs)]
    processed = dask.compute(processed)
    df = pd.concat(*processed)
    print("Saving df")
    df['index'] = list(range(df.shape[0]))
    return df


def submit_slurm_task_compute(task_path):
    command = ['sbatch', '-p', 'compute', '--mem=128Gb', '-n4', task_path]
    subprocess.run(command)


def write_spectral_extraction_script_for_slurm(chunk_file, task_path):
    with open(task_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('module load miniconda3\n')
        f.write(f'source activate {LNODE_ENV_NAME}')
        f.write('\n')
        f.write(f'python {work_dir}/holis_get_chunk_spectral_info_slab_mouse.py ')
        f.write(chunk_file)
        f.write(' ')
        f.write(NUCLEI_DIR)
        f.write('\n')


def get_spectral_info_slurm(chunk_numbers, chunks_folder, jobs_folder, spectral_info_folder):
    # exclude already saved chunks
    # write bash scripts
    # submit to compute partition
    for chunk_number in chunk_numbers:
        if os.path.exists(os.path.join(spectral_info_folder, f"spectral_chunk_{str(chunk_number).zfill(5)}.csv")):
            print(f"Skipping chunk {chunk_number}")
            continue
        chunk_file = os.path.join(chunks_folder, f"chunk_{str(chunk_number).zfill(5)}.tif")
        task_path = os.path.join(jobs_folder, f"get_color_info_chunk_{str(chunk_number).zfill(5)}.sh")
        write_spectral_extraction_script_for_slurm(chunk_file, task_path)
        submit_slurm_task_compute(task_path)


def write_chunk_extraction_script_for_slurm(chunk_file, task_path):
    with open(task_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('module load miniconda3\n')
        f.write(f'source activate {LNODE_ENV_NAME}')
        f.write('\n')
        f.write(f'python {work_dir}/holis_extract_resized_chunk_slab_mouse.py ')
        f.write(chunk_file)
        f.write(' ')
        f.write(NUCLEI_DIR)
        f.write('\n')


def write_chunk_inpainting_script_for_slurm(chunk_file, task_path):
    with open(task_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('module load miniconda3\n')
        f.write(f'source activate {LNODE_ENV_NAME}')
        f.write('\n')
        f.write(f'python {work_dir}/holis_extract_resized_inpainted_chunk_slab_mouse.py ')
        f.write(chunk_file)
        f.write(' ')
        f.write(NUCLEI_DIR)
        f.write('\n')


def save_resized_chunks_slurm(chunks_folder, jobs_folder, chunks_to_save):
    for chunk_number in chunks_to_save:
        chunk_file = os.path.join(chunks_folder, f"chunk_{str(chunk_number).zfill(5)}.tif")
        task_path = os.path.join(jobs_folder, f"extract_chunk_{str(chunk_number).zfill(5)}.sh")
        # write bash script (slurm task)
        write_chunk_extraction_script_for_slurm(chunk_file, task_path)
        # send it to compute
        submit_slurm_task_compute(task_path)


def save_resized_inpainted_chunks_slurm(chunks_folder, jobs_folder,  chunks_to_save):
    for chunk_number in chunks_to_save:
        chunk_filename = os.path.join(chunks_folder, f"chunk_{str(chunk_number).zfill(5)}.tif")
        task_path = os.path.join(jobs_folder, f"extract_chunk_{str(chunk_number).zfill(5)}.sh")
        # write bash script (slurm task)
        write_chunk_inpainting_script_for_slurm(chunk_filename, task_path)
        # send it to compute
        submit_slurm_task_compute(task_path)


def merge_spectral_info_df(origin_coords, bg_chunks):
    spectral_info_folder = os.path.join(OUTPUT_DIR, "scale_0", "spectral_info")
    # get all csv in spectral info folder, check their number
    dfs = sorted(glob(os.path.join(spectral_info_folder, "spectral*.csv")))
    print("total dfs", len(dfs))
    # check they do not belong to bg
    dfs = [x for x in dfs if int(re.findall(r"\d+", os.path.basename(x))[-1]) not in bg_chunks]
    print("Dfs to merge", len(dfs))
    # read them (without dask)
    print("reading")
    to_merge = []
    for df_file in dfs:
        current_chunk = int(re.findall(r"\d+", os.path.basename(df_file))[-1])
        print("processing", current_chunk)
        df = pd.read_csv(df_file)
        to_merge.append(df)

    print("done reading. Concatenating")
    # concatenate them (without rescaling)
    df = pd.concat(to_merge, ignore_index=True)
    print("Concatenated. Saving")
    # save new df
    df.to_csv(os.path.join(OUTPUT_DIR, "scale_0", "detected_cells_pytorch_unet_bg_removed_px_w_color_info.csv"))
    return df


def main():
    # Log the start time
    tstart = datetime.now()
    log.info(f"START TIME: {tstart}")

    location = os.path.join(NUCLEI_DIR, f'scale{resolution_level}')
    store = H5_Nested_Store(location)
    zarray = zarr.open(store)
    dask_zarray = da.array(zarray)
    lazy_tiff_stack = dask_zarray[0, signal_channel, :, :, :]
    log.info(f"3D stack shape {lazy_tiff_stack.shape}")
    print("3D stack shape", lazy_tiff_stack.shape)

    chunks_folder = os.path.join(OUTPUT_DIR, f'scale_{resolution_level}', 'chunks_resized')
    if not os.path.exists(chunks_folder):
        os.makedirs(chunks_folder)
    jobs_folder = os.path.join(OUTPUT_DIR, f'scale_{resolution_level}', 'slurm_jobs')
    if not os.path.exists(jobs_folder):
        os.makedirs(jobs_folder)

    print("Chunks folder", chunks_folder)
    ratios = (np.array(lazy_tiff_stack.shape) / np.array(CHUNK_SIZE)).astype('int') + 1
    patchify_chunks_shape = (*list(ratios), *CHUNK_SIZE)
    origin_coords = get_origin_coords(3, patchify_chunks_shape, CHUNK_SIZE)
    chunk_indices = get_chunk_indices(origin_coords, CHUNK_SIZE)

    yx_ratio = float(NUCLEI_RESOLUTION[-1]) / NUCLEI_RESOLUTION[-2]  # make resolution isotropic, equal y resolution (only for spot detection)
    yz_ratio = float(NUCLEI_RESOLUTION[-3]) / NUCLEI_RESOLUTION[-2]  # make resolution isotropic, equal y resolution (only for spot detection)

    bg_chunks = set(np.load(os.path.join(OUTPUT_DIR, "zero_chunks.npy")))
    bright_chunks = set(np.load(os.path.join(OUTPUT_DIR, "bright_chunks.npy")))

    # ================= Save foreground chunks with no bright spots in them ================

    # The chunks are saved at isotropic resolution for deepblink
    chunks_to_save = [
        x for x in range(len(chunk_indices))
        if not os.path.exists(os.path.join(chunks_folder, f"chunk_{str(x).zfill(5)}.tif"))
        and x not in bg_chunks
        and x not in bright_chunks
    ]
    save_resized_chunks_slurm(chunks_folder, jobs_folder, chunks_to_save)
    log.info(f"Submitted all jobs to extract normal foreground chunks")

    # ================= Save foreground chunks with bright spots in them =================

    # The chunks are saved at isotropic resolution and inpainted with average where the signal is too bright
    chunks_to_save = [
        x for x in range(len(chunk_indices))
        if not os.path.exists(os.path.join(chunks_folder, f"chunk_{str(x).zfill(5)}.tif"))
        and x not in bg_chunks
        and x in bright_chunks
    ]
    save_resized_inpainted_chunks_slurm(chunks_folder, jobs_folder, chunks_to_save)
    log.info(f"Submitted all jobs to extract foreground chunks with bright spots")

    tstart_extraction = datetime.now()
    log.info(f"Time when all extraction jobs were submitted: {tstart_extraction}")

    # ================= Create nuclei detection jobs =================

    # check what chunks got extracted - save this info (compare with fg chunks list)
    # for extracted chunks generate and submit nuclei detection jobs
    fg_chunks = set([x for x in range(len(chunk_indices)) if x not in bg_chunks])
    log.info(f"Total foreground chunks: {len(fg_chunks)}")
    sent_tasks = set()
    remaining_chunks = fg_chunks.copy()
    while len(remaining_chunks):
        print("Chunks remaining to be extracted", len(remaining_chunks))
        extracted_chunks = set([
            int(re.findall(r"\d+", os.path.basename(x))[-1]) for x in glob(os.path.join(chunks_folder, "*.tif"))
        ])
        chunk_numbers_set = extracted_chunks - sent_tasks
        detect_cells_deepblink_slurm(list(chunk_numbers_set), chunks_folder, jobs_folder)
        sent_tasks.update(chunk_numbers_set)
        remaining_chunks = fg_chunks - sent_tasks
        time.sleep(2)

    log.info("All chunks were extracted")
    tfin_extraction = datetime.now()
    log.info(f"Time spent on extraction: {tfin_extraction - tstart}")
    log.info("All nuclei detection tasks were submitted")

    # ================= Create spectral extraction jobs =================

    # check which csv files have been generated
    # send these chunks for spectral information
    spectral_info_folder = os.path.join(OUTPUT_DIR, f'scale_{resolution_level}', 'spectral_info')
    sent_tasks = set()
    remaining_chunks = fg_chunks.copy()
    while len(remaining_chunks):
        print("Chunks remaining to do nuclei detection", len(remaining_chunks))
        detection_done = set([
            int(re.findall(r"\d+", os.path.basename(x))[-1]) for x in glob(os.path.join(chunks_folder, "*.csv"))
        ])
        chunk_numbers_set = detection_done - sent_tasks
        get_spectral_info_slurm(list(chunk_numbers_set), chunks_folder, jobs_folder, spectral_info_folder)
        sent_tasks.update(chunk_numbers_set)
        remaining_chunks = fg_chunks - sent_tasks
        time.sleep(2)

    log.info("All nuclei detection jobs finished")
    tfin_detection = datetime.now()
    log.info(f"Time spent on extraction + nuclei detection: {tfin_detection - tstart}")
    log.info("All spectral extraction jobs were submitted")

    # check which spectral info files are finished
    # log time when all are done

    remaining_chunks = fg_chunks.copy()
    while len(remaining_chunks):
        print("Chunks remaining to extract spectral info", len(remaining_chunks))
        print(remaining_chunks)
        spectral_info_done = set([
            int(re.findall(r"\d+", os.path.basename(x))[-1]) for x in glob(os.path.join(spectral_info_folder, "*.csv"))
        ])
        remaining_chunks = fg_chunks - spectral_info_done
        time.sleep(2)

    log.info("All spectral extraction jobs finished")
    tfin_spectral_info = datetime.now()
    log.info(f"Time spent on extraction + nuclei detection + spectral info: {tfin_spectral_info - tstart}")


    # merge and save the final df
    log.info("Merging spectral info df")
    df = merge_spectral_info_df(origin_coords, bg_chunks)
    # log time when finished
    tfin_save_df = datetime.now()
    log.info(f"Time spent on extraction + nuclei detection + spectral info + merging df: {tfin_save_df - tstart}")

    # drop columns with spectral info
    df = df[['axis-0', 'axis-1', 'axis-2']]
    # save df with just the coordinates
    df.to_csv(os.path.join(OUTPUT_DIR, "scale_0", "detected_cells_pytorch_unet_bg_removed_new_px.csv"))


if __name__ == "__main__":
    main()
