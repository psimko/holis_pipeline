import os
from glob import glob

import dask.array as da
import numpy as np
import tifffile

from holis_pipeline import settings
from holis_pipeline.preprocessing_functions import read_data_file
from holis_pipeline.read_data import read_fli_as_zarr
from holis_pipeline.utils.zarr_related import read_omehans


# def subtract_background(input_location, output_location):
def subtract_background(input_location, output_location, bg_file_name):

    """
    input: data .omehans
           empty frames (.mat) - the same shape as the data (1 file per nuclei+colors fli pair)
    output: flattened .omehans - same shape as input
    """
    print("reading zarr")
    image_dask_zarray = read_omehans(input_location)
    image_np_array = image_dask_zarray.compute()
    print("Reading BG file")
    BG_FOLDER = read_fli_as_zarr(bg_file_name, os.path.join(os.path.dirname(bg_file_name), os.path.basename(bg_file_name).replace('.fli', '')))
    bg_dask_zarray = read_omehans(BG_FOLDER)
    bg_array = bg_dask_zarray.compute()
    # # Generate background masks
    # bg_mask = da.mean(da.asarray(bg_array, dtype=np.float32), axis=2) - 2**10
    bg_mask_np = np.mean(bg_array.astype('float32'), axis=2).round()
    print("calculated mean")
    # bg_mask_np = bg_mask.compute()

    # image_dask_zarray_bgSubtracted = image_dask_zarray - bg_mask[:, :, np.newaxis]
    # image_np_array_bgSubtracted = image_dask_zarray_bgSubtracted.compute()
    image_np_array_bgSubtracted = image_np_array.astype('float32') - bg_mask_np[:, :, np.newaxis].astype('float32')
    image_np_array_bgSubtracted[image_np_array_bgSubtracted < 0] = 0
    tifffile.imwrite(os.path.join(output_location, "bg_subtracted.tif"), image_np_array_bgSubtracted.astype('uint16'))

    # image_np_array_bgSubtracted = image_np_array.astype('float32') - bg_array.astype('float32')
    # image_np_array_bgSubtracted[image_np_array_bgSubtracted < 0] = 0
    # print("RAW - BG min", image_np_array_bgSubtracted.min())
    # print("RAW - BG max", image_np_array_bgSubtracted.max())
    # tifffile.imwrite(os.path.join(output_location, "raw_minus_bg.tif"), image_np_array_bgSubtracted.astype('uint16'))
    print("Saved BG-subtracted file")


def split_color_channels(input_location, output_location):
    """
    input: flattened .omehans (only colors)
    output: reshaped 4-channel .omehans
    """
    pass


def laser_correction(input_location, output_location):  # tbd whether needs to be processed separately
    """
    Input:
        - flattened .omehans
        - laser pattern matrix (y,z) shape
        - absorption matrix (n_fluorophores, n_lasers) - (5x4) - first row   for nuclei
        - mixing (fluorescence) matrix (n_fluorophores, n_channels) - (5x5) - first column for nuclei
    Output: corrected .omehans the same shape as input
    """
    laser_pattern_matrix_path = settings.LASER_PATTERN_MATRIX
    absorption_matrix_path = settings.ABSORPTION_MATRIX
    mixing_matrix_path = settings.MIXING_MATRIX


def color_registration(input_location, output_location):
    """
    input: 4-channel corrected colors .omehans
    output:
        - 4-channel corrected colors registered to nuclei space
        - 4 affine matrices (for each color)
    """
    pass


def preprocess_nuclei(spool_file, location):
    bg_subtracted_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_bg_subtracted")
    try:
        os.makedirs(bg_subtracted_location)
    except:
        pass
    bg_file_name = glob(os.path.join(os.path.dirname(spool_file), f"{settings.EMPTY_FRAMES_FILE_NAME_FORMAT}272.fli"))[0]
    subtract_background(location, bg_subtracted_location, bg_file_name)
    laser_corrected_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_laser_corrected")
    try:
        os.makedirs(laser_corrected_location)
    except:
        pass
    laser_correction(bg_subtracted_location, laser_corrected_location)
    preprocessed_location = laser_corrected_location
    return preprocessed_location


def preprocess_colors(spool_file, location):
    bg_subtracted_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_bg_subtracted")
    try:
        os.makedirs(bg_subtracted_location)
    except:
        pass
    bg_file_name = glob(os.path.join(os.path.dirname(spool_file), f"{settings.EMPTY_FRAMES_FILE_NAME_FORMAT}088.fli"))[0]
    subtract_background(location, bg_subtracted_location, bg_file_name)
    color_split_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_color_split")
    try:
        os.makedirs(color_split_location)
    except:
        pass
    split_color_channels(bg_subtracted_location, color_split_location)
    laser_corrected_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_laser_corrected")
    try:
        os.makedirs(laser_corrected_location)
    except:
        pass
    laser_correction(color_split_location, laser_corrected_location)
    color_registered_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_color_registered")
    try:
        os.makedirs(color_registered_location)
    except:
        pass
    color_registration(laser_corrected_location, color_registered_location)
    preprocessed_location = color_registered_location
    return preprocessed_location
