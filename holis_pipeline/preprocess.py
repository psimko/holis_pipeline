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


def split_color_channels(data_temp):
    """
    Takes a 3D image splitter data and outputs a 4D array with shape (ch, z, y, x).
    """

    ss = data_temp.shape

    # Define four channel regions by splitting along y and x axes
    temp_ch1 = data_temp[round(0.5 * ss[0]):ss[0], round(0.5 * ss[1]):ss[1], :]
    temp_ch2 = data_temp[round(0.5 * ss[0]):ss[0], :round(0.5 * ss[1]), :]
    temp_ch3 = data_temp[:round(0.5 * ss[0]), round(0.5 * ss[1]):ss[1], :]
    temp_ch4 = data_temp[:round(0.5 * ss[0]), :round(0.5 * ss[1]), :]
    
    #     temp_ch1 = np.asarray(temp_ch1)
    #     temp_ch2 = np.asarray(temp_ch2)
    #     temp_ch3 = np.asarray(temp_ch3)
    #     temp_ch4 = np.asarray(temp_ch4)

    print(temp_ch1.shape)
    print(temp_ch2.shape)
    print(temp_ch3.shape)
    print(temp_ch4.shape)

    # Optional offsets for each channel (if needed)
    y = da.zeros(4, dtype=int)
    z = da.zeros(4, dtype=int)
    print(y)
    print(z)

    # Crop ranges for each channel
    cropY = 640
    cropZ = 512

    # Apply cropping to each channel
    temp_ch1 = temp_ch1[z[0]:z[0] + cropZ, y[0]:y[0] + cropY, :]
    temp_ch2 = temp_ch2[z[1]:z[1] + cropZ, y[1]:y[1] + cropY, :]
    temp_ch3 = temp_ch3[z[2]:z[2] + cropZ, y[2]:y[2] + cropY, :]
    temp_ch4 = temp_ch4[z[3]:z[3] + cropZ, y[3]:y[3] + cropY, :]
    
    print(temp_ch1.shape)
    print(temp_ch2.shape)
    print(temp_ch3.shape)
    print(temp_ch4.shape)

    # Stack channels along a new axis and permute to shape (ch, z, y, x)
    #data_temp = np.stack([temp_ch1, temp_ch2, temp_ch3, temp_ch4], axis=0)
    #data_temp = da.concatenate([temp_ch1[ :, :, :], temp_ch2[ :, :, :], temp_ch3[ :, :, :], temp_ch4[ :, :, :]], axis=0)
    data_temp = da.concatenate([temp_ch1[None, :, :, :], temp_ch2[None, :, :, :], temp_ch3[None, :, :, :], temp_ch4[None, :, :, :]], axis=0)
    
    return data_temp


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
    # color_data_zyx = np.transpose(color_data, (1, 2, 0)) I think the splitter data needs to be transposed like
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
