import os
from glob import glob

import dask.array as da
import numpy as np
import scipy
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
    Takes a 3D image splitter data and outputs a 4D array with shape (ch, z, y, x).
    """
    data_temp = tifffile.imread(os.path.join(input_location, "bg_subtracted.tif"))
    ss = data_temp.shape

    # Define four channel regions by splitting along y and x axes
    # temp_ch1 = data_temp[round(0.5 * ss[0]):ss[0], round(0.5 * ss[1]):ss[1], :]
    # temp_ch2 = data_temp[round(0.5 * ss[0]):ss[0], :round(0.5 * ss[1]), :]
    # temp_ch3 = data_temp[:round(0.5 * ss[0]), round(0.5 * ss[1]):ss[1], :]
    # temp_ch4 = data_temp[:round(0.5 * ss[0]), :round(0.5 * ss[1]), :]

    temp_ch1 = data_temp[:, :ss[1]//2, :ss[2]//2]
    temp_ch2 = data_temp[:, :ss[1]//2, ss[2]//2:]
    temp_ch3 = data_temp[:, ss[1]//2:, :ss[2]//2]
    temp_ch4 = data_temp[:, ss[1]//2:, ss[2]//2:]
    
    print(temp_ch1.shape)
    print(temp_ch2.shape)
    print(temp_ch3.shape)
    print(temp_ch4.shape)

    # Optional offsets for each channel (if needed)
    y = np.zeros(4, dtype=int)
    # y = da.zeros(4, dtype=int)
    z = np.zeros(4, dtype=int)
    # z = da.zeros(4, dtype=int)
    print(y)
    print(z)

    # Crop ranges for each channel
    cropY = 640
    cropZ = 512

    # Apply cropping to each channel
    # temp_ch1 = temp_ch1[z[0]:z[0] + cropZ, y[0]:y[0] + cropY, :]
    # temp_ch2 = temp_ch2[z[1]:z[1] + cropZ, y[1]:y[1] + cropY, :]
    # temp_ch3 = temp_ch3[z[2]:z[2] + cropZ, y[2]:y[2] + cropY, :]
    # temp_ch4 = temp_ch4[z[3]:z[3] + cropZ, y[3]:y[3] + cropY, :]
    
    # print(temp_ch1.shape)
    # print(temp_ch2.shape)
    # print(temp_ch3.shape)
    # print(temp_ch4.shape)

    # Stack channels along a new axis and permute to shape (ch, z, y, x)
    #data_temp = np.stack([temp_ch1, temp_ch2, temp_ch3, temp_ch4], axis=0)
    #data_temp = da.concatenate([temp_ch1[ :, :, :], temp_ch2[ :, :, :], temp_ch3[ :, :, :], temp_ch4[ :, :, :]], axis=0)
    # data_temp = da.concatenate([temp_ch1[None, :, :, :], temp_ch2[None, :, :, :], temp_ch3[None, :, :, :], temp_ch4[None, :, :, :]], axis=0)
    data_temp = np.concatenate([temp_ch1[None, :, :, :], temp_ch2[None, :, :, :], temp_ch3[None, :, :, :], temp_ch4[None, :, :, :]], axis=0)
    print("data_temp shape", data_temp.shape)
    tifffile.imwrite(os.path.join(output_location, "color_split.tif"), data_temp)


def laser_correction(input_location, output_location):  # tbd whether needs to be processed separately
    """
    Input:
        - flattened .omehans
        - laser pattern matrix (y,z) shape
        - absorption matrix (n_fluorophores, n_lasers) - (5x4) - first row   for nuclei
        - mixing (fluorescence) matrix (n_fluorophores, n_channels) - (5x5) - first column for nuclei
    Output: corrected .omehans the same shape as input
    """
    # data = tifffile.imread(os.path.join(input_location, "color_split.tif"))
    data = tifffile.imread(os.path.join(input_location, "bg_subtracted.tif"))
    corrected_data = laser_correction_byInverse(data)
    tifffile.imwrite(os.path.join(output_location, 'laser_corrected.tif'), corrected_data)


def laser_correction_byInverse(m):  # tbd whether needs to be processed separately
    """
    Input:
        - np.array (channels,z,y)
        - laser pattern matrix (y,z) shape
        - absorption matrix (n_fluorophores, n_lasers) - (5x4) - first row   for nuclei
        - mixing (fluorescence) matrix (n_fluorophores, n_channels) - (5x5) - first column for nuclei
    Output: corrected np.array (channels,z,y)
    """
    ## Load necessary matrices - this could be done outside of the function
    ## The notation here is what Malte uses, I change it (slightly) to my notation when I start the computation

    CORRECTION_DATA = scipy.io.loadmat(settings.CORRECTION_DATA)
    LASER_CORRECTION_DATA = scipy.io.loadmat(settings.LASER_CORRECTION_DATA)

    # Laser spatial pattern matrix - nuclear channel
    POWELL_NUC_MASK_FF_norm = CORRECTION_DATA['POWELL_NUC_MASK_FF_norm']
    print('POWELL_NUC_MASK_FF_norm', POWELL_NUC_MASK_FF_norm.shape)

    # Laser spatial pattern matrix - splitter (4 channels)
    POWELL_SPL_MASK_FF_norm = CORRECTION_DATA['POWELL_SPL_MASK_FF_norm']
    print('POWELL_SPL_MASK_FF_norm', POWELL_SPL_MASK_FF_norm.shape)

    # Fluorophore x Channel matrix (This is the fluorescnece matrix)
    Flch = LASER_CORRECTION_DATA['Flch']
    print('Flch', Flch.shape)

    # Normalize along columns - so each entry (i,j) is the percentage of the signal in channel j coming from fluorophore i
    Flch_rel = Flch.copy()
    Flch_rel = Flch_rel / np.sum(Flch_rel, axis=1, keepdims=True)

    # Fluorophore x Laser matrix (This is the absorbtion matrix)
    excitation_efficiency = LASER_CORRECTION_DATA['excitation_efficiency']

    print('excitation_efficiency', excitation_efficiency.shape)

    ## Computation 

    F = Flch_rel
    E = excitation_efficiency.T
    
    num_lasers = E.shape[0]
    print('num_lasers', num_lasers)
    num_channels = F.shape[0]
    print('num_channels', num_channels)

    # Q combines the absorbtion and fluorescence matrices, each will later be multiplied by the corresponding laser pattern, summed and inverted
    A = []

    for i in range(num_lasers):
        l = E[i,:]
        A_temp = np.tile(l, (num_channels, 1))
        A.append(A_temp)
        
    Q = []

    # For channels
    for i in range(num_lasers):
        Q_temp = F * A[i]
        #Q_temp = Q_temp[1:,1:]
        Q.append(Q_temp)

    z_size = 512
    y_size = 640

    # Corrected output array, m is for measurement
    m_corr = np.empty((num_channels, z_size, y_size))

    # Define the laser pattern matrices and resize the nuclear channel to the size of color channels
    S_nuc = POWELL_NUC_MASK_FF_norm # One matrix per laser: (lasers, z, y) = (4,1024,1280)
    S_channels = POWELL_SPL_MASK_FF_norm # One matrix per channel per laser: (channels, lasers, z, y) = (4,4,512,640)

    resize_factors = (1, 1/2, 1/2)  #(1, 1/8, 1/8)
    S_nuc_resized = zoom(S_nuc, resize_factors, order=1)
    
    # Combine S_nuc and S_channels into one array, inthe case it should be (channels, lasers, z, y) = (5,4,512,640)
    S_nuc_expanded = np.expand_dims(S_nuc_resized, axis=0)
    S_resized = np.concatenate((S_nuc_expanded, S_channels), axis=0) 

    # m_channels is the input array, each m[i] should be (512,640) in this function
    m_channels = np.array([m[0], m[1], m[2], m[3], m[4]])

    for z in tqdm(range(z_size)):
        for y in range(y_size):
            S = []
            for i in range(num_lasers):
                #v_temp = S_channels_resized[:,i,z,y]
                v_temp = S_resized[:,i,z,y]
                S_temp = np.diag(v_temp)
                S.append(S_temp)
            G = sum(np.dot(S[i], Q[i]) for i in range(num_lasers))
            # Check if matrix is invertible
            G_det = np.linalg.det(G)
            if G_det != 0:
                c = []
                G_inv = np.linalg.pinv(G)
                c = np.dot(G_inv,m_channels[:,z,y])
                m_corr[:, z, y] = c
            else:
                print("Matrix is not invertible.")   

    return m_corr


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

    return

    color_registered_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_color_registered")
    try:
        os.makedirs(color_registered_location)
    except:
        pass
    color_registration(laser_corrected_location, color_registered_location)
    preprocessed_location = color_registered_location
    return preprocessed_location
