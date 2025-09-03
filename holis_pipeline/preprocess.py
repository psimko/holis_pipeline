import os
import sys
from glob import glob
import logging

import dask.array as da
import numpy as np
import scipy
import tifffile
from skimage import io, img_as_float32, img_as_float, img_as_uint
from skimage.transform import resize
import zarr

from holis_pipeline import settings
from holis_pipeline.preprocessing_functions import read_data_file
from holis_pipeline.read_data import read_fli_as_zarr
from holis_pipeline.utils.zarr_related import read_omehans, write_omehans, write_zarr, write_dask_zarr_compatible_with_napari


log = logging.getLogger(__name__)


def subtract_background(input_location, output_location, bg_file_name):

    """
    input: data .omehans
           empty frames (.mat) - the same shape as the data (1 file per nuclei+colors fli pair)
    output: flattened .omehans - same shape as input
    """
    log.info("Subtracting BG...")
    if os.path.exists(os.path.join(output_location, 'omehans', '0', '0', '0')) and os.path.exists(os.path.join(output_location, "bg_subtracted.tif")):
        log.info("BG already subtracted previously")
        return
    log.info("reading omehans")
    image_dask_zarray = read_omehans(input_location)
    image_np_array = image_dask_zarray.compute()
    # tifffile.imwrite(os.path.join(output_location, "original.tif"), image_np_array.astype('uint16'))
    write_zarr(os.path.join(output_location, 'original_zarr'), image_np_array.astype('uint16'))
    log.info("Reading BG file")
    BG_FOLDER = read_fli_as_zarr(bg_file_name, os.path.join(os.path.dirname(output_location), os.path.basename(bg_file_name).replace('.fli', '')))
    bg_dask_zarray = read_omehans(BG_FOLDER)
    bg_dask_zarray_sample = bg_dask_zarray[15000:18000,:,:]
    bg_array = bg_dask_zarray_sample.compute()
    write_zarr(os.path.join(output_location, 'bg_zarr'), bg_array.astype('uint16'))
    # # Generate background masks
    # bg_mask = da.mean(da.asarray(bg_array, dtype=np.float32), axis=2) - 2**10
    bg_mask_np = np.mean(bg_array.astype('float32'), axis=0).round()
    print("calculated mean")

    # bg_mask_np = bg_mask.compute()

    # image_dask_zarray_bgSubtracted = image_dask_zarray - bg_mask[:, :, np.newaxis]
    # image_np_array_bgSubtracted = image_dask_zarray_bgSubtracted.compute()
    image_np_array_bgSubtracted = image_np_array.astype('float32') - bg_mask_np[np.newaxis, :, :].astype('float32')

    image_np_array_bgSubtracted[image_np_array_bgSubtracted < 0] = 0
    log.info("corrected negative values")
    # tifffile.imwrite(os.path.join(output_location, "bg_subtracted.tif"), image_np_array_bgSubtracted.astype('uint16'))
    try:
        write_omehans(os.path.join(output_location, 'omehans'), image_np_array_bgSubtracted.astype('uint16'))
    except zarr.errors.ContainsArrayError:
        log.info(".omehans array already exists")
    write_zarr(os.path.join(output_location, 'zarr'), image_np_array_bgSubtracted.astype('uint16'))

    # image_np_array_bgSubtracted = image_np_array.astype('float32') - bg_array.astype('float32')
    # image_np_array_bgSubtracted[image_np_array_bgSubtracted < 0] = 0
    # tifffile.imwrite(os.path.join(output_location, "raw_minus_bg.tif"), image_np_array_bgSubtracted.astype('uint16'))
    log.info("Saved BG-subtracted file")


def subtract_background_dask(input_location, output_location, bg_file_name):
    """
    Subtract background from input omehans using mean of background file.
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["NUMEXPR_NUM_THREADS"] = "8"

    import dask
    from dask.diagnostics import ProgressBar

    # Optional: configure Dask to use 8 threads
    dask.config.set(scheduler='threads', num_workers=8)

    # Optional: show progress
    ProgressBar().register()

    log.info("Reading omehans")
    image_dask = read_omehans(input_location)  # Dask array

    log.info("Reading BG file")
    bg_folder = read_fli_as_zarr(bg_file_name, os.path.join(os.path.dirname(output_location),
                                                             os.path.basename(bg_file_name).replace('.fli', '')))
    bg_dask = read_omehans(bg_folder)  # Dask array

    log.info("Calculating mean background")
    bg_mask = bg_dask.astype('float32').mean(axis=2).round()  # (H, W)

    # Expand dims for broadcasting: (H, W) → (H, W, 1)
    bg_mask = bg_mask[:, :, None]

    log.info("Subtracting background lazily")
    # Lazy subtraction, clamp negatives to 0
    subtracted = image_dask.astype('float32') - bg_mask
    subtracted = da.maximum(subtracted, 0).astype('uint16')  # Apply clamp + convert to uint16

    log.info("Writing intermediate and final results")
    # write_zarr(os.path.join(output_location, 'original_zarr'), image_dask.astype('uint16'))
    write_dask_zarr_compatible_with_napari(os.path.join(output_location, 'original_zarr'), image_dask.astype('uint16'))
    # image_dask.astype('uint16').to_zarr(os.path.join(output_location, 'original_zarr'), overwrite=False)
    # write_zarr(os.path.join(output_location, 'bg_zarr'), bg_dask.astype('uint16'))
    write_dask_zarr_compatible_with_napari(os.path.join(output_location, 'bg_zarr'), bg_dask.astype('uint16'))
    # bg_dask.astype('uint16').to_zarr(os.path.join(output_location, 'bg_zarr'), overwrite=False)
    # write_zarr(os.path.join(output_location, 'zarr'), subtracted)
    write_dask_zarr_compatible_with_napari(os.path.join(output_location, 'zarr'), subtracted)
    # subtracted.to_zarr(os.path.join(output_location, 'zarr'), overwrite=False)

    # try:
    #     write_omehans(os.path.join(output_location, 'omehans'), subtracted)
    # except zarr.errors.ContainsArrayError:
    #     log.info(".omehans array already exists")


def split_color_channels(input_location, output_location):
    """
    Takes a 3D image splitter data and outputs a 4D array with shape (ch, z, y, x).
    """
    print("Splitting color channels...")
    if os.path.exists(os.path.join(output_location, 'zarr')) and os.path.exists(os.path.join(output_location, "color_split.tif")):
        print("Colors already split previously")
        return
    print("Reading data...")
    # data_temp = tifffile.imread(os.path.join(input_location, "bg_subtracted.tif"))
    data_temp_zarray = read_omehans(os.path.join(input_location, "omehans"))
    data_temp = data_temp_zarray.compute()

    data_temp = np.transpose(data_temp, (1, 2, 0)) 

    print("Calculating split...")
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

    # Optional offsets for each channel (if needed)
    y = np.zeros(4, dtype=int)
    # y = da.zeros(4, dtype=int)
    z = np.zeros(4, dtype=int)
    # z = da.zeros(4, dtype=int)

    # # Crop ranges for each channel
    # cropY = 640
    # cropZ = 512

    # Apply cropping to each channel
    # temp_ch1 = temp_ch1[z[0]:z[0] + cropZ, y[0]:y[0] + cropY, :]
    # temp_ch2 = temp_ch2[z[1]:z[1] + cropZ, y[1]:y[1] + cropY, :]
    # temp_ch3 = temp_ch3[z[2]:z[2] + cropZ, y[2]:y[2] + cropY, :]
    # temp_ch4 = temp_ch4[z[3]:z[3] + cropZ, y[3]:y[3] + cropY, :]
    
    # Stack channels along a new axis and permute to shape (ch, z, y, x)
    #data_temp = np.stack([temp_ch1, temp_ch2, temp_ch3, temp_ch4], axis=0)
    #data_temp = da.concatenate([temp_ch1[ :, :, :], temp_ch2[ :, :, :], temp_ch3[ :, :, :], temp_ch4[ :, :, :]], axis=0)
    # data_temp = da.concatenate([temp_ch1[None, :, :, :], temp_ch2[None, :, :, :], temp_ch3[None, :, :, :], temp_ch4[None, :, :, :]], axis=0)
    data_temp = np.concatenate([temp_ch1[None, :, :, :], temp_ch2[None, :, :, :], temp_ch3[None, :, :, :], temp_ch4[None, :, :, :]], axis=0)

    print("Saving data...")
    # tifffile.imwrite(os.path.join(output_location, "color_split.tif"), data_temp)
    try:
        write_omehans(os.path.join(output_location, "omehans"), data_temp.astype('uint16'))
    except zarr.errors.ContainsArrayError:
        print(".omehans array already exists")
    write_zarr(os.path.join(output_location, "zarr"), data_temp.astype('uint16'))


def laser_correction_nuclei(input_location, output_location):
    """
    Input:
        - flattened .omehans
        - laser pattern matrix (y,z) shape
        - absorption matrix (n_fluorophores, n_lasers) - (5x4) - first row   for nuclei
        - mixing (fluorescence) matrix (n_fluorophores, n_channels) - (5x5) - first column for nuclei
    Output: corrected .omehans the same shape as input
    """
    #data = tifffile.imread(os.path.join(input_location, "bg_subtracted.tif"))

    print("Laser pattern (nuclei channel) correction...")
    if os.path.exists(os.path.join(output_location, 'omehans', '0', '0', '0')) and os.path.exists(os.path.join(output_location, "laser_corrected.tif")):
        print("Laser pattern corrected (nuclei channel) previously")
        return
    print("reading omehans")

    image_dask_zarray = read_omehans(os.path.join(input_location, "omehans"))
    image_np_array = image_dask_zarray.compute()
    CORRECTION_MATRICES = scipy.io.loadmat(settings.CORRECTION_DATA)
    SIMULATION_MATRICES = scipy.io.loadmat(settings.LASER_CORRECTION_DATA)
    excitation_efficiency = SIMULATION_MATRICES['excitation_efficiency']

    # Laser power normalization
    #laser_power_at_sample = np.array([0.022, 0.115, 0.263, 0.285])
    laser_power_at_sample = np.array([0.10, 0.44, 0.40, 1.08]) # First hemibrain slab 6/7
    laser_power = laser_power_at_sample / np.max(laser_power_at_sample)  # Normalize laser powers

    Flch = SIMULATION_MATRICES['Flch']
    Flch_rel = Flch.copy()
    Flch_rel = Flch_rel / np.sum(Flch_rel, axis=1, keepdims=True)
    POWELL_NUC_MASK_FF_norm = CORRECTION_MATRICES['POWELL_NUC_MASK_FF_norm']

    # 1. Multiply the laser pattern by laser intensity
    laser_correction_Nuc = laser_power[:, np.newaxis, np.newaxis] * POWELL_NUC_MASK_FF_norm

    # 2. Reshape correction matrix to a 2d matrix: (lasers) x (z x y)
    # and multiply (dot product) by the first row (that's the nuclei fluor) of the excitation matrix
    # you're computing how much each laser's pattern contributes to the nuclei fluor
    # output is a 1 x (z x y) matrix
    sLCN = laser_correction_Nuc.shape
    laser_correction_ch_Nuc = np.dot(excitation_efficiency[0, :],
                                     laser_correction_Nuc.reshape((sLCN[0], sLCN[1] * sLCN[2])))

    # 3. Multiply (dot product) by the first column of the fluorescence matrix (that's the distribution of fluors in the nuclear channel)
    # you're computing how much of the laser pattern comes from which fluor (see comment)
    # output is a 5 x (z x y) 2d matrix (that's why we need the np.newaxis)
    laser_correction_ch_Nuc = np.dot(Flch_rel[:, 0][:, np.newaxis], laser_correction_ch_Nuc[np.newaxis, :])

    # 4. Reshape back to a 3d matrix fluors x z x y and sum across the fluors
    laser_correction_ch_Nuc = laser_correction_ch_Nuc.reshape(5, sLCN[1], sLCN[2])
    laser_correction_ch_Nuc = np.sum(laser_correction_ch_Nuc, axis=0)  # Squeeze sum over first axis

    # 5. Divide the nuclear channel image by the correction matrix (element-wise)
    image_np_array = image_np_array.astype('float32')

    """     print(type(image_np_array))
    print(image_np_array.dtype)
    print(image_np_array.shape)
    print(image_np_array)  """

    laser_correction_ch_Nuc = laser_correction_ch_Nuc.astype('float32')

    corrected_data = image_np_array / (laser_correction_ch_Nuc + 1)
    """     min_val = corrected_data.min()
    max_val = corrected_data.max()
    corrected_data = (corrected_data - min_val) / (max_val - min_val)
    corrected_data = corrected_data * 65535 """

    image_np_array_laserCorrected = corrected_data.astype('float32')

    # Comment: so if we do step 3 I think in step 4 we shouldn't add across fluors but divide in step 5 by laser_correction_ch_Nuc[0]
    # Otherwise, if we are going to sum anyway we don't need step (4) and can just divide by laser_correction_ch_Nuc, because otherwise it seems like we are redistributing a value and then summing it back.

    #tifffile.imwrite(os.path.join(output_location, 'laser_corrected.tif'), np.round(corrected_data).astype('uint16'))

    try:
        write_omehans(os.path.join(output_location, 'omehans'), image_np_array_laserCorrected.astype('uint16'))
    except zarr.errors.ContainsArrayError:
        print(".omehans array already exists")
    try:
        write_zarr(os.path.join(output_location, 'zarr'), image_np_array_laserCorrected.astype('uint16'))
    except zarr.errors.ContainsArrayError:
        print(".zarr array already exists")
    print("Saved laser-pattern-corrected file")



def laser_correction_colors(input_location, output_location):
    #data = tifffile.imread(os.path.join(input_location, "color_split.tif"))
    print("Laser pattern (color channels) correction...")
    if os.path.exists(os.path.join(output_location, 'omehans', '0', '0', '0')) and os.path.exists(os.path.join(output_location, "laser_corrected.tif")):
        print("Laser pattern corrected (color channels) previously")
        return
    print("reading omehans")
    #bg_file_name = glob(os.path.join(os.path.dirname(spool_file), f"{settings.EMPTY_FRAMES_FILE_NAME_FORMAT}272.fli"))[0]
    image_dask_zarray = read_omehans(os.path.join(input_location, "omehans"))
    color_data_split = image_dask_zarray.compute()
    CORRECTION_MATRICES = scipy.io.loadmat(settings.CORRECTION_DATA)
    SIMULATION_MATRICES = scipy.io.loadmat(settings.LASER_CORRECTION_DATA)
    excitation_efficiency = SIMULATION_MATRICES['excitation_efficiency']

    # Laser power normalization
    #laser_power_at_sample = np.array([0.022, 0.115, 0.263, 0.285])
    laser_power_at_sample = np.array([0.10, 0.44, 0.40, 1.08]) #First hemibrain
    laser_power = laser_power_at_sample / np.max(laser_power_at_sample)  # Normalize laser powers

    POWELL_SPL_MASK_FF_norm = CORRECTION_MATRICES['POWELL_SPL_MASK_FF_norm']
    Flch = SIMULATION_MATRICES['Flch']
    Flch_rel = Flch.copy()
    Flch_rel = Flch_rel / np.sum(Flch_rel, axis=1, keepdims=True)

    # 1. Multiply the laser pattern by laser intensity
    laser_correction_SP = laser_power[:, np.newaxis, np.newaxis, np.newaxis] * POWELL_SPL_MASK_FF_norm
    sLCSp = laser_correction_SP.shape

    # 2. Reshape correction matrix to a 2d matrix: (lasers) x (channels x z x y) a
    # and matrix multiply by the the excitation matrix
    # you're computing how much each laser's pattern contributes to each of the color fluors
    # output is a 5 x (ch x z x y) matrix
    laser_correction_ch_SP = np.dot(
        excitation_efficiency,
        laser_correction_SP.reshape(sLCSp[0], sLCSp[1] * sLCSp[2] * sLCSp[3])
    )  # Equivalent to excitation_efficiency(:,:) * reshape(Laser_correction_SP, [sLCSp(1), sLCSp(2)*sLCSp(3)*sLCSp(4)])

    # 3. Matrix multiply by the fluorescence matrix(except the first column)
    # you're computing how much of the laser pattern comes from which fluor (see comment)
    # output is a 4 x (ch x z x y) 2d matrix (that's why we need the np.newaxis)
    laser_correction_ch_SP = np.dot(Flch_rel[:, 1:].T, laser_correction_ch_SP)  # Flch is fl x ch

    # 4. Reshape back to a 3d matrix fluors x channels x z x y and sum across the fluors
    laser_correction_ch_SP = laser_correction_ch_SP.reshape(4, sLCSp[1], sLCSp[2], sLCSp[3])
    laser_correction_ch_SP = np.sum(laser_correction_ch_SP, axis=0)

    # 5. Normalize the result and divide elemnt-wise, resulting shape should be (ch x z x y)
    max_values = np.max(np.max(laser_correction_ch_SP[:, 100:-100, 100:-100], axis=1), axis=1)
    laser_correction_ch_SP = laser_correction_ch_SP / max_values[:, np.newaxis, np.newaxis]

    color_data_split = color_data_split.astype('float32')

    laser_correction_ch_SP = laser_correction_ch_SP.astype('float32')

    ch0 = color_data_split[0,:, :, :]
    ch1 = color_data_split[1,:, :, :]
    ch2 = color_data_split[2,:, :, :]
    ch3 = color_data_split[3,:, :, :]

    color_stack = [ch0, ch1, ch2 ,ch3]

    laser_pattern_colors = laser_correction_ch_SP[:, np.newaxis, :, :] + 1

    ch0_pattern = laser_pattern_colors[0, :, :]
    ch1_pattern = laser_pattern_colors[1, :, :]
    ch2_pattern = laser_pattern_colors[2, :, :]
    ch3_pattern = laser_pattern_colors[3, :, :]

    pattern_stack = [ch0_pattern, ch1_pattern, ch2_pattern, ch3_pattern]

    for i in range(len(color_stack)):
        corrected_data = color_stack[i] / pattern_stack[i]

        """         min_val = corrected_data.min()
        max_val = corrected_data.max()
        corrected_data = (corrected_data - min_val) / (max_val - min_val)
        corrected_data = corrected_data * 65535 """

        image_np_array_laserCorrected = corrected_data.astype('float32')

        #tifffile.imwrite(os.path.join(output_location, 'laser_corrected.tif'), np.round(corrected_data).astype('uint16'))
        try:
            write_omehans(os.path.join(output_location, f'ch{i}_omehans'), image_np_array_laserCorrected.astype('uint16'))
        except zarr.errors.ContainsArrayError:
            print(".omehans array already exists")
        try:
            write_zarr(os.path.join(output_location, f'ch{i}_zarr'), image_np_array_laserCorrected.astype('uint16'))
        except zarr.errors.ContainsArrayError:
            print(".zarr array already exists")
        print("Saved laser-pattern-corrected file")


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
    input:
        - 4-channel corrected colors .omehans
        - 1-channel corrected nuclei .omehans
    output:
        - 4-channel corrected colors registered to nuclei space
        - 4 affine matrices (for each color)
    """
    print("Registering...")

    from skimage import exposure
    from skimage.exposure import match_histograms

    def equalize(img):
        img = (img - img.min()) / (img.max() - img.min())
        img = exposure.equalize_adapthist(img)
        return img

    def match(moving_image, fixed_image):
        matched_img = match_histograms(moving_image, fixed_image)
        return matched_img

    def sitk_align_translation(fixed, moving, output_offsets=False):
        '''
        Input:
            fixed: numpy array (same shape as moving)
            moving: numpy array (same shape as fixed)

        Output:
            If output_offsets == False (default), an aligned image is returned
            If output_offsets == True, a tuple of pixel offsets is returned

            Images are returned in the same dtype as input
        '''
        import SimpleITK as sitk

        dtype = moving.dtype
        # Convert numpy arrays to sitk images
        if fixed.dtype != float:
            fixed = img_as_float32(fixed)
        if moving.dtype != float:
            moving = img_as_float32(moving)
        fixed = sitk.GetImageFromArray(fixed)
        moving = sitk.GetImageFromArray(moving)
        fixed.SetOrigin((0, 0))
        moving.SetOrigin((0, 0))

        # Calculate alignment
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation()
        R.SetOptimizerAsRegularStepGradientDescent(1.0, 0.01, 200)
        R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
        R.SetInterpolator(sitk.sitkLinear)

        # Pyramidal registration
        R.SetShrinkFactorsPerLevel([6, 2, 1])
        R.SetSmoothingSigmasPerLevel([6, 2, 1])

        outTx = R.Execute(fixed, moving)
        # return outTx
        if output_offsets:
            # Return revered tuple of pixel offsets (sitk and numpy axes are reversed)
            # Returned axes are in order (y,x)
            offsets = outTx.GetParameters()[::-1]
            # invert offsets
            return tuple([-x for x in offsets])

        # Produce aligned image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        R.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(outTx)
        out = resampler.Execute(moving)

        # Return numpy array of aligned image
        out = sitk.GetArrayFromImage(out)
        if dtype == out.dtype:
            return out
        if dtype == np.dtype('uint16'):
            return img_as_uint(out)
        if dtype == np.dtype('float32'):
            return img_as_float32(out)
        if dtype == float:
            return img_as_float(out)

    def calculate_channels_shift(multi_channel_z_stack, reference_channel=None):
        from skimage.registration import phase_cross_correlation
        from skimage.metrics import normalized_mutual_information

        if reference_channel is None:
            reference_channel = 0

        shift_dict = {
            0: [],
            1: [],
            2: [],
            3: []
        }
        nmi_scores = []
        moving_array = multi_channel_z_stack[2]  # channel 2 vs channel 0
        ref_array = multi_channel_z_stack[reference_channel]
        for z_idx in range(moving_array.shape[0]):
            print("Calculating mutual information", z_idx)
            reference = ref_array[z_idx].copy()
            moving = moving_array[z_idx].copy()
            nmi = normalized_mutual_information(reference, moving)
            nmi_scores.append(nmi)
        nmi_scores = np.array(nmi_scores)
        threshold = np.percentile(nmi_scores, 90)  # top 10% highest NMI
        top_10_percent_indices = np.where(nmi_scores >= threshold)[0]

        for ch_idx in range(multi_channel_z_stack.shape[0]):
            if ch_idx != reference_channel:
                moving_array = multi_channel_z_stack[ch_idx].copy()
                for z_idx in list(top_10_percent_indices):
                # for z_idx in range(moving_array.shape[0]):
                    print(f'Getting Reference and Moving images')
                    reference = ref_array[z_idx].copy()
                    moving = moving_array[z_idx].copy()
                    reference = equalize(reference)
                    moving = equalize(moving)
                    moving = match(moving, reference)
                    print(f'Aligning Image {z_idx} of {multi_channel_z_stack.shape[1]}')
                    shift, error, phasediff = phase_cross_correlation(reference,moving)
                    # shift = sitk_align_translation(reference, moving, output_offsets=True)
                    shift_dict[ch_idx].append(shift)

        return shift_dict

    def calculate_channel_shifts(data_array, anchor_channel=0):
        '''
        This function takes a 4 channels image and aligns the channels relative to one of the 4 channel (anchor_channel)
        then calculates the translational shift required to overlay them.

        This version of the function uses a geometric mean to calculate the shifts
        '''
        shift_dict = calculate_channels_shift(data_array, reference_channel=anchor_channel)

        shift_medians = {
            0: (0, 0),
            1: None,
            2: None,
            3: None
        }

        for channel in range(1, 4):
            shift_medians[channel] = int(round(np.median(np.array(shift_dict[channel])[:, 0]))), int(round(np.median(np.array(shift_dict[channel])[:, 1])))

        return shift_medians

    print("Reading data...")
    # data_array = tifffile.imread(os.path.join(input_location, "color_split.tif"))
    data_zarray = read_omehans(os.path.join(input_location, "omehans"))
    data_array = data_zarray.compute()
    print("Calculating alignment...")
    shifts = calculate_channel_shifts(data_array)
    print("Shifts", shifts)
    shifted_array = np.zeros_like(data_array)
    shifted_array[0, :, :, :] = data_array[0, :, :, :].copy()
    for channel in range(1, 4):
        channel_data = data_array[channel, :, :, :].copy()
        channel_shifts = shifts[channel]
        channel_data[:] = np.roll(channel_data, channel_shifts[0], axis=1)
        channel_data[:] = np.roll(channel_data, channel_shifts[1], axis=2)
        shifted_array[channel, :, :, :] = channel_data.copy()
    tifffile.imwrite(os.path.join(output_location, "color_registered.tif"), shifted_array.astype('uint16'))
    write_omehans(os.path.join(output_location, "zarr"), shifted_array.astype('uint16'))


def nuclei_color_registration(input_nuclei_location, input_color_location, output_location):
    """
    input:
        - 4-channel corrected colors .omehans
        - 1-channel corrected nuclei .omehans
    output:
        - 4-channel corrected colors registered to nuclei space
        - 4 affine matrices (for each color)
    """
    print("Registering...")

    from skimage import exposure

    def equalize(img):
        img = (img - img.min()) / (img.max() - img.min())
        img = exposure.equalize_adapthist(img)
        return img

    def match(moving_image, fixed_image):
        matched_img = exposure.match_histograms(moving_image, fixed_image)
        return matched_img

    def calculate_channels_shift(multi_channel_z_stack, reference_channel=None):
        print('multi_channel_z_stack', multi_channel_z_stack.shape)
        from skimage.registration import phase_cross_correlation
        from skimage.metrics import normalized_mutual_information

        if reference_channel is None:
            reference_channel = 0

        shift_dict = {
            0: [],
            1: [],
            2: [],
            3: [],
            4: []
        }
        nmi_scores = []
        moving_array = multi_channel_z_stack[4]  # channel 4 vs channel 0
        ref_array = multi_channel_z_stack[reference_channel]
        for z_idx in range(moving_array.shape[0]):
            print("Calculating mutual information", z_idx)
            reference = ref_array[z_idx].copy()
            moving = moving_array[z_idx].copy()
            nmi = normalized_mutual_information(reference, moving)
            nmi_scores.append(nmi)
        nmi_scores = np.array(nmi_scores)
        threshold = np.percentile(nmi_scores, 90)  # top 10% highest NMI
        top_10_percent_indices = np.where(nmi_scores >= threshold)[0]

        for ch_idx in range(multi_channel_z_stack.shape[0]):
            if ch_idx != reference_channel:
                moving_array = multi_channel_z_stack[ch_idx].copy()
                for z_idx in list(top_10_percent_indices):
                # for z_idx in range(moving_array.shape[0]):
                    print(f'Getting Reference and Moving images')
                    reference = ref_array[z_idx].copy()
                    moving = moving_array[z_idx].copy()
                    reference = equalize(reference)
                    moving = equalize(moving)
                    moving = match(moving, reference)
                    print(f'Aligning Image {z_idx} of {multi_channel_z_stack.shape[1]}')
                    shift, error, phasediff = phase_cross_correlation(reference,moving)
                    # shift = sitk_align_translation(reference, moving, output_offsets=True)
                    shift_dict[ch_idx].append(shift)

        return shift_dict

    def calculate_channel_shifts(data_array, anchor_channel=0):
        '''
        This function takes a 4 channels image and aligns the channels relative to one of the 4 channel (anchor_channel)
        then calculates the translational shift required to overlay them.

        This version of the function uses a geometric mean to calculate the shifts
        '''
        print("Data array", data_array.shape[0])
        shift_dict = calculate_channels_shift(data_array, reference_channel=anchor_channel)

        shift_medians = {
            0: (0, 0),
            1: None,
            2: None,
            3: None,
            4: None
        }

        for channel in range(1, data_array.shape[0]):
            shift_medians[channel] = int(round(np.median(np.array(shift_dict[channel])[:, 0]))), int(round(np.median(np.array(shift_dict[channel])[:, 1])))

        return shift_medians

    if not os.path.exists(output_location):
        os.makedirs(output_location)

    import dask
    import numpy as np
    import dask.array as da
    from skimage.transform import resize
    from skimage.util import img_as_uint
    from dask import delayed, compute

    print("Reading data...")
    # data_array = tifffile.imread(os.path.join(input_location, "color_split.tif"))
    print("Reading nuclei")
    nuclei_data_zarray = read_omehans(os.path.join(input_nuclei_location, "omehans"))
    nuclei_data_array = nuclei_data_zarray.compute()
    print("Reading colors")
    color_data_zarray = read_omehans(os.path.join(input_color_location, "omehans"))
    color_data_array = color_data_zarray.compute()
    print("Resizing nuclei")
    # resized_nuclei_data_array = resize(nuclei_data_array, (1, color_data_array.shape[1], color_data_array.shape[2], color_data_array.shape[3]), anti_aliasing=True)
    # resized_nuclei_data_array = np.empty(color_data_array.shape[1:], color_data_array.dtype)
    # print("resized_nuclei_data_array", resized_nuclei_data_array.shape)

    # for x in range(nuclei_data_array.shape[0]):
    #     print("resizing", x)
    #     frame = nuclei_data_array[x, :, :]
    #     resized_frame = resize(frame, (color_data_array.shape[2], color_data_array.shape[3]), anti_aliasing=True)
    #     resized_nuclei_data_array[x, :, :] = img_as_uint(resized_frame)

    # Assume nuclei_data_array and color_data_array are already loaded NumPy arrays
    shape_y, shape_x = color_data_array.shape[2], color_data_array.shape[3]

    @delayed
    def process_frame(frame):
        resized = resize(frame, (shape_y, shape_x), anti_aliasing=True)
        return img_as_uint(resized)

    # Apply the function to each frame in parallel
    tasks = [process_frame(nuclei_data_array[x, :, :]) for x in range(nuclei_data_array.shape[0])]

    # Compute all tasks and stack the result
    resized_frames = compute(*tasks)
    resized_nuclei_data_array = np.stack(resized_frames, axis=0)
    print("resized_nuclei_data_array", resized_nuclei_data_array.shape)

    print("CHanging nuclei dimensions")
    resized_nuclei_data_array = resized_nuclei_data_array[:, ::-1, ::-1]
    tifffile.imwrite(os.path.join(output_location, "ch0.tif"), resized_nuclei_data_array.astype('uint16'))
    tifffile.imwrite(os.path.join(output_location, "ch1.tif"), color_data_array[0,:,:,:].astype('uint16'))
    tifffile.imwrite(os.path.join(output_location, "ch2.tif"), color_data_array[1,:,:,:].astype('uint16'))
    tifffile.imwrite(os.path.join(output_location, "ch3.tif"), color_data_array[2,:,:,:].astype('uint16'))
    tifffile.imwrite(os.path.join(output_location, "ch4.tif"), color_data_array[3,:,:,:].astype('uint16'))
    resized_nuclei_data_array = resized_nuclei_data_array.reshape((1, resized_nuclei_data_array.shape[0], resized_nuclei_data_array.shape[1], resized_nuclei_data_array.shape[2]))
    print("resized_nuclei_data_array", resized_nuclei_data_array.shape)
    print("Concatenating nuclei & colors")
    data_array = np.concatenate([resized_nuclei_data_array, color_data_array], axis=0)
    print("data_array", data_array.shape)
    print("Saving tiff")
    tifffile.imwrite(os.path.join(output_location, "nuclei_plus_colors.tif"), data_array.astype('uint16'))

    print("Calculating alignment...")
    shifts = calculate_channel_shifts(data_array)
    print("Shifts", shifts)
    shifted_array = np.zeros_like(data_array)
    shifted_array[0, :, :, :] = data_array[0, :, :, :].copy()
    for channel in range(1, data_array.shape[0]):
        channel_data = data_array[channel, :, :, :].copy()
        channel_shifts = shifts[channel]
        channel_data[:] = np.roll(channel_data, channel_shifts[0], axis=1)
        channel_data[:] = np.roll(channel_data, channel_shifts[1], axis=2)
        shifted_array[channel, :, :, :] = channel_data.copy()
    # tifffile.imwrite(os.path.join(output_location, "nuclei_color_registered.tif"), shifted_array.astype('uint16'))
    try:
        write_omehans(os.path.join(output_location, "omehans"), shifted_array.astype('uint16'))
    except zarr.errors.ContainsArrayError:
        print(".omehans array already exists")
    write_zarr(os.path.join(output_location, "zarr"), shifted_array.astype('uint16'))


def preprocess_nuclei(spool_file, location):
    # BG subtraction
    bg_subtracted_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_bg_subtracted")
    try:
        os.makedirs(bg_subtracted_location)
    except:
        pass
    bg_file_name = glob(os.path.join(os.path.dirname(spool_file), f"{settings.EMPTY_FRAMES_FILE_NAME_FORMAT}272.fli"))[0]

    subtract_background(location, bg_subtracted_location, bg_file_name)

    # Laser correction
    laser_corrected_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_laser_corrected")
    try:
        os.makedirs(laser_corrected_location)
    except:
        pass
    laser_correction_nuclei(bg_subtracted_location, laser_corrected_location)
    preprocessed_location = laser_corrected_location
    return preprocessed_location


def preprocess_colors(spool_file, location):
    # BG subtraction
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

    # Laser correction
    #color_data_zyx = np.transpose(color_data, (1, 2, 0)) 
    split_color_channels(bg_subtracted_location, color_split_location)


    laser_corrected_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_laser_corrected")
    
    try:
        os.makedirs(laser_corrected_location)
    except:
        pass
    laser_correction_colors(color_split_location, laser_corrected_location)

    # color_registered_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_color_registered")
    # try:
    #     os.makedirs(color_registered_location)
    # except:
    #     pass
    # color_registration(color_split_location, color_registered_location)  # register to the first color channel

    preprocessed_location = laser_corrected_location

    return preprocessed_location


###############################################
## Unmixing
###############################################

###############################################
# NNLS - no GPU

def unmix_data():

    # c (stands for corrected) is a list (length of channels) of images, e.g. c=[c_nuc, c_ch1, ...]
    ss = c.shape
    c_reshaped = c.reshape(ss[0], ss[1] * ss[2])

    # Preallocate output array
    spectral_data_unmixed = np.zeros((Flch_rel.shape[1], ss[1] * ss[2]))

    # Solve NNLS for each pixel
    for i in range(c_reshaped.shape[1]):  # Iterate over flattened pixels
        spectral_data_unmixed[:, i], _ = nnls(Flch_rel, c_reshaped[:, i])  # NNLS ensures non-negativity
        # spectral_data_unmixed[:, i] *= 2**10
    # Reshape back to original dimensions
    spectral_data_unmixed = spectral_data_unmixed.reshape(Flch_rel.shape[1], ss[1], ss[2])

# ###############################################
# # NNLS - with GPU

#
# import torch
# import pickle
# import numpy as np
# from tqdm import tqdm
# import scipy.io as sio
# from scipy.io import loadmat
# from numpy.linalg import inv

# # ---------- Load input ----------
# # c should be of shape (ch, z, y, x)
# c = np.load("array_to_unmix.npy", allow_pickle=True)

# # Fluorophore x Channel matrix normalization
# Flch = laser_correction_data['Flch']
# Flch_rel = Flch.copy()

# # Normalize along columns - so each entry (i,j) is the percentage of the signal in channel j coming from fluorophore i
# Flch_rel = Flch_rel / np.sum(Flch_rel, axis=1, keepdims=True)
# Flch_rel = np.array(Flch_rel)

# assert Flch_rel.shape[1] == c.shape[0], "Channel count mismatch!"

# # ---------- Setup ----------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Unmixing running on {device}")

# A = torch.tensor(Flch_rel, dtype=torch.float32, device=device)  # shape: (C, S)

# # bias_column = torch.ones((Flch_rel.shape[0], 1), dtype=torch.float32)
# # A_augmented = np.concatenate([Flch_rel, bias_column.numpy()], axis=1)
# # A = torch.tensor(A_augmented, dtype=torch.float32, device=device)

# AtA = A.T @ A
# AtA_inv = torch.linalg.pinv(AtA)
# At = A.T

# # Preprocess data
# C_np = c.reshape(c.shape[0], -1).T  # shape: (pixels, C)
# C = torch.tensor(C_np, dtype=torch.float32, device=device)  # shape: (N, C)

#
# # ---------- Load input ----------
# # c should be of shape (ch, z, y, x)
# c = np.load("array_to_unmix.npy", allow_pickle=True)
#
# # Fluorophore x Channel matrix normalization
# Flch = laser_correction_data['Flch']
# Flch_rel = Flch.copy()
#
# # Normalize along columns - so each entry (i,j) is the percentage of the signal in channel j coming from fluorophore i
# Flch_rel = Flch_rel / np.sum(Flch_rel, axis=1, keepdims=True)
# Flch_rel = np.array(Flch_rel)
#
# assert Flch_rel.shape[1] == c.shape[0], "Channel count mismatch!"
#
# # ---------- Setup ----------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Unmixing running on {device}")
#
# A = torch.tensor(Flch_rel, dtype=torch.float32, device=device)  # shape: (C, S)
#
# # bias_column = torch.ones((Flch_rel.shape[0], 1), dtype=torch.float32)
# # A_augmented = np.concatenate([Flch_rel, bias_column.numpy()], axis=1)
# # A = torch.tensor(A_augmented, dtype=torch.float32, device=device)
#
# AtA = A.T @ A
# AtA_inv = torch.linalg.pinv(AtA)
# At = A.T
#
# # Preprocess data
# C_np = c.reshape(c.shape[0], -1).T  # shape: (pixels, C)
# C = torch.tensor(C_np, dtype=torch.float32, device=device)  # shape: (N, C)
#

# # ---------- Batched NNLS via projection ----------
# def nnls_torch(A, C, max_iter=500, lr=1e-2):
#     """
#     Solve NNLS: minimize ||Ax - b||^2 s.t. x >= 0 using projected gradient descent
#     A: (C, F), C: (N, C) where N is the number of pixels, C is the number of channels and F is the number of fluors
#     Returns: X: (N, F)
#     """
#     N, C_dim = C.shape
#     F = A.shape[1]
#     X = torch.zeros((N, F), device=device, dtype=torch.float32, requires_grad=True)
#     optimizer = torch.optim.SGD([X], lr=lr)



#
#     for _ in range(max_iter):
#         optimizer.zero_grad()
#         pred = C @ A.T - X @ AtA.T  # Equivalent to A @ X.T - C.T
#         loss = torch.sum(pred**2)
#         loss.backward()
#         optimizer.step()
#         with torch.no_grad():
#             X.clamp_(min=0)  # enforce non-negativity
#     return X.detach()

# print("Running GPU NNLS...")
# X_unmixed = nnls_torch(A, C, max_iter=100, lr=1e-2)  # shape: (pixels, sources)

# # Reshape and save
# S = Flch_rel.shape[1] # add 1 to shape if you're using bias
# X_unmixed_np = X_unmixed.cpu().numpy().T.reshape(S, *c.shape[1:])

# #return X_unmixed_np

#
# print("Running GPU NNLS...")
# X_unmixed = nnls_torch(A, C, max_iter=100, lr=1e-2)  # shape: (pixels, sources)
#
# # Reshape and save
# S = Flch_rel.shape[1] # add 1 to shape if you're using bias
# X_unmixed_np = X_unmixed.cpu().numpy().T.reshape(S, *c.shape[1:])
#
# return X_unmixed_np

# #with open("spectral_data_unmixed_test.pkl", "wb") as f:
# #    pickle.dump(X_unmixed_np, f)

# #print("Saved: spectral_data_unmixed_test.pkl")
