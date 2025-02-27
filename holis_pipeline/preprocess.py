from holis_pipeline import settings


# def subtract_background(input_location, output_location):
def subtract_background(image_dask_zarray, bg_mask):

    """
    input: data .omehans
           empty frames (.mat) - the same shape as the data (1 file per nuclei+colors fli pair)
    output: flattened .omehans - same shape as input
    """
    # dir_name = '/bil/proj/rf1hillman/2024_07_29_AI7_EH5k_human_finalMarkerCombination_100mm/raw_HiCAMdata/'
    # bg_info = sio.loadmat(os.path.join(f'{input_location}','wholeScanBG_run001_info.mat'))
    bg_nuclei_filename = 'wholeScanBG-run001_HiCAM FLUO_1875-ST-272.fli'
    # bg_colors_filename = 'wholeScanBG-run001_HiCAM FLUO_1875-ST-088.fli'
    bg_nuclei_path = os.path.join(dir_name, bg_nuclei_filename)
    # bg_colors_path = os.path.join(dir_name, bg_colors_filename)
    bg_nuclei = read_data_file(bg_nuclei_path)
    # bg_colors = read_data_file(bg_colors_path)

    # Generate background masks
    bg_nuclei_mask = da.mean(da.asarray(bg_nuclei, dtype=np.float32), axis=2) - 2**10
    # bg_colors_mask = da.mean(da.asarray(bg_colors, dtype=np.float32), axis=2) - 2**10


    # empty_frames_location = settings.EMPTY_FRAMES_LOCATION


    image_dask_zarray_bgSubtracted = image_dask_zarray - bg_nuclei_mask[:, :, np.newaxis]
    
    return image_dask_zarray_bgSubtracted


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


def preprocess_nuclei(location):
    bg_subtracted_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_bg_subtracted")
    try:
        os.makedirs(bg_subtracted_location)
    except:
        pass
    subtract_background(location, bg_subtracted_location)
    laser_corrected_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_laser_corrected")
    try:
        os.makedirs(laser_corrected_location)
    except:
        pass
    laser_correction(bg_subtracted_location, laser_corrected_location)
    preprocessed_location = laser_corrected_location
    return preprocessed_location


def preprocess_colors(location):
    bg_subtracted_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_bg_subtracted")
    try:
        os.makedirs(bg_subtracted_location)
    except:
        pass
    subtract_background(location, bg_subtracted_location)
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
