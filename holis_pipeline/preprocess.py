
dir_name = '/bil/proj/rf1hillman/2024_07_29_AI7_EH5k_human_finalMarkerCombination_100mm/raw_HiCAMdata/'
bg_info = sio.loadmat(os.path.join(f'{dir_name }','wholeScanBG_run001_info.mat'))
bg_nuclei_filename = 'wholeScanBG-run001_HiCAM FLUO_1875-ST-272.fli'
bg_colors_filename = 'wholeScanBG-run001_HiCAM FLUO_1875-ST-088.fli'
bg_nuclei_path = os.path.join(dir_name, bg_nuclei_filename)
bg_colors_path = os.path.join(dir_name, bg_colors_filename)
bg_nuclei = read_data_file(bg_nuclei_path)
bg_colors = read_data_file(bg_colors_path)

# Generate background masks
bg_nuclei_mask = da.mean(da.asarray(bg_nuclei, dtype=np.float32), axis=2) - 2**10
bg_colors_mask = da.mean(da.asarray(bg_colors, dtype=np.float32), axis=2) - 2**10

def subtract_background(image_dask_zarray, bg_mask):
    """
    input: data .omehans
           empty frames (.mat) - the same shape as the data
    output: flattened .omehans - same shape as input
    """

    image_dask_zarray_bgSubtracted = image_dask_zarray - bg_mask[:, :, np.newaxis]
    
    return image_dask_zarray_bgSubtracted


def split_color_channels():
    """
    input: flattened .omehans (only colors)
    output: reshaped 4-channel .omehans
    """
    pass


def laser_correction():  # tbd whether needs to be processed separately
    """
    Input:
        - flattened .omehans
        - laser pattern matrix (y,z) shape
        - absorption matrix (n_fluorophores, n_lasers) - (5x4) - first row   for nuclei
        - mixing (fluorescence) matrix (n_fluorophores, n_channels) - (5x5) - first column for nuclei
    Output: corrected .omehans the same shape as input
    """
    pass


def color_registration():
    """
    input: 4-channel corrected colors .omehans
    output:
        - 4-channel corrected colors registered to nuclei space
        - 4 affine matrices (for each color)
    """
    pass


def preprocess_nuclei(location):
    pass


def preprocess_colors(location):
    pass
