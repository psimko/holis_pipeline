# NUCLEI_DIR = '/bil/proj/rf1hillman/results/2023_08_15_combinatorialSlides2_AI7_EH5f/AI7_EH5f3/omezarr_16bit/nuclei.omehans'  # nuclei
# COLORS_DIR = '/bil/proj/rf1hillman/results/2023_08_15_combinatorialSlides2_AI7_EH5f/AI7_EH5f3/omezarr_16bit/colors.omehans'
# OUTPUT_DIR = '/bil/proj/rf1hillman/results/2023_08_15_combinatorialSlides2_AI7_EH5f/AI7_EH5f3/output_bridges2_wholeSlab/'
#CHUNK_SIZE = (40, 1700, 1700)
#CHUNK_SIZE = (20,850,850)
CHUNK_SIZE = (850,850,850)
MODEL_PATH = '/bil/proj/rf1hillman/pynet/segmentation_mouse_pruned_norm_scaled__128_oneVol_v2_w1_b1_e12_from345/model.pth'  # mouse model
NUCLEI_RESOLUTION = [1.34, 1.54, 2.0]  # nuclei
COLOR_RESOLUTION = [1.34, 1.54, 2.0]
CUBE_SIZE = 10  # um to cut around nuclei
GPU_ENV_NAME = 'stack_to_multiscale_ngff' #'pytorch'
LNODE_ENV_NAME = 'deepblink2'
SCALE = 0  # resolution level to use for analysis
NUCLEI_CHANNEL = 0


# masks
FOREGROUND_MASKS_ENABLED = False
DENSE_REGION_MASK_ENABLED = False
SCALE_USED_FOR_MASKS = 2
DENSE_REGIONS_MASK = '/bil/proj/rf1hillman/results/2023_08_15_combinatorialSlides2_AI7_EH5f/AI7_EH5f3/output_bridges2_wholeSlab/bright_mask.tif'
FOREGROUND_MASK = '/bil/proj/rf1hillman/results/2023_08_15_combinatorialSlides2_AI7_EH5f/AI7_EH5f3/output_bridges2_wholeSlab/bg_fg_mask.tif'


# Pre-processing
EMPTY_FRAMES_FILE_NAME_FORMAT = "*-darkFrames_*"  # background (empty) scans
# EMPTY_FRAMES_FILE_NAME_FORMAT = "wholeScanBG-*"  # background (empty) scans
LASER_PATTERN_MATRIX = ""  # text file location
ABSORPTION_MATRIX = ""  # text file location
MIXING_MATRIX = ""  # text file location

# Load matlab files
#CORRECTION_DATA = '/CBI_FastStore/Iana/holis/hicam/code_Matlab/correctionMatrices_dualHiCAM_100mm_07-29-2024.mat'
#REGISTRATION_DATA = '/CBI_FastStore/Iana/holis/hicam/code_Matlab/registrationMatrices_sample.mat'
#LASER_CORRECTION_DATA = '/CBI_FastStore/Iana/holis/hicam/code_Matlab/SimulationMatrices_equalPower_firstHemibrain'

CORRECTION_DATA = '/bil/proj/rf1hillman/2024_07_29_AI7_EH5k_human_finalMarkerCombination_100mm/code_Matlab/correctionMatrices_dualHiCAM_100mm_07-29-2024.mat'
REGISTRATION_DATA = '/bil/proj/rf1hillman/2024_07_29_AI7_EH5k_human_finalMarkerCombination_100mm/code_Matlab/registrationMatrices_sample.mat'
LASER_CORRECTION_DATA = '/bil/proj/rf1hillman/2024_07_29_AI7_EH5k_human_finalMarkerCombination_100mm/code_Matlab/SimulationMatrices_equalPower_firstHemibrain'

dir_name = '/bil/proj/rf1hillman/2024_07_29_AI7_EH5k_human_finalMarkerCombination_100mm/raw_HiCAMdata/'
location_nuclei = 'wholeScan-run013-z01-y13-Exc-488nm-561nm-594nm-660nm_HiCAM FLUO_1875-ST-272.fli_ZARR_OUT/'
location_colors = 'wholeScan-run013-z01-y13-Exc-488nm-561nm-594nm-660nm_HiCAM FLUO_1875-ST-088.fli_ZARR_OUT/'
