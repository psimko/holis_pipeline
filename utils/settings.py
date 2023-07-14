NUCLEI_DIR = '/bil/proj/rf1hillman/results/2023_04_04_combinatorialSlide_mouse_tiff_forIana/dataset_noOverlay_skewed/omezarr/nuclei.omehans'  # nuclei
COLORS_DIR = "/bil/proj/rf1hillman/results/2023_04_04_combinatorialSlide_mouse_tiff_forIana/dataset_noOverlay_skewed/omezarr/colors.omehans/scale0"
OUTPUT_DIR = '/bil/proj/rf1hillman/results/2023_04_04_combinatorialSlide_mouse_tiff_forIana/dataset_noOverlay_skewed/output/pytorch_unet_mouse_model/'
CHUNK_SIZE = (40, 1700, 1700)
MODEL_PATH = '/bil/proj/rf1hillman/pynet/segmentation_combMouse_128_oneVolume_v1/model.pth'  # mouse model
NUCLEI_RESOLUTION = [1.34, 1.54, 2.0]  # nuclei
COLOR_RESOLUTION = [1.34, 1.54, 2.0]
CUBE_SIZE = 10  # um to cut around nuclei
