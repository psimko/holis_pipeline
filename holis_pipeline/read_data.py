import os

from holis_tools.hicam_utils import send_hicam_to_zarr_par_read_once


def read_fli_as_zarr(path_to_fli, output_location):
    """
    input: .fli file
    Output: .omehans
    Run for colors and nuclei, and the empty frames
    """
    try:
        os.makedirs(output_location)
    except:
        pass
    send_hicam_to_zarr_par_read_once(path_to_fli, output_location)
    return output_location
