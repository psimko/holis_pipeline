def subtract_background():
    """
    input: data .omehans
           empty frames (.mat) - the same shape as the data
    output: flattened .omehans - same shape as input
    """
    pass


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
