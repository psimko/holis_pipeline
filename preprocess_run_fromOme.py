#import cv2  # OpenCV library for reading videos
import numpy as np
import os
import dask.array as da
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store
import zarr
import matplotlib.pyplot as plt
import tifffile
import scipy.io as sio
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from flifile import FliFile
from numcodecs import Blosc
import json
from scipy.ndimage import zoom
from scipy.optimize import nnls
from joblib import Parallel, delayed
from skimage.transform import warp, AffineTransform
from skimage.transform import resize
from skimage.measure import block_reduce

def read_omehans(path_to_omehans, scale=None):
    location = os.path.join(path_to_omehans, f'scale{scale}' if scale else "")
    store = H5_Nested_Store(location)
    zarray = zarr.open(store)
    dask_zarray = da.array(zarray)
    return dask_zarray

def color_split(data_temp, center_pos=None):
    """
    Splits a 3D image splitter input into 4 cropped channels.
    Returns a 4D array with shape (ch, z, y, x)
    """
    ss = data_temp.shape  # (z, y, x)
    center_z = center_pos[0] if center_pos else ss[0] // 2
    center_y = center_pos[1] if center_pos else ss[1] // 2

    # Split
    temp_ch1 = data_temp[center_z:, center_y:, :]
    temp_ch2 = data_temp[center_z:, :center_y, :]
    temp_ch3 = data_temp[:center_z, center_y:, :]
    temp_ch4 = data_temp[:center_z, :center_y, :]

    # Compute minimum crop
    shapes = [ch.shape for ch in [temp_ch1, temp_ch2, temp_ch3, temp_ch4]]
    crop_z = min(s[0] for s in shapes)
    crop_y = min(s[1] for s in shapes)

    # Crop to match
    temp_ch1 = temp_ch1[:crop_z, :crop_y, :]
    temp_ch2 = temp_ch2[:crop_z, :crop_y, :]
    temp_ch3 = temp_ch3[:crop_z, :crop_y, :]
    temp_ch4 = temp_ch4[:crop_z, :crop_y, :]

    # Stack to (4, z, y, x)
    return da.stack([temp_ch1, temp_ch2, temp_ch3, temp_ch4], axis=0)

##############################################################################
#### Load correction matrices
##############################################################################

correction_path = "/bil/proj/rf1hillman/results/2025_06_15_NPBB328_surface_corrections/"

corr_BG_nuclei = np.load(correction_path + "Corr_BG_nuclei.npy")
corr_BG_colors = np.load(correction_path + "Corr_BG_colors.npy")
ff_nuclei_norm = np.load(correction_path + "FF_nuc_norm.npy")
ff_colors_norm = np.load(correction_path + "FF_colors_norm.npy")
laser_correction_Nuclei_pattern = np.load(correction_path + "Laser_correction_Nuclei_pattern.npy")
laser_correction_Colors_pattern = np.load(correction_path + "Laser_correction_Colors_pattern.npy")

##############################################################################
#### Load run
##############################################################################

location_nuclei = '/bil/proj/rf1hillman/results/2025_06_15_NPBB328_surface/NPBB328-surface-run009-z02-y08-Exc-488nm-561nm-594nm-660nm_HiCAM FLUO_1875-ST-272.fli_ZARR_OUT'
location_colors = '/bil/proj/rf1hillman/results/2025_06_15_NPBB328_surface/NPBB328-surface-run009-z02-y08-Exc-488nm-561nm-594nm-660nm_HiCAM FLUO_1875-ST-088.fli_ZARR_OUT'
im_nuclei = read_omehans(location_nuclei)
im_colors = read_omehans(location_colors)

##############################################################################
#### Correct for background and laser pattern
##############################################################################

# Bg correction
corr_BG_nuclei_mask = corr_BG_nuclei - 1024 
corr_BG_colors_mask = corr_BG_colors - 1024 

im_nuclei_bg_corr = im_nuclei - corr_BG_nuclei_mask
im_colors_bg_corr = im_colors - corr_BG_colors_mask

print(f'im_nuclei_bg_corr has shape {im_nuclei_bg_corr.shape}')
print(f'im_colors_bg_corr has shape {im_colors_bg_corr.shape}')

# FF correction

im_nuclei_ff_corr = im_nuclei_bg_corr / ff_nuclei_norm
im_colors_ff_corr = im_colors_bg_corr / ff_colors_norm

# Laser pattern correction

im_nuclei_corr = im_nuclei_ff_corr / laser_correction_Nuclei_pattern
im_colors_corr = im_colors_ff_corr / np.sum(laser_correction_Colors_pattern, axis=0)

print(f'im_nuclei_corr has shape {im_nuclei_corr.shape}')
print(f'im_colors_corr has shape {im_colors_corr.shape}')

##############################################################################
#### Color split
##############################################################################

# Load registration transforms
transforms = sio.loadmat(os.path.join(f'/bil/proj/rf1hillman/2025_06_26_NPBB328_surface_processingMatlabCode_SLURM/NPBB328_colorMerge_transforms.mat')) 

im_nuclei_corr_zyx = np.transpose(im_nuclei_corr, (1, 2, 0)) 
im_colors_corr_zyx = np.transpose(im_colors_corr, (1, 2, 0))

if not isinstance(im_nuclei_corr_zyx, np.ndarray):
    im_nuclei_corr_zyx = np.array(im_nuclei_corr_zyx)

print('Performing color split.')
center_split_position = list([transforms['CenterSplitPosition'][0][0], transforms['CenterSplitPosition'][0][1]])
im_colors_corr_zyx_split = color_split(im_colors_corr_zyx, center_split_position)
print(f'im_nuclei_corr_zyx has shape {im_nuclei_corr_zyx.shape}')
print(f'im_colors_corr_zyx_split has shape {im_colors_corr_zyx_split.shape}')
print('Color split done.')

##############################################################################
#### Registration
##############################################################################

import torch
import kornia

def to_affine_batch(matrix, batch_size):
    """Convert 3x3 affine to Kornia-compatible (B, 2, 3)"""
    M = torch.tensor(matrix[:2, :], dtype=torch.float32)
    M_batch = M.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, 3)
    return M_batch

# def register_channel_stack_kornia(stack_zyx, transform_matrix, device="cuda"):
#     """
#     Apply 2D affine transform to every (Z,Y) slice at each X.

#     stack_zyx: (Z, Y, X) NumPy array
#     transform_matrix: 3x3 affine matrix (NumPy)
#     """

#     if not isinstance(stack_zyx, np.ndarray):
#         stack_zyx = np.array(stack_zyx)
            
#     Z, Y, X = stack_zyx.shape

#     # Rearrange: (Z, Y, X) → (X, Z, Y)
#     stack_xzy = np.transpose(stack_zyx, (2, 0, 1))
#     if not isinstance(stack_zyx, np.ndarray):
#         stack_zyx = np.array(stack_zyx)

#     slices = torch.from_numpy(stack_xzy).unsqueeze(1).float().to(device)  # (X, 1, Z, Y)

#     # Create batch affine matrix
#     M_batch = to_affine_batch(transform_matrix, batch_size=slices.shape[0]).to(device)

#     # Apply warp
#     warped = kornia.geometry.transform.warp_affine(
#         slices,
#         M_batch,
#         dsize=(Z, Y),
#         align_corners=False
#     )  # Output: (X, 1, Z, Y)

#     # Rearrange: (X, Z, Y) → (Z, Y, X)
#     return warped.squeeze(1).permute(1, 2, 0).cpu().numpy()

def register_channel_stack_kornia(stack_zyx, transform_matrix, device="cuda", batch_size=128):
    """
    Apply 2D affine transform to every (Z,Y) slice at each X using mini-batching.
    Kornia expect the batching dimension (here X) to follow the batch size (here B), hence the transpose.
    """
    if not isinstance(stack_zyx, np.ndarray):
        stack_zyx = np.array(stack_zyx)

    assert transform_matrix.shape == (3, 3), f"Transform matrix has shape {transform_matrix.shape}, expected (3, 3)"

    Z, Y, X = stack_zyx.shape
    stack_xzy = np.transpose(stack_zyx, (2, 0, 1))  # (X, Z, Y)

    output = np.zeros((X, Z, Y), dtype=np.float32)  # Will hold all warped slices

    for i in tqdm(range(0, X, batch_size), desc="Warping batches"):
        end = min(i + batch_size, X)
        batch = torch.from_numpy(stack_xzy[i:end].copy()).unsqueeze(1).float().to(device)  # (B, 1, Z, Y)

        M = torch.tensor(transform_matrix[:2, :], dtype=torch.float32).unsqueeze(0).repeat(end - i, 1, 1).to(device)

        warped = kornia.geometry.transform.warp_affine(
            batch, M, dsize=(Z, Y), align_corners=False
        )  # (B, 1, Z, Y)

        output[i:end] = warped.squeeze(1).cpu().numpy()

    return np.transpose(output, (1, 2, 0))  # (Z, Y, X)

# === INPUT DATA ===
# Assume the following variables are loaded:
# im_nuclei_corr_zyx: shape (Z, Y, X)
# im_colors_corr_zyx: shape (4, Z, Y, X)
# transforms['Transforms'][0][0][ch]: each a 3x3 or object-like affine matrix
# correction_path: output directory

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Downsample im_nuclei_corr_zyx

# target shape from im_colors_corr_zyx: (Z, Y, X)
z, y, x = im_colors_corr_zyx_split.shape[1:]
volReg = np.zeros((5, z, y, x), dtype=np.float32)
target_shape = im_colors_corr_zyx_split.shape[1:]  # skip channel dim
print(f'Target shape is {target_shape}')

im_nuclei_corr_zyx_rescaled = resize(
    im_nuclei_corr_zyx,
    output_shape=target_shape,
    order=1,              # bilinear interpolation
    preserve_range=True,  # don't normalize intensity
    anti_aliasing=True
).astype(np.float32)

print(f'im_nuclei_corr_zyx_rescaled has shape {im_nuclei_corr_zyx_rescaled.shape}')

np.save(correction_path + "im_nuclei_corr_zyx_rescaled.npy", im_nuclei_corr_zyx_rescaled[ :, :, 15000:16000]) 

# Faster but worse way to rescale
""" # Compute zoom factors
scale_factors = (
    target_shape[0] / im_nuclei_corr_zyx.shape[0],
    target_shape[1] / im_nuclei_corr_zyx.shape[1],
    target_shape[2] / im_nuclei_corr_zyx.shape[2],
)

# Fast zoom with nearest-neighbor (order=0) or linear (order=1)
im_nuclei_corr_zyx_rescaled = zoom(
    im_nuclei_corr_zyx,
    zoom=scale_factors,
    order=0  # nearest-neighbor; change to 1 for bilinear
).astype(np.float32).copy()

print(f'im_nuclei_corr_zyx_rescaled has shape {im_nuclei_corr_zyx_rescaled.shape}')

np.save(correction_path + "im_nuclei_corr_zyx_rescaled.npy", im_nuclei_corr_zyx_rescaled[ :, :, 15000:16000].astype(np.float32)) """
#################################

print("Transforms structure:", type(transforms['Transforms']), transforms['Transforms'].shape)

for ch in range(5):
    print(f"Registering channel {ch}")

    # Extract and convert transform matrix
    #raw = transforms['Transforms'][0][0][ch]

    print("Transforms type:", type(transforms['Transforms']))
    print("Transforms shape:", np.shape(transforms['Transforms']))
    print("Transforms dtype:", transforms['Transforms'].dtype)
    #print("Transforms content preview:", transforms['Transforms'])


    transform_matrix = transforms['Transforms'][0][ch]['A2D'].astype(np.float32)

    print(type(transform_matrix))           # <class 'tuple'>
    print(type(transform_matrix[0]))        # <class 'numpy.ndarray'>
    print(transform_matrix[0].shape) 

    print(f'Transform_matrix shape is: {transform_matrix.shape}')

    #M = np.array(raw.toarray() if hasattr(raw, "toarray") else raw).astype(np.float32)
    M = transform_matrix

    # Get stack to be transformed
    if ch == 0:
        #stack = np.rot90(np.rot90(im_nuclei_corr_zyx_rescaled, axes=(0, 1)), axes=(0, 1))  # rotate 180°
        #stack = np.rot90(im_nuclei_corr_zyx_rescaled, k=2, axes=(0, 1))
        #stack = np.flip(im_nuclei_corr_zyx_rescaled, axis=(0, 1))
        #stack = im_nuclei_corr_zyx_rescaled
        stack = np.zeros_like(im_nuclei_corr_zyx_rescaled)
        for x in range(im_nuclei_corr_zyx_rescaled.shape[2]):
            stack[:, :, x] = np.rot90(np.rot90(im_nuclei_corr_zyx_rescaled[:, :, x])).copy()
        print(f'Stack {ch} shape is {stack.shape}')
    else:
        stack = im_colors_corr_zyx_split[ch - 1]
        print(f'Stack {ch} has shape {stack.shape}')

    # Apply batch warp
    if ch == 0:
        registered_stack = stack
    else:
        registered_stack = register_channel_stack_kornia(stack, M, device=device)
    print(f'Registered stack {ch} has shape {registered_stack.shape}')
    volReg[ch] = registered_stack
    np.save(correction_path + f"channel_{ch}_corr_zyx_rescaled.npy", registered_stack[ :, :, 15000:16000])

# Save subset of result
print('Saving registered volume')
#np.save(correction_path + "volReg.npy", volReg[:, :, :, 15000:16000])
tifffile.imwrite(
    correction_path + "volReg.tiff",
    volReg[:, :, :, 15000:16000].astype(np.float32),  # Ensure it's a supported type
    imagej=True  # Optional: if you want ImageJ compatibility
)

# Checks
print("volReg shape before slicing:", volReg.shape)
print("volReg[0] shape:", volReg[0].shape)
loaded = np.load(correction_path + "volReg.npy", mmap_mode='r')
print("Loaded shape from file:", loaded.shape)
print("Channel 0 shape in file:", loaded[0].shape)

##############################################################################
#### Unmixing
##############################################################################

print('Unmixing registered channels')
simulation_matrices = sio.loadmat('/bil/proj/rf1hillman/2024_07_29_AI7_EH5k_human_finalMarkerCombination_100mm/code_Matlab/SimulationMatrices_equalPower_firstHemibrain')# Fluorophore x Channel matrix normalization
Flch = simulation_matrices['Flch']
Flch_rel = Flch.copy()
# Normalize along columns - so each entry (i,j) is the percentage of the signal in channel j coming from fluorophore i
Flch_rel = Flch_rel / np.sum(Flch_rel, axis=1, keepdims=True)
M_inv = np.linalg.inv(Flch_rel)

# Assume volReg has shape (5, Z, Y, X)
""" ss = volReg.shape  # (5, Z, Y, X)
volReg_flat = volReg.reshape(ss[0], -1)  # shape: (5, Z*Y*X)
volReg_unmixed_flat = M_inv.T @ volReg_flat  # shape: (5, Z*Y*X)
volReg_unmixed = volReg_um_flat.reshape(ss)  """

def unmix_volReg_batched(volReg, M_inv, batch_size=10_000_000):
    C, Z, Y, X = volReg.shape
    N = Z * Y * X
    volReg_flat = volReg.reshape(C, N)

    volReg_unmixed_flat = np.empty_like(volReg_flat)

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        volReg_unmixed_flat[:, i:end] = M_inv.T @ volReg_flat[:, i:end]

    return volReg_unmixed_flat.reshape((C, Z, Y, X))

print("Unmixing registered channels in batches...")
volReg_unmixed = unmix_volReg_batched(volReg, M_inv, batch_size=10_000_000)

# Save subset of result
print('Saving unmixed registered volume')
np.save(correction_path + "volReg_unmixed.npy", volReg_unmixed[:, :, :, 15000:16000])























