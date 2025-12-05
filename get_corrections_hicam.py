import os

from glob import glob

import numpy as np
import dask.array as da
import zarr
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store
from numcodecs import Blosc
from scipy.ndimage import gaussian_filter1d


def read_omehans(path_to_omehans, scale=None):
    location = os.path.join(path_to_omehans, f'scale{scale}' if scale else "")
    store = H5_Nested_Store(location)
    zarray = zarr.open(store)
    dask_zarray = da.array(zarray)
    return dask_zarray

correction_path = "/h20/Public/holis/2025_06_15_NPBB328_surface_corrections/"


# ## ------- Flat field correction ------- ##

ffNuclei_path = correction_path + 'externalEpoxy-FF-run001-LEDblue_HiCAM FLUO_1875-ST-272.fli'
ffNuclei_path = os.path.join(os.path.dirname(correction_path), os.path.basename(ffNuclei_path).replace('.fli', '.omehans'))
ffNuclei = read_omehans(ffNuclei_path)
print("ffNuclei shape", ffNuclei.shape)
ffNuclei = ffNuclei.compute()
ffNuclei = ffNuclei.astype(np.float32)
FFb_nuclei = np.median(ffNuclei, axis=0)  # shape: (Y, X)
print("FFb_nuclei shape", FFb_nuclei.shape)
np.save(correction_path + "FFb_nuclei1.npy", FFb_nuclei)


# --- Load background flat field volume ---
ffbgNuclei_path = correction_path + "externalEpoxy-Laser-run001-darkFrames_HiCAM FLUO_1875-ST-272.fli"
ffbgNuclei_path = os.path.join(os.path.dirname(correction_path), os.path.basename(ffbgNuclei_path).replace('.fli', '.omehans'))
ffbgNuclei = read_omehans(ffbgNuclei_path)
print("ffbgNuclei shape", ffbgNuclei.shape)
ffbgNuclei = ffbgNuclei.compute()
ffbgNuclei = ffbgNuclei.astype(np.float32)

# --- Compute flat field background mask ---
FF_BG_nuclei = np.median(ffbgNuclei, axis=0)  # shape: (Y, X)
print("FF_BG_nuclei shape", FF_BG_nuclei.shape)
np.save(correction_path + "FF_BG_nuclei.npy", FF_BG_nuclei)

FF_BG_nuclei_mask = FF_BG_nuclei - 1024.0  # subtract 2^10
FFb_nuclei = FFb_nuclei - FF_BG_nuclei_mask  # shape: (Y, X)
print("FFb_nuclei.shape", FFb_nuclei.shape)
np.save(correction_path + "FFb_nuclei2.npy", FFb_nuclei)


# --- Normalize flat field ---
FF_nuc_norm = FFb_nuclei / np.median(FFb_nuclei)
print("FF_nuc_norm.shape", FF_nuc_norm.shape)
np.save(correction_path + "FF_nuc_norm.npy", FF_nuc_norm)


# ## ------- Laser pattern correction ------- ##

corrbgNuclei_path = correction_path + "NPBB328-corrections-run001-darkFrames_HiCAM FLUO_1875-ST-272.fli"
corrbgNuclei_path = os.path.join(os.path.dirname(correction_path), os.path.basename(corrbgNuclei_path).replace('.fli', '.omehans'))
print("reading corrbgNuclei")
corrbgNuclei = read_omehans(corrbgNuclei_path)
print("corrbgNuclei shape", corrbgNuclei.shape)
Corr_BG_nuclei = np.median(corrbgNuclei.astype(np.float32), axis=0)
print("Corr_BG_nuclei shape", Corr_BG_nuclei.shape)
np.save(correction_path + "Corr_BG_nuclei", Corr_BG_nuclei)


Corr_BG_nuclei_mask = Corr_BG_nuclei - 1024.0

# --- Prepare arrays to store masks ---
lasers = [488, 561, 594, 660]
n_lasers = len(lasers)
Y, X = Corr_BG_nuclei_mask.shape
POWELL_NUC_MASK = np.zeros((n_lasers, Y, X), dtype=np.float32)
print("POWELL_NUC_MASK shape", POWELL_NUC_MASK.shape)

# --- Loop over lasers ---
for i, las in enumerate(lasers):
    print("laser", las)
    # Locate file
    Powell_Nuclei_path = glob(correction_path + f'NPBB328-corrections-epoxy-run*{las}nm_HiCAM FLUO_1875-ST-272.fli')[0]
    Powell_Nuclei_path = os.path.join(os.path.dirname(correction_path), os.path.basename(Powell_Nuclei_path).replace('.fli', '.omehans'))

    # Read HiCAM stacks
    print("reading Powell_Nuclei")
    Powell_Nuclei = read_omehans(Powell_Nuclei_path)

    Powell_Nuclei = Powell_Nuclei.astype(np.float32)

    # Median projection and background subtraction
    Powell_Nuclei_mask = np.median(Powell_Nuclei, axis=0) - Corr_BG_nuclei_mask
    print("Powell_Nuclei_mask shape", Powell_Nuclei_mask.shape)

    # Save masks
    POWELL_NUC_MASK[i, :, :] = Powell_Nuclei_mask

np.save(correction_path + "POWELL_NUC_MASK.npy", POWELL_NUC_MASK)

FF_nuc_norm = np.load('/h20/Public/holis/2025_06_15_NPBB328_surface_corrections/FF_nuc_norm.npy')

Y, X = FF_nuc_norm.shape
print("Y, X", Y, X)

# Output array
POWELL_NUC_MASK_FF_norm = np.zeros((n_lasers, Y, X), dtype=np.float32)
print("POWELL_NUC_MASK_FF_norm shape", POWELL_NUC_MASK_FF_norm.shape)

for i, las in enumerate(lasers):
    nuc_temp = POWELL_NUC_MASK[i, :, :] / FF_nuc_norm
    print("nuc_temp shape", nuc_temp.shape)

    cropped_nuc = nuc_temp[250:-250, 250:-250]
    nuc_temp -= np.min(cropped_nuc)
    nuc_temp /= np.max(cropped_nuc)

    POWELL_NUC_MASK_FF_norm[i, :, :] = nuc_temp

np.save(correction_path + "POWELL_NUC_MASK_FF_norm.npy", POWELL_NUC_MASK_FF_norm)

# --- Nuclei correction from first laser only ---
Laser_correction_Nuclei = POWELL_NUC_MASK_FF_norm[0, :, :]  # shape: (Y, X)


# Compute smooth mask (column stripes) from nuclei pattern
median_profile = np.median(Laser_correction_Nuclei, axis=0)  # shape: (X,)

print("median_profile", median_profile.shape)

smoothed_profile = gaussian_filter1d(median_profile, sigma=Laser_correction_Nuclei.shape[1] / 10)
Laser_correction_Nuclei_pattern = np.outer(np.ones(Laser_correction_Nuclei.shape[0]), median_profile)
Laser_correction_Nuclei_pattern /= smoothed_profile  # broadcasting column-wise

Laser_correction_Nuclei_pattern = ((Laser_correction_Nuclei_pattern - 1) / 3) + 1

print("Laser_correction_Nuclei_pattern.shape", Laser_correction_Nuclei_pattern.shape)

np.save(correction_path + "Laser_correction_Nuclei_pattern.npy", Laser_correction_Nuclei_pattern)
