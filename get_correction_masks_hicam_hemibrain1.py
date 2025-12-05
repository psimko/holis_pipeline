import os
import re

from glob import glob
from pathlib import Path

import numpy as np
import dask.array as da
import zarr
import pickle
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store
from numcodecs import Blosc
from scipy.ndimage import gaussian_filter1d
import scipy.io as sio
from holis_pipeline.preprocess_v2_sep2025 import split_color_channels_2D_dict, apply_affine_2d, normalize_by_registered_max_dict, parse_A2D_list, _unwrap_obj


#bg_file_name = f'NPBB328-Cortex-Slab06-run*-z{z:02d}-y000-darkFrames_HiCAM FLUO_1875-ST-272.fli.zst'

def read_omehans(path_to_omehans, scale=None):
    location = os.path.join(path_to_omehans, f'scale{scale}' if scale else "")
    store = H5_Nested_Store(location)
    zarray = zarr.open(store)
    dask_zarray = da.array(zarray)
    return dask_zarray

def get_mask(input_path, output_path):
    bg = read_omehans(input_path)
    print(f'Bg omehans volume shape is {bg.shape}')
    bg = bg.compute()
    bg = bg.astype(np.float32)
    bg_mask = np.median(bg, axis=0)  # shape: (Z, Y)
    print(f'Bg mask shape is {bg_mask.shape}')
    np.save(output_path, bg_mask)
    print(f'Saved bg mask to {output_path}')
    return output_path

def get_laser_pattern(input_path, output_path):
    pattern = read_omehans(input_path)
    print(f'Laser pattern omehans volume shape is {pattern.shape}')
    pattern = pattern.compute()
    pattern = pattern.astype(np.float32)
    pattern_mask = np.median(pattern, axis=0)  # shape: (Z, Y)
    print(f'Laser pattern mask shape is {pattern_mask.shape}')
    np.save(output_path, pattern_mask)
    print(f'Saved laser pattern mask to {output_path}')
    return output_path

##############################################################################################################

def normalize_patterns(CFG, input_dir, output_dir):

    # Import transorms - I import the CFG configuration, but this could also be passed through SLURM
    TRANSFORMS_PATH = CFG.transforms_matlab_path
    MATRICES_PATH = CFG.matrices_matlab_path 
    transforms = sio.loadmat(TRANSFORMS_PATH)
    matrices = sio.loadmat(MATRICES_PATH)

    transforms_list = parse_A2D_list(transforms)

    # Get the z range
    Z_RE = re.compile(r"laser_pattern_nuclei_BG_z(\d+)\.npy$")
    z_vals = []
    for p in Path(input_dir).glob("laser_pattern_nuclei_BG_z*.npy"):
        m = Z_RE.search(p.name)
        if m:
            z_vals.append(m.group(1))   # keeps '01', '02', ...

    z_unique = sorted(set(z_vals), key=int)
    print("found z:", z_unique)

    # Get laser intensities and masks
    lasers = ['488', '561', '594', '660']
    laser_power_at_sample = np.array([0.3276, 1.1894, 1.1290, 3.0164]) # First hemibrain slab 6/7
    laser_power = laser_power_at_sample / np.max(laser_power_at_sample)  # Normalize laser powers

    for z_str in z_unique:
        # Load laser BG from the patterns
        laser_BG_nuclei = np.load(os.path.join(input_dir,f"laser_pattern_nuclei_BG_z{z_str}.npy"))
        laser_BG_colors = np.load(os.path.join(input_dir,f"laser_pattern_colors_BG_z{z_str}.npy"))
        laser_BG_colors = np.rot90(np.rot90(laser_BG_colors))

        # Get the laser patterns for each laser
        patterns_nuclei = {}
        patterns_colors = {}
        for laser in lasers:
            patterns_nuclei[laser] = np.load(os.path.join(input_dir, f'laser_pattern_nuclei_mask_z{z_str}_{laser}.npy'))
            patterns_nuclei[laser] -= laser_BG_nuclei
            patterns_colors[laser] = np.load(os.path.join(input_dir, f'laser_pattern_colors_mask_z{z_str}_{laser}.npy'))
            patterns_colors[laser] = np.rot90(np.rot90(patterns_colors[laser]))
            patterns_colors[laser] -= laser_BG_colors


        # Load in the FOVcrop and the center position
        FOVcrop_raw = transforms['FOVcrop']
        Z_raw, Y_raw = FOVcrop_raw[0][0][0][0], FOVcrop_raw[0][0][1][0]
        Z = tuple(int(v) for v in np.atleast_1d(Z_raw).ravel())
        Y = tuple(int(v) for v in np.atleast_1d(Y_raw).ravel())
        FOVcrop = {"Z": Z, "Y": Y}

        center_raw = transforms['CenterSplitPosition'] 
        center_arr = np.atleast_1d(center_raw).astype(int).ravel()
        centerPos = (int(center_arr[0]), int(center_arr[1]))    

        # Color split - output a dictionary {laser: [ch,z,y]}
        patterns_colors_split = split_color_channels_2D_dict(patterns_colors, centerPos)
        patterns_nuclei_norm, patterns_colors_split_norm = normalize_by_registered_max_dict(patterns_nuclei, patterns_colors_split, transforms_list, FOVcrop)

        # Saving
        output_path_nuclei = os.path.join(output_dir, f'laser_pattern_nuclei_mask_z_norm{z_str}.pkl')
        output_path_colors = os.path.join(output_dir, f'laser_pattern_colors_mask_z_norm{z_str}.pkl')

        with open(output_path_nuclei , "wb") as f:
            pickle.dump(patterns_nuclei_norm, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(output_path_colors, "wb") as f:
            pickle.dump(patterns_colors_split_norm, f, protocol=pickle.HIGHEST_PROTOCOL)

##############################################################################################################

def final_normalize_patterns(CORR_MASK_NUC, CORR_MASK_SPL, transforms_list, FOVcrop):
    # 1) register all 5 channels into splitter FOV
    _, Zc, Yc = CORR_MASK_SPL.shape   # (4, Zc, Yc)
    Ztop, Zbot = FOVcrop["Z"]
    Yleft, Yright = FOVcrop["Y"]

    reg = np.empty((5, Zc, Yc), dtype=np.float32)
    reg[0] = apply_affine_2d(CORR_MASK_NUC, transforms_list[0], (Zc, Yc))
    for k in range(4):
        reg[1 + k] = apply_affine_2d(CORR_MASK_SPL[k], transforms_list[1 + k], (Zc, Yc))

    # 2) crop to FOV
    reg_c = reg[:, Ztop:Zc - Zbot, Yleft:Yc - Yright]   # (5, Zf, Yf)

    # 3) per-channel min & mean
    ch_min  = reg_c.reshape(5, -1).min(axis=1)
    ch_mean = reg_c.reshape(5, -1).mean(axis=1)
    scale = np.maximum(ch_mean - ch_min, 1e-6)          # avoid zero

    # 4) normalize unmorphed originals by those scales
    CORR_MASK_NUC = CORR_MASK_NUC / scale[0]
    CORR_MASK_SPL = CORR_MASK_SPL / scale[1:, None, None]

    # 5) add bias and re-normalize
    bias = 1.0
    CORR_MASK_NUC = CORR_MASK_NUC + bias
    CORR_MASK_SPL = CORR_MASK_SPL + bias

    CORR_MASK_NUC = CORR_MASK_NUC / CORR_MASK_NUC.mean()
    CORR_MASK_SPL = CORR_MASK_SPL / CORR_MASK_SPL.mean(axis=(1, 2), keepdims=True)

    return CORR_MASK_NUC.astype(np.float32), CORR_MASK_SPL.astype(np.float32)

##############################################################################################################

def mix_patterns(CFG, input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------- load mixing matrices from MATLAB file --------
    # Import transorms - I import the CFG configuration, but this could also be passed through SLURM
    matrices = sio.loadmat(CFG.matrices_matlab_path, squeeze_me=True, struct_as_record=False)
    transforms = sio.loadmat(CFG.transforms_matlab_path)
    transforms_list = parse_A2D_list(transforms)

    Flch = np.asarray(matrices["Flch"], dtype=np.float32)                  # (fl, ch)
    exc_eff = np.asarray(matrices["excitation_efficiency"], dtype=np.float32)  # (fl, la)

    # Normalize along rows (fluorophores): each row = contribution % per channel
    Flch_rel = Flch / np.sum(Flch, axis=1, keepdims=True)

    # lasers, powers
    lasers = ["488", "561", "594", "660"]
    laser_power_at_sample = np.array([0.3276, 1.1894, 1.1290, 3.0164], dtype=np.float32)
    laser_power = laser_power_at_sample / laser_power_at_sample.max()  # (la,)

    # LaserChannelContribution: (la, ch)
    # excitation_efficiency is fl x la → exc_eff.T is (la, fl)
    # la x fl @ fl x ch = la x ch
    LaserChannelContribution = laser_power[:, None] * (exc_eff.T @ Flch_rel)

    # -------- find z’s from normalized nuclei pickle names --------
    Z_RE = re.compile(r"laser_pattern_nuclei_mask_z_norm(\d+)\.pkl$")
    z_vals = []
    for p in input_dir.glob("laser_pattern_nuclei_mask_z_norm*.pkl"):
        m = Z_RE.search(p.name)
        if m:
            z_vals.append(m.group(1))
    z_unique = sorted(set(z_vals), key=int)
    print("mix_patterns: found z =", z_unique)

    for z_str in z_unique:
        nuc_pkl = input_dir / f"laser_pattern_nuclei_mask_z_norm{z_str}.pkl"
        col_pkl = input_dir / f"laser_pattern_colors_mask_z_norm{z_str}.pkl"

        if not nuc_pkl.exists() or not col_pkl.exists():
            print(f"Skipping z={z_str}: missing {nuc_pkl} or {col_pkl}")
            continue

        with open(nuc_pkl, "rb") as f:
            nuc_dict = pickle.load(f)      # {laser: (Z,Y)}
        with open(col_pkl, "rb") as f:
            col_dict = pickle.load(f)      # {laser: (4,Zc,Yc)}

        # stack over lasers in fixed order
        nuc_list = []
        col_list = []
        for laser in lasers:
            if laser not in nuc_dict or laser not in col_dict:
                raise KeyError(f"Laser {laser!r} missing in z={z_str}")
            nuc_list.append(np.asarray(nuc_dict[laser], dtype=np.float32))
            col_list.append(np.asarray(col_dict[laser], dtype=np.float32))

        nuc_stack = np.stack(nuc_list, axis=0)   # (la, Z_n, Y_n)
        col_stack = np.stack(col_list, axis=0)   # (la, 4, Z_c, Y_c)

        la, Z_n, Y_n = nuc_stack.shape
        la2, C, Z_c, Y_c = col_stack.shape
        if la2 != la or la != len(lasers) or C != 4:
            raise ValueError(f"Shape mismatch for z={z_str}: "
                             f"nuc {nuc_stack.shape}, col {col_stack.shape}")

        # -------- mix nuclei channel (ch=0) --------
        flat_n = nuc_stack.reshape(la, -1)              # (la, N)
        weights_n = LaserChannelContribution[:, 0:1]    # (la, 1)
        mixed_n_flat = (weights_n * flat_n).sum(axis=0) # (N,)
        CORR_MASK_NUC = mixed_n_flat.reshape(Z_n, Y_n)  # (Z_n, Y_n)

        # -------- mix splitter channels (ch=1..4) --------
        flat_s = col_stack.reshape(la, C, -1)           # (la, 4, N_c)
        weights_s = LaserChannelContribution[:, 1:1+C]  # (la, 4)
        mixed_s_flat = (weights_s[:, :, None] * flat_s).sum(axis=0)  # (4, N_c)
        CORR_MASK_SPL = mixed_s_flat.reshape(C, Z_c, Y_c)            # (4, Z_c, Y_c)

        # -------- final normalization --------
        FOVcrop_raw = transforms['FOVcrop']
        Z_raw, Y_raw = FOVcrop_raw[0][0][0][0], FOVcrop_raw[0][0][1][0]
        Z = tuple(int(v) for v in np.atleast_1d(Z_raw).ravel())
        Y = tuple(int(v) for v in np.atleast_1d(Y_raw).ravel())
        FOVcrop = {"Z": Z, "Y": Y}
        CORR_MASK_NUC, CORR_MASK_SPL = final_normalize_patterns(CORR_MASK_NUC, CORR_MASK_SPL, transforms_list, FOVcrop)

        # -------- save per z --------
        out_n = output_dir / f"laser_pattern_nuclei_mask_z{z_str}_final.npy"
        out_s = output_dir / f"laser_pattern_colors_mask_z{z_str}_final.npy"
        np.save(out_n, CORR_MASK_NUC.astype(np.float32))
        np.save(out_s, CORR_MASK_SPL.astype(np.float32))

        print(f"mixed patterns for z={z_str} -> {out_n.name}, {out_s.name}")


