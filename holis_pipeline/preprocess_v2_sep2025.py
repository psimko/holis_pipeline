import os
import sys
from glob import glob
import re

import dask.array as da
import numpy as np
import scipy
import tifffile
from skimage import io, img_as_float32, img_as_float, img_as_uint
from skimage.transform import resize
import zarr
import scipy.io as sio
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import cv2
import pickle

from holis_pipeline import settings
from holis_pipeline.preprocessing_functions import read_data_file, infer_z, infer_laser_nm, EXC_RE
from holis_pipeline.read_data import read_fli_as_zarr
from holis_pipeline.utils.zarr_related import read_omehans, write_omehans, write_zarr, write_omehans_from_dask


##############################################################################
#### Load correction matrices
##############################################################################

# These should be global constants

#correction_path = "/bil/proj/rf1hillman/results/2025_06_15_NPBB328_surface_corrections/"

#corr_BG_nuclei = np.load(correction_path + "Corr_BG_nuclei.npy")
#corr_BG_colors = np.load(correction_path + "Corr_BG_colors.npy")
#ff_nuclei_norm = np.load(correction_path + "FF_nuc_norm.npy")
#ff_colors_norm = np.load(correction_path + "FF_colors_norm.npy")
#laser_correction_Nuclei_pattern = np.load(correction_path + "Laser_correction_Nuclei_pattern.npy")
#laser_correction_Colors_pattern = np.load(correction_path + "Laser_correction_Colors_pattern.npy")

##############################################################################
#### Background and laser pattern correction functions
##############################################################################

def _clip16(a):
    return np.clip(a, 0, 65535).astype(np.uint16)

def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-6):
    return a / (b + eps)

def subtract_background(input_location, output_location, bg_zy: np.ndarray):

    """
    Input: data .omehans (X,Z,Y), dark frames (.npy) (Z,Y) (1 file per nuclei+colors fli pair), dark frames will be broadcast across X 
    Output: .omehans (X,Z,Y)
    """

    omehans_root = os.path.join(output_location, 'omehans')
    if os.path.exists(os.path.join(omehans_root, '0', '0', '0')):
        print("BG already subtracted previously")
        return

    print("Subtracting BG...")

    #####################
    # Numpy version
    #vol_xzy = read_omehans(input_location).compute().astype(np.float32)  # (X,Z,Y)
    #X, Z, Y = vol_xzy.shape
    #if bg_zy.shape != (Z, Y):
    #    raise ValueError(f"BG mask must be (Z,Y)={Z,Y}, got {bg_yz.shape}")
    #bg = bg_zy.astype(np.float32) - 1024.0  # your original -1024
    #out = vol_xzy - bg  # broadcast over X
    #np.maximum(out, 0, out)
    #####################    

    #####################
    # Dask version
    vol_xzy = read_omehans(input_location).astype('float32')  # (X,Z,Y)
    X, Z, Y = vol_xzy.shape
    if bg_zy.shape != (Z, Y):
        raise ValueError(f"BG mask must be (Z,Y)={Z,Y}, got {bg_zy.shape}")
    bg_zy_d = da.from_array(bg_zy, chunks=(vol_xzy.chunks[1], vol_xzy.chunks[2])).astype('float32') #- 1024.0
    out = da.maximum(vol_xzy.astype('float32') - bg_zy_d[None, :, :], 0)
    #####################

    # Save
    try:
        #write_omehans(omehans_root, _clip16(out))           #Use when using numpy version
        write_omehans_from_dask(omehans_root, da.clip(out, 0, 65535).astype('uint16')) 
    except zarr.errors.ContainsArrayError:
        print(".omehans array already exists")
    #try:
    #    write_zarr(os.path.join(output_location, 'zarr'), _clip16(out))
    #except zarr.errors.ContainsArrayError:
    #    print(".zarr array already exists")

    # tifffile.imwrite(os.path.join(output_location, "bg_subtracted.tif"), image_np_array_bgSubtracted.astype('uint16'))

    print("Saved BG-subtracted file")

    #return image_np_array_bgSubtracted



def laser_correction_nuclei(input_location, output_location, pattern_zy):
    """
    Input:
        - flattened .omehans (X,Z,Y)
        - laser pattern matrix (Z,Y)
        - absorption matrix (n_fluorophores, n_lasers) - (5x4) - first row   for nuclei
        - mixing (fluorescence) matrix (n_fluorophores, n_channels) - (5x5) - first column for nuclei
    Output: corrected .omehans the same shape as input
    """

    root = os.path.join(output_location, 'omehans')
    if os.path.exists(os.path.join(root, '0', '0', '0')):
        print("Laser pattern corrected (nuclei) previously")
        return

    print("Laser correction (nuclei): XZY vol ÷ ZY pattern")
    img = read_omehans(os.path.join(input_location, "omehans")).compute().astype(np.float32)  # (X,Z,Y)
    X, Z, Y = img.shape
    #if ff_zy.shape != (Z, Y) or pattern_zy.shape != (Z, Y):
    #    raise ValueError("FF and pattern must be (Z,Y)")

    #ff = ff_zy.astype(np.float32)
    pat = pattern_zy.astype(np.float32)

    #ff_corr = _safe_div(img, ff)
    #corr = _safe_div(ff_corr, pat)
    corr = _safe_div(img, pat)

    # Save
    try:
        write_omehans(root, _clip16(corr))
    except zarr.errors.ContainsArrayError:
        print(".omehans already exists")
    #try:
    #    write_zarr(os.path.join(output_location, 'zarr'), _clip16(corr))
    #except zarr.errors.ContainsArrayError:
    #    print(".zarr already exists")
    #tifffile.imwrite(os.path.join(output_location, 'laser_corrected.tif'), np.round(corrected_data).astype('uint16'))
    print("Saved laser-pattern-corrected (nuclei) file")

def laser_correction_colors(input_location, output_location, pattern_czy, center_pos=None):
    """
    Laser correction for color splitter data.

    input_location:  folder with raw colors 'omehans' (X,Z,Y)
    output_location: where corrected 4-channel omehans will be written
    pattern_czy:     np.ndarray of shape (C=4, Zc, Yc) final mixed patterns
    center_pos:      (cz, cy) in (Z,Y); if None, use midpoints
    """

    root = os.path.join(output_location, "omehans")
    # if corrected already present, skip
    if os.path.exists(os.path.join(root, "0", "0", "0")):
        print("Laser pattern corrected (colors) previously")
        return

    print("Laser correction (colors): split (X,Z,Y) -> (C,X,Zc,Yc) and divide by (C,Zc,Yc) pattern")

    # 1) load raw splitter volume (X,Z,Y)
    vol = read_omehans(os.path.join(input_location, "omehans")).compute().astype(np.float32)
    X, Z, Y = vol.shape

    # 2) split into 4 quadrants in (Z,Y), same as your split_color_channels
    if center_pos is not None:
        cz, cy = int(center_pos[0]), int(center_pos[1])
    else:
        cz, cy = Z // 2, Y // 2

    ch1 = vol[:, :cz, :cy]
    ch2 = vol[:, cz:, :cy]
    ch3 = vol[:, :cz, cy:]
    ch4 = vol[:, cz:, cy:]

    # crop to common min size along Z/Y
    Zc = min(ch1.shape[1], ch2.shape[1], ch3.shape[1], ch4.shape[1])
    Yc = min(ch1.shape[2], ch2.shape[2], ch3.shape[2], ch4.shape[2])
    chs = [c[:, :Zc, :Yc] for c in (ch1, ch2, ch3, ch4)]
    # (4, X, Zc, Yc)
    colors_cxzy = np.stack(chs, axis=0).astype(np.float32)

    # 3) check pattern and divide
    pat = np.asarray(pattern_czy, dtype=np.float32)
    if pat.ndim != 3 or pat.shape[0] != colors_cxzy.shape[0] or pat.shape[1:] != (Zc, Yc):
        raise ValueError(
            f"pattern_czy must be (C,Zc,Yc) matching split volume; "
            f"got {pat.shape}, expected ({colors_cxzy.shape[0]},{Zc},{Yc})"
        )

    # broadcast pattern over X: (C,1,Zc,Yc) → (C,X,Zc,Yc)
    pat_cxzy = pat[:, None, :, :]
    corr = _safe_div(colors_cxzy, pat_cxzy)  # same helper as nuclei

    # 4) save corrected 4-channel omehans
    try:
        write_omehans(root, _clip16(corr))   # corr shape (4,X,Zc,Yc)
    except zarr.errors.ContainsArrayError:
        print(".omehans already exists (colors, corrected)")

    # optional zarr, like in split_color_channels
    #write_zarr(os.path.join(output_location, "zarr"), _clip16(corr))

    print("Saved laser-pattern-corrected colors (C=4, X, Zc, Yc)")



""" def laser_correction_colors(input_location, output_location, pattern_czy, chunk_x: Optional[int] = None, target_bytes: int = 512 * 1024 * 1024, center_crop: int = 100,):

    #data = tifffile.imread(os.path.join(input_location, "color_split.tif"))

    root = os.path.join(output_location, 'omehans')
    if os.path.exists(os.path.join(root, '0', '0', '0')):
        print("Color laser correction already done")
        return

    print("Laser correction (colors): XZY ÷ ZY ÷ ZY")
    img = read_omehans(os.path.join(input_location, "omehans")).compute().astype(np.float32)  # (X,Z,Y)
    X, Z, Y = img.shape

    def to_zy(mask: np.ndarray, name: str) -> np.ndarray:
        m = np.asarray(mask)
        if m.ndim == 2:
            if m.shape != (Z,Y):
                raise ValueError(f"{name} must be (Z,Y)={(Z,Y)}, got {m.shape}")
            out = m.astype(np.float32, copy=False)
        elif m.ndim == 3:
            if m.shape[1:] != (Z,Y):
                raise ValueError(f"{name} (C,Z,Y) must match (Z,Y)={(Z,Y)}, got {m.shape}")
            out = np.nanmean(m.astype(np.float32, copy=False), axis=0)  # average over C → (Z,Y)
        else:
            raise ValueError(f"{name} must be (Z,Y) or (C,Z,Y); got {m.shape}")

        # Optional normalization by interior max (like your 100:-100 crop logic)
        if center_crop > 0:
            y0, y1 = center_crop, max(center_crop, Y - center_crop)
            z0, z1 = center_crop, max(center_crop, Z - center_crop)
            if y1 > y0 and z1 > z0:
                denom = float(np.nanmax(out[z0:z1, y0:y1]))
                if denom > 0:
                    out = _safe_div(out, denom)
        return out

    #ff = to_zy(ff_czy, "ff_czy")
    pat = to_zy(pattern_czy, "pattern_czy")

    # keep non-negatives (BG-subtracted data can dip slightly below 0)
    #np.maximum(img, 0, out=img)

    out = np.empty_like(img, dtype=np.float32)

    # FAST PATH: try full-broadcast in one go
    try:
        out = _safe_div(_safe_div(img, ff), pat)  # broadcasts (Z,Y)) over X
    except MemoryError:
        # FALLBACK: chunk along X to reduce peak memory
        if chunk_x is None:
            # rough heuristic: bytes per X-slice ≈ Y*Z*4
            bytes_per_x = Y * Z * 4
            chunk_x = max(1, int(target_bytes // bytes_per_x))

        for i in range(0, X, chunk_x):
            j = min(i + chunk_x, X)
            block = img[i:j]  # (x, Z, Y)) view
            block = _safe_div(_safe_div(block, ff), pat)
            out[i:j] = block

    # Save
    try:
        write_omehans(root, _clip16(out))
    except zarr.errors.ContainsArrayError:
        print(".omehans already exists")
    #try:
    #    write_zarr(os.path.join(output_location, 'zarr'), _clip16(out))
    #except zarr.errors.ContainsArrayError:
    #    print(".zarr already exists")
    print("Saved laser-pattern-corrected colors") """



##############################################################################
#### Color split function
##############################################################################

def split_color_channels(input_location, output_location, center_pos=None):
    """
    Splits a 3D image splitter input (X,Z,Y) into 4 cropped channels.
    Returns a 4D array with shape (C=4, X, Zc, Yc) cropped to common (Zc, Yc)
    """

    # data_temp = tifffile.imread(os.path.join(input_location, "bg_subtracted.tif"))

    root = os.path.join(output_location, 'omehans')
    if os.path.exists(os.path.join(root, '0', '0', '0')):
        print("Colors already split previously")
        return

    print("Splitting color channels on (Z,Y) axes...")
    vol = read_omehans(os.path.join(input_location, "omehans")).compute().astype(np.float32)  # (X,Z,Y)
    X, Z, Y = vol.shape
    cz = center_pos[0] if center_pos else Z // 2
    cy = center_pos[1] if center_pos else Y // 2

    ch1 = vol[:, :cz, :cy]   # top-left in (Z,Y)
    ch2 = vol[:, cz:, :cy]
    ch3 = vol[:, :cz, cy:]
    ch4 = vol[:, cz:, cy:]

    # crop to common min size along Z/Y
    Zc = min(ch1.shape[1], ch2.shape[1], ch3.shape[1], ch4.shape[1])
    Yc = min(ch1.shape[2], ch2.shape[2], ch3.shape[2], ch4.shape[2])
    chs = [c[:, :Zc, :Yc] for c in (ch1, ch2, ch3, ch4)]
    out = np.stack(chs, axis=0)  # (4, X, Zc, Yc)

    """
    data_temp_zarray = read_omehans(os.path.join(input_location, "omehans"))
    data_temp = data_temp_zarray.compute()
    data_temp_zyx = np.transpose(data_temp, (1, 2, 0)) 

    ss = data_temp_zyx.shape  # (z, y, x)
    center_z = center_pos[0] if center_pos else ss[0] // 2
    center_y = center_pos[1] if center_pos else ss[1] // 2

    # Split
    temp_ch1 = data_temp_zyx[:center_z, :center_y, :]
    temp_ch2 = data_temp_zyx[:center_z, center_y:, :]
    temp_ch3 = data_temp_zyx[center_z:, :center_y, :]
    temp_ch4 = data_temp_zyx[center_z:, center_y:, :]

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
    #return da.stack([temp_ch1, temp_ch2, temp_ch3, temp_ch4], axis=0)

    data_temp_czyx = np.concatenate([temp_ch1[None, :, :, :], temp_ch2[None, :, :, :], temp_ch3[None, :, :, :], temp_ch4[None, :, :, :]], axis=0) """

    # Save
    # tifffile.imwrite(os.path.join(output_location, "color_split.tif"), data_temp)
    #try:
    #    write_omehans(root, _clip16(out))
    #except zarr.errors.ContainsArrayError:
    #    print(".omehans already exists")
    #write_zarr(os.path.join(output_location, "zarr"), _clip16(out))
    #print("Saved color split (C=4, X, Z, Y)")

    return out

##############################################################################

def split_color_channels_2D_dict(patterns_colors: dict, center_pos):
    """
    patterns_colors: {laser: np.ndarray of shape (Z, Y)}
    center_pos: (cz, cy) splitter center in (Z, Y)
    returns: {laser: np.ndarray of shape (4, Zc, Yc)} with common (Zc, Yc)
    """
    if not patterns_colors:
        raise ValueError("patterns_colors is empty")

    cz, cy = int(center_pos[0]), int(center_pos[1])

    # First pass: compute per-laser quadrant sizes → global common crop (Zc, Yc)
    Zc = Yc = None
    quads_tmp = {}  # {laser: (ch1,ch2,ch3,ch4)}
    for laser, arr in patterns_colors.items():
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(f"Laser {laser!r} expected 2D (Z,Y), got {a.shape}")

        # layout to match the matlab code:
        # ch1: [0:cz, 0:cy], ch2: [0:cz, cy:], ch3: [cz:, 0:cy], ch4: [cz:, cy:]
        ch1 = a[:cz, :cy]
        ch2 = a[:cz, cy:]
        ch3 = a[cz:, :cy]
        ch4 = a[cz:, cy:]
        quads = (ch1, ch2, ch3, ch4)
        quads_tmp[laser] = quads

        zmin = min(q.shape[0] for q in quads)
        ymin = min(q.shape[1] for q in quads)
        Zc = zmin if Zc is None else min(Zc, zmin)
        Yc = ymin if Yc is None else min(Yc, ymin)

    # Second pass: crop all to (Zc, Yc) and stack per laser → (4, Zc, Yc)
    out = {}
    for laser, quads in quads_tmp.items():
        stacked = np.empty((4, Zc, Yc), dtype=np.float32)
        for c, q in enumerate(quads):
            stacked[c] = q[:Zc, :Yc]
        out[laser] = stacked

    return out



##############################################################################
#### Registration functions
##############################################################################

def apply_affine_2d(img_zy: np.ndarray, A2D: np.ndarray, out_shape_zy) -> np.ndarray:
    """
    img_zy: (Z,Y) float32
    A2D: 3x3 or 2x3 affine matrix (row/col coords). If 3x3, takes top 2 rows.
    out_shape_zy: (Z, Y)
    """
    if A2D.shape == (3, 3):
        M = A2D[:2, :]
    elif A2D.shape == (2, 3):
        M = A2D
    else:
        raise ValueError(f"Affine must be 2x3 or 3x3, got {A2D.shape}")
    Z, Y = out_shape_zy
    # OpenCV uses (width, height) == (Y, Z)
    return cv2.warpAffine(img_zy, M, dsize=(Y, Z), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)

def normalize_by_registered_max_dict(
    patterns_nuclei: dict,           # {laser: (Z,Y) float / uint}
    patterns_colors: dict,           # {laser: (4,Z,Y) float / uint}
    transforms: list,                # [T_nuc, T_ch1, T_ch2, T_ch3, T_ch4] each 2x3/3x3
    FOVcrop: dict                    # {"Z": (top,bottom), "Y": (left,right)}
):
    """
    Returns:
      nuclei_norm: {laser: (Z,Y) float32}
      colors_norm: {laser: (4,Zc,Yc) float32}
    """
    lasers_n = set(patterns_nuclei.keys())
    lasers_c = set(patterns_colors.keys())
    if lasers_n != lasers_c:
        missing = lasers_n ^ lasers_c
        raise ValueError(f"Mismatched lasers between nuclei/colors: {missing}")
    if len(transforms) != 5:
        raise ValueError("Need 5 transforms: [nuc, ch1, ch2, ch3, ch4].")

    # reference shape
    _, z0, y0 = next(iter(patterns_colors.values())).shape
    Ztop, Zbot = FOVcrop["Z"]
    Yleft, Yright = FOVcrop["Y"]
    eps = 1e-6

    nuclei_norm, colors_norm, maskRegMax = {}, {}, {}

    for L in patterns_nuclei.keys():
        nuc = np.asarray(patterns_nuclei[L], dtype=np.float32)
        col = np.asarray(patterns_colors[L], dtype=np.float32)   # (4,Z,Y)
        if  col.shape != (4, z0, y0):
            raise ValueError(f"Laser {L!r} inconsistent shapes: nuc {nuc.shape}, col {col.shape}")

        # 1) register each of the 5 channels
        reg = np.empty((5, z0, y0), dtype=np.float32)
        reg[0] = apply_affine_2d(nuc, transforms[0], (z0, y0))
        for k in range(4):
            reg[1 + k] = apply_affine_2d(col[k], transforms[1 + k], (z0, y0))

        # 2) crop to eventual FOV
        reg_c = reg[:, Ztop:z0 - Zbot, Yleft:y0 - Yright]  # (5, Zc, Yc)

        # 3) per-channel max over (Z,Y)
        ch_max = reg_c.reshape(5, -1).max(axis=1).astype(np.float32)
        ch_max = np.maximum(ch_max, eps)  # avoid div-by-zero
        maskRegMax[L] = ch_max

        # 4) normalize the UNregistered originals
        nuclei_norm[L] = (nuc / ch_max[0]).astype(np.float32, copy=False)
        colors_norm[L] = (col / ch_max[1:, None, None]).astype(np.float32, copy=False)

    return nuclei_norm, colors_norm

##############################################################################

import torch
import kornia

def register_channel_stack_kornia(stack_zyx, transform_matrix, device="cuda", batch_size=128):
    """
    Apply 2D affine transform to every (Z,Y) slice at each X using mini-batching.
    Kornia expect the batching dimension (here X) to follow the batch size (here B), hence the transpose.
    """
    if not isinstance(stack_zyx, np.ndarray):
        stack_zyx = np.array(stack_zyx)

    assert transform_matrix.shape == (3, 3), f"Transform matrix has shape {transform_matrix.shape}, expected (3, 3)"

    Z, Y, X = stack_zyx.shape
    xzy = np.transpose(stack_zyx, (2, 0, 1))  # (X,Z,Y)
    out = np.empty_like(xzy, dtype=np.float32)

    M = torch.tensor(transform_matrix[:2, :], dtype=torch.float32, device=device)

    for i in tqdm(range(0, X, batch_size), desc="Warping batches"):
        j = min(i + batch_size, X)
        batch = torch.from_numpy(xzy[i:j]).to(device).unsqueeze(1).float()  # (B,1,Z,Y)
        Mb = M.unsqueeze(0).repeat(j - i, 1, 1)  # (B,2,3)
        warped = kornia.geometry.transform.warp_affine(batch, Mb, dsize=(Z, Y), align_corners=False)
        out[i:j] = warped.squeeze(1).cpu().numpy()

    return np.transpose(out, (1, 2, 0))  # (Z,Y,X)

def warp_stack_xzy_kornia(stack_xzy, M_pix, *, device="cuda", batch_size=128):
    """
    stack_xzy: np.ndarray float/uint16 etc. shape (X, Z, Y)
    M_pix    : 3x3 (or 2x3) affine in pixel coords; applied to each (Z,Y) slice
    returns  : np.float32 (X, Z, Y)
    """
    if not isinstance(stack_xzy, np.ndarray):
        stack_xzy = np.asarray(stack_xzy)

    X, Z, Y = stack_xzy.shape
    out = np.empty((X, Z, Y), dtype=np.float32)

    # 2x3 for kornia
    M = np.asarray(M_pix, dtype=np.float32)
    if M.shape == (3, 3):
        M = M[:2, :]
    elif M.shape != (2, 3):
        raise ValueError(f"Expected 2x3 or 3x3, got {M.shape}")

    dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    M_t = torch.tensor(M, dtype=torch.float32, device=dev)

    for i in range(0, X, batch_size):
        j = min(i + batch_size, X)
        # (B,1,H,W) with H=Z, W=Y
        batch = torch.from_numpy(stack_xzy[i:j]).to(dev).unsqueeze(1).float()  # (B,1,Z,Y)
        Mb = M_t.unsqueeze(0).repeat(j - i, 1, 1)                               # (B,2,3)
        warped = kornia.geometry.transform.warp_affine(
            batch, Mb, dsize=(Z, Y), align_corners=False
        )
        out[i:j] = warped.squeeze(1).cpu().numpy()

    return out

# === INPUT DATA ===
# Assume the following variables are loaded:
# im_nuclei_corr_zyx: shape (Z, Y, X)
# im_colors_corr_zyx: shape (4, Z, Y, X)
# transforms['Transforms'][0][0][ch]: each a 3x3 or object-like affine matrix
# correction_path: output directory

def register_channel_stack_kornia_xyz(stack_xyz, M_pix, device="cuda", batch_size=128):
    """
    Warp each X-slice over the (Y,Z) plane by reusing the ZYX helper.
    """
    zyx = stack_xyz.transpose(2,1,0)  # (Z,Y,X)
    zyx_warp = register_channel_stack_kornia(zyx, M_pix, device=device, batch_size=batch_size)  # (Z,Y,X)
    return zyx_warp.transpose(2,1,0)  # back to (X,Y,Z)

def _unwrap_obj(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.reshape(-1)[0]
    return x

def parse_A2D_list(transforms_mat) -> list:
    """
    Input: dict returned by scipy.io.loadmat(..., struct_as_record=False, squeeze_me=True)
    Output: list of 3x3 float32 affines [A2D_ch0, A2D_ch1, ..., A2D_chN]
    Works whether 'Transforms' is:
      - an object array/list of mat_structs each with .A2D
      - a single mat_struct whose .A2D is an array/cell of matrices
    """
    T = transforms_mat['Transforms']
    T = np.squeeze(T)

    mats = []

    # Case A: array/list of channel structs
    if isinstance(T, (list, tuple)) or (isinstance(T, np.ndarray) and T.dtype == object and T.ndim == 1):
        for node in (T if isinstance(T, (list, tuple)) else list(T)):
            node = _unwrap_obj(node)
            A2D = getattr(node, 'A2D', None)
            if A2D is None:
                A2D = node['A2D']  # rare, but some mats keep dict-style
            A2D = _unwrap_obj(A2D)
            A2D = np.asarray(A2D, dtype=np.float32)
            if A2D.shape == (2,3):
                A2D = np.vstack([A2D, [0,0,1]]).astype(np.float32)
            if A2D.shape != (3,3):
                raise ValueError(f"Unexpected A2D shape {A2D.shape}")
            mats.append(A2D)

    else:
        # Case B: single struct with a field A2D that is a collection of per-channel matrices
        node = _unwrap_obj(T)
        A_all = getattr(node, 'A2D', None)
        if A_all is None:
            A_all = node['A2D']
        A_all = np.squeeze(np.asarray(A_all, dtype=object))
        # normalize to 1D iterable of objects
        if A_all.ndim == 0:
            A_iter = [A_all.item()]
        elif A_all.ndim == 1:
            A_iter = list(A_all)
        elif A_all.shape[0] == 1:
            A_iter = list(A_all[0])
        elif A_all.shape[1] == 1:
            A_iter = list(A_all[:,0])
        else:
            A_iter = list(A_all)

        for a in A_iter:
            a = _unwrap_obj(a)
            M = np.asarray(a, dtype=np.float32)
            if M.shape == (2,3):
                M = np.vstack([M, [0,0,1]]).astype(np.float32)
            if M.shape != (3,3):
                raise ValueError(f"Unexpected A2D shape {M.shape}")
            mats.append(M)

    if not mats:
        raise ValueError("No A2D matrices found in 'Transforms'")

    return mats

def register_channels_xzy(nuclei_xzy_location, colors_cxzy_location, output_location, transforms, device="cuda", batch_size=128):
    """
    nuclei_xzy : (X,Z,Y)
    colors_cxzy: (C,X,Z,Y)
    A2D_list   : list of 3x3 affines, length >= 5 (one per output channel)
    returns    : (5, X, Z, Y) float32
    """
    # skip if already written
    out_root = os.path.join(output_location, "omehans")
    if os.path.exists(os.path.join(out_root, "0", "0", "0")):
        print("Registered volume already exists at", out_root)
        return

    device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    print("Registering on:", device)

    # Load images
    nuclei_xyz = read_omehans(os.path.join(nuclei_xyz_location, "omehans")).compute()
    colors_cxyz = read_omehans(os.path.join(colors_cxyz_location, "omehans")).compute()

    C, Xc, Zc, Yc = colors_cxzy.shape
    # resize nuclei to (Xc,Yc,Zc)
    nuc_res_xzy = resize(nuclei_xzy, (Xc, Zc, Yc), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

    # build output as (5, Z, Y, X) to match your unmixing later
    volReg = np.zeros((5, Xc, Zc, Yc), dtype=np.float32)

    for ch in range(5):
        print(f"Registering ch {ch}")

        #A2D = transforms['Transforms'][0][ch]['A2D'].astype(np.float32)
        A2D = transforms[ch]

        if ch == 0:
            # your 180° rotate on (Z,Y) plane – do it vectorized on XZY:
            stack_xzy = np.rot90(nuc_res_xzy, k=2, axes=(1,2)).copy()  # rotate in (Z,Y)
            warped_xzy = stack_xzy
        else:
            stack_xzy = colors_cxzy[ch-1]
            warped_xzy = warp_stack_kornia_xzy(stack_xzy, A2D, device=device, batch_size=batch_size)

        volReg[ch] = warped_xyz.transpose(2,1,0)
        #volReg[ch] = warped_xzy

    # Save
    try:
        write_omehans(out_root, _clip16(volReg))
    except zarr.errors.ContainsArrayError:
        print(".omehans array already exists")
    try:
        write_zarr(os.path.join(output_location, "zarr"), _clip16(volReg))
    except zarr.errors.ContainsArrayError:
        print(".zarr array already exists")

    print("Saved registered volume at", output_location)

    #return volReg  # (5,Z,Y,X)


##############################################################################
#### Unmixing functions
##############################################################################

# Assume volReg has shape (5, Z, Y, X)
""" ss = volReg.shape  # (5, Z, Y, X)
volReg_flat = volReg.reshape(ss[0], -1)  # shape: (5, Z*Y*X)
volReg_unmixed_flat = M_inv.T @ volReg_flat  # shape: (5, Z*Y*X)
volReg_unmixed = volReg_um_flat.reshape(ss)  """
""" 
def unmix_channels(input_location, output_location, M_inv, batch_size=10_000_000):
    print('Unmixing registered channels')
    vol = read_omehans(os.path.join(input_location, "omehans")).compute().astype(np.float32)  # (C,Z,Y,X)
    C, Z, Y, X = vol.shape
    N = Z * Y * X

    vol_flat = vol.reshape(C, N)
    out_flat = np.empty_like(vol_flat, dtype=np.float32)

    # Prefer solve over explicit inverse if possible:
    # out_flat[:, i:j] = np.linalg.solve(M.T, vol_flat[:, i:j])
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        out_flat[:, i:j] = (M_inv.T @ vol_flat[:, i:j])

    vol_unmixed = out_flat.reshape(C, Z, Y, X)
    #return volReg_unmixed_flat.reshape((C, Z, Y, X))

    root = os.path.join(output_location, "omehans")
    try:
        write_omehans(root, _clip16(vol_unmixed))
    except zarr.errors.ContainsArrayError:
        print(".omehans array already exists")
    write_zarr(os.path.join(output_location, "zarr"), _clip16(vol_unmixed)) """

def unmix_channels(input_location: str, output_location: str, M_inv: np.ndarray, batch_size: int = 10_000_000):
    """
    Reads registered volume from input_location/omehans, ensures shape (C,Z,Y,X),
    applies linear unmixing using M_inv (C x C), writes to output_location/{omehans,zarr}.
    """

    # skip if already written
    out_root = os.path.join(output_location, "omehans")
    if os.path.exists(os.path.join(out_root, "0", "0", "0")):
        print("Unmixed volume already exists at", out_root)
        return

    vol_raw = read_omehans(os.path.join(input_location, "omehans"))
    vol = vol_raw.compute()  # could be ndarray (float/uint16), or object array, or list-like

    # --- normalize to (C,Z,Y,X) ---
    if isinstance(vol, (list, tuple)):
        # list of per-channel arrays
        vol = np.stack([np.asarray(v) for v in vol], axis=0)
    elif isinstance(vol, np.ndarray) and vol.dtype == object:
        # object array (e.g., shape (C,))
        vol = np.stack([np.asarray(v) for v in vol.reshape(-1)], axis=0)
    elif isinstance(vol, np.ndarray) and vol.ndim == 3:
        # single volume, no channel axis
        vol = vol[None, ...]  # (1,Z,Y,X)
    elif isinstance(vol, np.ndarray) and vol.ndim == 4:
        # could be (C,Z,Y,X) or (Z,Y,X,C)
        if vol.shape[0] in (4,5):
            pass  # already (C,Z,Y,X)
        elif vol.shape[-1] in (4,5):
            vol = np.moveaxis(vol, -1, 0)  # (Z,Y,X,C) -> (C,Z,Y,X)
        else:
            raise ValueError(f"Unexpected 4-D volume shape {vol.shape}; cannot locate channel axis.")
    else:
        raise ValueError(f"Unexpected volume type/shape: type={type(vol)}, shape={getattr(vol,'shape',None)}")

    # dtype → float32 for math
    if vol.dtype != np.float32:
        vol = vol.astype(np.float32, copy=False)

    C, Z, Y, X = vol.shape
    if M_inv.shape != (C, C):
        raise ValueError(f"M_inv shape {M_inv.shape} does not match channel count C={C}")

    # --- unmix in chunks along flattened pixels ---
    N = Z * Y * X
    vol_flat = vol.reshape(C, N)
    out_flat = np.empty_like(vol_flat, dtype=np.float32)

    # prefer solve over explicit inverse if you have mixing M: here you supplied M_inv already
    MinvT = M_inv.T.astype(np.float32, copy=False)
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        out_flat[:, i:j] = MinvT @ vol_flat[:, i:j]

    vol_unmixed = out_flat.reshape(C, Z, Y, X)

    # write
    try:
        write_omehans(os.path.join(output_location, "omehans"),  _clip16(vol_unmixed))
    except zarr.errors.ContainsArrayError:
        print(".omehans array already exists")
    try:
        write_zarr(os.path.join(output_location, "zarr"), _clip16(vol_unmixed))
    except zarr.errors.ContainsArrayError:
        print(".zarr array already exists")

    print("Saved unmixed volume at", output_location)


##############################################################################
#### Processing
##############################################################################

# Load correction matrices
simulation_matrices = sio.loadmat('/bil/proj/rf1hillman/2024_07_29_AI7_EH5k_human_finalMarkerCombination_100mm/code_Matlab/SimulationMatrices_equalPower_firstHemibrain')# Fluorophore x Channel matrix normalization
Flch = simulation_matrices['Flch']
Flch_rel = Flch.copy()
Flch_rel = Flch_rel / np.sum(Flch_rel, axis=1, keepdims=True) # Normalize along columns - so each entry (i,j) is the percentage of the signal in channel j coming from fluorophore i
M_inv = np.linalg.inv(Flch_rel)
#transforms = sio.loadmat(
#    '/bil/proj/rf1hillman/2025_06_26_NPBB328_surface_processingMatlabCode_SLURM/NPBB328_colorMerge_transforms.mat',
#    struct_as_record=False, squeeze_me=True
#)
#center = transforms.get('CenterSplitPosition', None)
    


def preprocess_nuclei(omehans_file, nuclei_location_xzy):
    """
    nuclei_location_xzy: folder with nuclei omehans (XZY layout)
    Returns path to laser-corrected nuclei (XZY) folder.
    """
    base = Path(nuclei_location_xzy)
    bg_sub = base.with_name(base.name + "_bg_subtracted")
    bg_sub.mkdir(parents=True, exist_ok=True)

    # BG subtraction (XZY vol, ZY mask)
    #Get z and define mask file names
    base_ome = os.path.basename(omehans_file)  
    root = os.path.splitext(os.path.splitext(base_ome)[0])[0] 
    z_str = infer_z(root)

    corr_BG_nuclei = np.load(os.path.join(os.path.split(omehans_file)[0] ,'correction_files',f"bg_nuclei_mask_z{z_str}.npy"))
    subtract_background(str(omehans_file), str(bg_sub), corr_BG_nuclei)

    # Laser correction (FF + pattern; both ZY)

    # Create output folder
    laser_corr = base.with_name(base.name + "_laser_corrected")
    laser_corr.mkdir(parents=True, exist_ok=True)

    # Load pattern
    pattern_nuclei_path = os.path.join(os.path.split(omehans_file)[0] ,'correction_files', 'laser_patterns', f"laser_pattern_nuclei_mask_z{z_str}_final.npy")
    laser_correction_Nuclei_pattern = np.load(pattern_nuclei_path)
    #with open(pattern_nuclei_path , "rb") as f:
    #    laser_correction_Nuclei_pattern = pickle.load(pattern_nuclei_path, f, protocol=pickle.HIGHEST_PROTOCOL)


    # Apply correction
    laser_correction_nuclei(
        str(bg_sub),
        str(laser_corr),
        laser_correction_Nuclei_pattern
    )
    return str(laser_corr)


def preprocess_colors(omehans_file, colors_location_xzy, nuclei_preprocessed_location):
    """
    colors_location_xyz: folder with *color* OME-HANS at colors_location_xyz/omehans (XZY).
    nuclei_preprocessed_location: output of preprocess_nuclei(...), used for registration.
    Returns path to unmixed colors.
    """
    base = Path(colors_location_xzy)

    # 1) BG subtraction on raw *colors* XZY
    bg_sub = base.with_name(base.name + "_bg_subtracted")
    bg_sub.mkdir(parents=True, exist_ok=True)
    #Get z and define mask file names
    base_ome = os.path.basename(omehans_file)  
    root = os.path.splitext(os.path.splitext(base_ome)[0])[0] 
    z_str = infer_z(root)
    corr_BG_colors = np.load(os.path.join(os.path.split(omehans_file)[0],'correction_files',f"bg_colors_mask_z{z_str}.npy"))
    subtract_background(str(omehans_file), str(bg_sub), corr_BG_colors)

    # 2) LASER CORRECTION on full XZY (single YZ mask for the whole chip → applies to all 4 quadrants)
    color_lcorr = base.with_name(base.name + "_laser_corrected")
    color_lcorr.mkdir(parents=True, exist_ok=True)

    # Load pattern
    pattern_colors_path = os.path.join(os.path.split(omehans_file)[0] ,'correction_files', 'laser_patterns', f"laser_pattern_colors_mask_z{z_str}_final.npy")
    laser_correction_Colors_pattern = np.load(pattern_colors_path)
    #with open(pattern_colors_path , "rb") as f:
    #    laser_correction_Colors_pattern = pickle.load(pattern_colors_path, f, protocol=pickle.HIGHEST_PROTOCOL)

    transforms = sio.loadmat("/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Matlab_info/NPBB328_colorMerge_transforms.mat")
    center_raw = transforms['CenterSplitPosition'] 
    center_arr = np.atleast_1d(center_raw).astype(int).ravel()
    centerPos = (int(center_arr[0]), int(center_arr[1]))  

    laser_correction_colors(
        str(bg_sub),
        str(color_lcorr),
        laser_correction_Colors_pattern,
        centerPos
    )

    """# 3) COLOR SPLIT (after correction) → (C=4, X, Zc, Yc)
    #    Use MATLAB-provided center if present.
    transforms = sio.loadmat(
        '/bil/proj/rf1hillman/2025_06_26_NPBB328_surface_processingMatlabCode_SLURM/NPBB328_colorMerge_transforms.mat',
        struct_as_record=False, squeeze_me=True
    )

    center = transforms.get('CenterSplitPosition', None)
    if center is not None and isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
        center_pos = (int(center[0]), int(center[1]))  # (Z, Y)
    else:
        center_pos = None

    color_split = base.with_name(base.name + "_color_split")
    color_split.mkdir(parents=True, exist_ok=True)
    split_color_channels(str(color_lcorr), str(color_split), center_pos=center_pos)

    # 4) REGISTRATION: nuclei (XZY, preprocessed) + colors (CXZY) → write (5,X,Z,Y) into *_registered
    registered = base.with_name(base.name + "_registered")
    registered.mkdir(parents=True, exist_ok=True)

    A2D_list = parse_A2D_list(transforms) 

    volReg = register_channels_xzy(
        nuclei_xzy_location=str(nuclei_preprocessed_location),  # expects omehans inside
        colors_cxzy_location=str(color_split),                  # expects omehans inside
        output_location=str(registered),
        transforms=A2D_list,
        device="cuda",
        batch_size=128
    )

    # 5) UNMIX (reads from *_registered/omehans and writes to *_unmixed)
    unmixed = base.with_name(base.name + "_unmixed")
    unmixed.mkdir(parents=True, exist_ok=True)
    unmix_channels(str(registered), str(unmixed), M_inv, batch_size=10_000_000) """

    return str(color_lcorr)

