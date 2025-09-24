import os
import sys
from glob import glob

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

from holis_pipeline import settings
from holis_pipeline.preprocessing_functions import read_data_file
from holis_pipeline.read_data import read_fli_as_zarr
from holis_pipeline.utils.zarr_related import read_omehans, write_omehans, write_zarr


##############################################################################
#### Load correction matrices
##############################################################################

# These should be global constants

correction_path = "/bil/proj/rf1hillman/results/2025_06_15_NPBB328_surface_corrections/"

corr_BG_nuclei = np.load(correction_path + "Corr_BG_nuclei.npy")
corr_BG_colors = np.load(correction_path + "Corr_BG_colors.npy")
ff_nuclei_norm = np.load(correction_path + "FF_nuc_norm.npy")
ff_colors_norm = np.load(correction_path + "FF_colors_norm.npy")
laser_correction_Nuclei_pattern = np.load(correction_path + "Laser_correction_Nuclei_pattern.npy")
laser_correction_Colors_pattern = np.load(correction_path + "Laser_correction_Colors_pattern.npy")

##############################################################################
#### Background and laser pattern correction functions
##############################################################################

def _clip16(a):
    return np.clip(a, 0, 65535).astype(np.uint16)

def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-6):
    return a / (b + eps)

def subtract_background(input_location, output_location, bg_yz: np.ndarray):

    """
    Input: data .omehans (X,Y,Z), dark frames (.npy) (Y,Z) (1 file per nuclei+colors fli pair), dark frames will be broadcast across X 
    Output: .omehans (X,Y,Z)
    """

    omehans_root = os.path.join(output_location, 'omehans')
    if os.path.exists(os.path.join(omehans_root, '0', '0', '0')):
        print("BG already subtracted previously")
        return

    print("Subtracting BG...")
    vol_xyz = read_omehans(input_location).compute().astype(np.float32)  # (X,Y,Z)
    X, Y, Z = vol_xyz.shape

    # Optional save of the original image
    # tifffile.imwrite(os.path.join(output_location, "original.tif"), img.astype('uint16'))
    #write_zarr(os.path.join(output_location, 'original_zarr'), img.astype('uint16'))

    if bg_yz.shape != (Y, Z):
        raise ValueError(f"BG mask must be (Y,Z)={Y,Z}, got {bg_yz.shape}")

    bg = bg_yz.astype(np.float32) - 1024.0  # your original -1024
    out = vol_xyz - bg  # broadcast over X
    np.maximum(out, 0, out)

    # Save
    try:
        write_omehans(omehans_root, _clip16(out))
    except zarr.errors.ContainsArrayError:
        print(".omehans array already exists")
    try:
        write_zarr(os.path.join(output_location, 'zarr'), _clip16(out))
    except zarr.errors.ContainsArrayError:
        print(".zarr array already exists")

    # tifffile.imwrite(os.path.join(output_location, "bg_subtracted.tif"), image_np_array_bgSubtracted.astype('uint16'))

    print("Saved BG-subtracted file")

    #return image_np_array_bgSubtracted



def laser_correction_nuclei(input_location, output_location, ff_yz, pattern_yz):
    """
    Input:
        - flattened .omehans (X,Y,Z)
        - laser pattern matrix (Y,Z)
        - absorption matrix (n_fluorophores, n_lasers) - (5x4) - first row   for nuclei
        - mixing (fluorescence) matrix (n_fluorophores, n_channels) - (5x5) - first column for nuclei
    Output: corrected .omehans the same shape as input
    """
    #data = tifffile.imread(os.path.join(input_location, "bg_subtracted.tif"))

    root = os.path.join(output_location, 'omehans')
    if os.path.exists(os.path.join(root, '0', '0', '0')):
        print("Laser pattern corrected (nuclei) previously")
        return

    print("Laser correction (nuclei): XYZ vol ÷ YZ FF ÷ YZ pattern")
    img = read_omehans(os.path.join(input_location, "omehans")).compute().astype(np.float32)  # (X,Y,Z)
    X, Y, Z = img.shape
    if ff_yz.shape != (Y, Z) or pattern_yz.shape != (Y, Z):
        raise ValueError("FF and pattern must be (Y,Z)")

    ff = ff_yz.astype(np.float32)
    pat = pattern_yz.astype(np.float32)

    ff_corr = _safe_div(img, ff)
    corr = _safe_div(ff_corr, pat)

    # Save
    try:
        write_omehans(root, _clip16(corr))
    except zarr.errors.ContainsArrayError:
        print(".omehans already exists")
    try:
        write_zarr(os.path.join(output_location, 'zarr'), _clip16(corr))
    except zarr.errors.ContainsArrayError:
        print(".zarr already exists")
    #tifffile.imwrite(os.path.join(output_location, 'laser_corrected.tif'), np.round(corrected_data).astype('uint16'))
    print("Saved laser-pattern-corrected (nuclei) file")



def laser_correction_colors(input_location, output_location, ff_cyz, pattern_cyz, chunk_x: Optional[int] = None, target_bytes: int = 512 * 1024 * 1024, center_crop: int = 100,):
    """
    Input:
        - flattened .omehans (X,Y,Z)
        - ff correction (Y,Z)
        - laser pattern matrix (Y,Z)
        - absorption matrix (n_fluorophores, n_lasers) - (5x4) - first row   for nuclei
        - mixing (fluorescence) matrix (n_fluorophores, n_channels) - (5x5) - first column for nuclei
    Output: corrected .omehans the same shape as input
    """

    #data = tifffile.imread(os.path.join(input_location, "color_split.tif"))

    root = os.path.join(output_location, 'omehans')
    if os.path.exists(os.path.join(root, '0', '0', '0')):
        print("Color laser correction already done")
        return

    print("Laser correction (colors): XYZ ÷ YZ ÷ YZ")
    img = read_omehans(os.path.join(input_location, "omehans")).compute().astype(np.float32)  # (X,Y,Z)
    X, Y, Z = img.shape

    def to_yz(mask: np.ndarray, name: str) -> np.ndarray:
        m = np.asarray(mask)
        if m.ndim == 2:
            if m.shape != (Y, Z):
                raise ValueError(f"{name} must be (Y,Z)={(Y,Z)}, got {m.shape}")
            out = m.astype(np.float32, copy=False)
        elif m.ndim == 3:
            if m.shape[1:] != (Y, Z):
                raise ValueError(f"{name} (C,Y,Z) must match (Y,Z)={(Y,Z)}, got {m.shape}")
            out = np.nanmean(m.astype(np.float32, copy=False), axis=0)  # average over C → (Y,Z)
        else:
            raise ValueError(f"{name} must be (Y,Z) or (C,Y,Z); got {m.shape}")

        # Optional normalization by interior max (like your 100:-100 crop logic)
        if center_crop > 0:
            y0, y1 = center_crop, max(center_crop, Y - center_crop)
            z0, z1 = center_crop, max(center_crop, Z - center_crop)
            if y1 > y0 and z1 > z0:
                denom = float(np.nanmax(out[y0:y1, z0:z1]))
                if denom > 0:
                    out = _safe_div(out, denom)
        return out

    ff = to_yz(ff_cyz, "ff_cyz")
    pat = to_yz(pattern_cyz, "pattern_cyz")

    # keep non-negatives (BG-subtracted data can dip slightly below 0)
    np.maximum(img, 0, out=img)

    out = np.empty_like(img, dtype=np.float32)

    # FAST PATH: try full-broadcast in one go
    try:
        out = _safe_div(_safe_div(img, ff), pat)  # broadcasts (Y,Z) over X
    except MemoryError:
        # FALLBACK: chunk along X to reduce peak memory
        if chunk_x is None:
            # rough heuristic: bytes per X-slice ≈ Y*Z*4
            bytes_per_x = Y * Z * 4
            chunk_x = max(1, int(target_bytes // bytes_per_x))

        for i in range(0, X, chunk_x):
            j = min(i + chunk_x, X)
            block = img[i:j]  # (x, Y, Z) view
            block = _safe_div(_safe_div(block, ff), pat)
            out[i:j] = block

    # Save
    try:
        write_omehans(root, _clip16(out))
    except zarr.errors.ContainsArrayError:
        print(".omehans already exists")
    try:
        write_zarr(os.path.join(output_location, 'zarr'), _clip16(out))
    except zarr.errors.ContainsArrayError:
        print(".zarr already exists")
    print("Saved laser-pattern-corrected colors")


##############################################################################
#### Color split function
##############################################################################

def split_color_channels(input_location, output_location, center_pos=None):
    """
    Splits a 3D image splitter input (X,Y,Z) into 4 cropped channels.
    Returns a 4D array with shape (C=4, X, Yc, Zc) cropped to common (Yc,Zc)
    """

    # data_temp = tifffile.imread(os.path.join(input_location, "bg_subtracted.tif"))

    root = os.path.join(output_location, 'omehans')
    if os.path.exists(os.path.join(root, '0', '0', '0')):
        print("Colors already split previously")
        return

    print("Splitting color channels on (Y,Z) axes...")
    vol = read_omehans(os.path.join(input_location, "omehans")).compute().astype(np.float32)  # (X,Y,Z)
    X, Y, Z = vol.shape
    cy = center_pos[0] if center_pos else Y // 2
    cz = center_pos[1] if center_pos else Z // 2

    ch1 = vol[:, :cy, :cz]   # top-left in (Y,Z)
    ch2 = vol[:, cy:, :cz]
    ch3 = vol[:, :cy, cz:]
    ch4 = vol[:, cy:, cz:]

    # crop to common min size along Y/Z
    Yc = min(ch1.shape[1], ch2.shape[1], ch3.shape[1], ch4.shape[1])
    Zc = min(ch1.shape[2], ch2.shape[2], ch3.shape[2], ch4.shape[2])
    chs = [c[:, :Yc, :Zc] for c in (ch1, ch2, ch3, ch4)]
    out = np.stack(chs, axis=0)  # (4, X, Yc, Zc)

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
    try:
        write_omehans(root, _clip16(out))
    except zarr.errors.ContainsArrayError:
        print(".omehans already exists")
    write_zarr(os.path.join(output_location, "zarr"), _clip16(out))
    print("Saved color split (C=4, X, Y, Z)")


##############################################################################
#### Registration functions
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

""" def register_channels(input_location_nuclei, input_location_colors, output_location, transform_matrix, device="cuda", batch_size=128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Load images
    nuclei_dask_zarray = read_omehans(os.path.join(input_location_nuclei, "omehans"))
    im_nuclei_corr_zyx = nuclei_dask_zarray.compute()

    colors_dask_zarray = read_omehans(os.path.join(input_location_colors, "omehans"))
    im_colors_corr_zyx_split = colors_dask_zarray.compute()

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

    # Save subset of result
    print('Saving registered volume')
    #np.save(correction_path + "volReg.npy", volReg[:, :, :, 15000:16000])
    #tifffile.imwrite(
    #    correction_path + "volReg.tiff",
    #    volReg[:, :, :, 15000:16000].astype(np.float32),  # Ensure it's a supported type
    #    imagej=True  # Optional: if you want ImageJ compatibility
    #)

    try:
        write_omehans(os.path.join(output_location, "omehans"), volReg.astype('uint16'))
    except zarr.errors.ContainsArrayError:
        print(".omehans array already exists")
    write_zarr(os.path.join(output_location, "zarr"), volReg.astype('uint16')) """

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

def register_channels_xyz(nuclei_xyz_location, colors_cxyz_location, output_location, transforms, device="cuda", batch_size=128):
    """
    nuclei_xyz:  (X,Y,Z)
    colors_cxyz: (C,X,Y,Z)
    returns volReg_czyx to keep downstream unmixing unchanged
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

    C, Xc, Yc, Zc = colors_cxyz.shape
    # resize nuclei to (Xc,Yc,Zc)
    nuc_res_xyz = resize(nuclei_xyz, (Xc, Yc, Zc), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

    # build output as (5, Z, Y, X) to match your unmixing later
    volReg = np.zeros((5, Zc, Yc, Xc), dtype=np.float32)

    for ch in range(5):
        print(f"Registering ch {ch}")

        #A2D = transforms['Transforms'][0][ch]['A2D'].astype(np.float32)
        A2D = transforms[ch]

        if ch == 0:
            # your 180° rotate on (Y,Z) plane – do it vectorized on XYZ:
            stack_xyz = np.rot90(nuc_res_xyz, k=2, axes=(1,2)).copy()  # rotate in (Y,Z)
        else:
            stack_xyz = colors_cxyz[ch-1]

        if ch == 0:
            warped_xyz = stack_xyz  # if nuclei need no affine warp
        else:
            warped_xyz = register_channel_stack_kornia_xyz(stack_xyz, A2D, device=device, batch_size=batch_size)

        # move to (Z,Y,X) for downstream consistency
        volReg[ch] = warped_xyz.transpose(2,1,0)

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

def preprocess_nuclei(spool_file, nuclei_location_xyz):
    """
    nuclei_location_xyz: folder with nuclei omehans (XYZ layout)
    Returns path to laser-corrected nuclei (XYZ) folder.
    """
    base = Path(nuclei_location_xyz)
    bg_sub = base.with_name(base.name + "_bg_subtracted")
    bg_sub.mkdir(parents=True, exist_ok=True)

    # BG subtraction (XYZ vol, YZ mask)
    subtract_background(str(base), str(bg_sub), corr_BG_nuclei)

    # Laser correction (FF + pattern; both YZ)
    laser_corr = base.with_name(base.name + "_laser_corrected")
    laser_corr.mkdir(parents=True, exist_ok=True)
    laser_correction_nuclei(
        str(bg_sub),
        str(laser_corr),
        ff_nuclei_norm,
        laser_correction_Nuclei_pattern
    )
    return str(laser_corr)


def preprocess_colors(spool_file, colors_location_xyz, nuclei_preprocessed_location):
    """
    colors_location_xyz: folder with *color* OME-HANS at colors_location_xyz/omehans (XYZ).
    nuclei_preprocessed_location: output of preprocess_nuclei(...), used for registration.
    Returns path to unmixed colors.
    """
    base = Path(colors_location_xyz)

    # 1) BG subtraction on raw *colors* XYZ
    bg_sub = base.with_name(base.name + "_bg_subtracted")
    bg_sub.mkdir(parents=True, exist_ok=True)
    subtract_background(str(base), str(bg_sub), corr_BG_colors)

    # 2) LASER CORRECTION on full XYZ (single YZ mask for the whole chip → applies to all 4 quadrants)
    color_lcorr = base.with_name(base.name + "_laser_corrected")
    color_lcorr.mkdir(parents=True, exist_ok=True)
    laser_correction_colors(
        str(bg_sub),
        str(color_lcorr),
        ff_colors_norm,
        laser_correction_Colors_pattern
    )

    # 3) COLOR SPLIT (after correction) → (C=4, X, Yc, Zc)
    #    Use MATLAB-provided center if present.
    transforms = sio.loadmat(
        '/bil/proj/rf1hillman/2025_06_26_NPBB328_surface_processingMatlabCode_SLURM/NPBB328_colorMerge_transforms.mat',
        struct_as_record=False, squeeze_me=True
    )

    center = transforms.get('CenterSplitPosition', None)
    if center is not None and isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
        center_pos = (int(center[0]), int(center[1]))  # (Y, Z)
    else:
        center_pos = None

    color_split = base.with_name(base.name + "_color_split")
    color_split.mkdir(parents=True, exist_ok=True)
    split_color_channels(str(color_lcorr), str(color_split), center_pos=center_pos)

    # 4) REGISTRATION: nuclei (XYZ, preprocessed) + colors (CXYZ) → write (5,Z,Y,X) into *_registered
    registered = base.with_name(base.name + "_registered")
    registered.mkdir(parents=True, exist_ok=True)

    A2D_list = parse_A2D_list(transforms) 

    volReg = register_channels_xyz(
        nuclei_xyz_location=str(nuclei_preprocessed_location),  # expects omehans inside
        colors_cxyz_location=str(color_split),                  # expects omehans inside
        output_location=str(registered),
        transforms=A2D_list,
        device="cuda",
        batch_size=128
    )

    # 5) UNMIX (reads from *_registered/omehans and writes to *_unmixed)
    unmixed = base.with_name(base.name + "_unmixed")
    unmixed.mkdir(parents=True, exist_ok=True)
    unmix_channels(str(registered), str(unmixed), M_inv, batch_size=10_000_000)

    return str(unmixed)

