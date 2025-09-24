import itertools
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import dask.array as da
import tifffile as tiff
import zarr
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import DBSCAN
from stack_to_multiscale_ngff.archived_nested_store import Archived_Nested_Store
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store

from holis_pipeline.utils.chunks import get_chunk_indices, get_origin_coords
from holis_pipeline.settings import *
from holis_pipeline.utils.zarr_related import *
from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage.segmentation import expand_labels
from skimage.measure import label
from holis_pipeline.unchunk_data import combine_masks

def get_chunk(ind, lazy_data, yz_ratio=1.0, yx_ratio=1.0, order=1, preserve_range=True):
    """
    Extract a chunk from lazy_data shaped (C, Z, Y, X) using slice tuple ind=(z_s, y_s, x_s),
    then resize Z by yz_ratio and X by yx_ratio (Y unchanged). Channels are preserved.
    """
    # Grab the chunk across all channels
    chunk = np.asarray(lazy_data[:, ind[0], ind[1], ind[2]])
    chunk_dtype = chunk.dtype
    C, Z, Y, X = chunk.shape

    Z2 = int(round(Z * yz_ratio))
    Y2 = Y
    X2 = int(round(X * yx_ratio))

    # Resize without mixing channels
    try:
        # skimage >= 0.19
        chunk_resized = resize(
            chunk, (Z2, Y2, X2),
            order=order, preserve_range=preserve_range, anti_aliasing=False,
            channel_axis=0  # channels-first
        )
    except TypeError:
        # Fallback: resize per-channel
        per_ch = [
            resize(chunk[c], (Z2, Y2, X2),
                   order=order, preserve_range=preserve_range, anti_aliasing=False)
            for c in range(C)
        ]
        chunk_resized = np.stack(per_ch, axis=0)

    # If your data were already in native dtype/range, keep dtype.
    # If they were normalized floats, optionally scale back (example for uint16):
    if np.issubdtype(chunk_dtype, np.integer):
        out = np.rint(chunk_resized).astype(chunk_dtype)
    else:
        # Optional: scale back to uint16 if you know values are 0..1
        # out = (chunk_resized * 65535).astype(np.uint16)
        out = chunk_resized.astype(chunk_dtype)

    return out


def get_box_slicing(z, y, x, img_shape, box_size):
    """
    Extract cube like cellfinder does. Subsequent zoom needed.
    """
    #print('img_shape', img_shape)
    z_left = int(round(max([z - box_size[0] // 2, 0])))
    z_right = int(round(min([z + box_size[0] // 2, img_shape[0]])))
    y_left = int(round(max([y - box_size[1] // 2, 0])))
    y_right = int(round(min([y_left + box_size[1], img_shape[1]])))
    x_left = int(round(max([x - box_size[2] // 2, 0])))
    x_right = int(round(min([x_left + box_size[2], img_shape[2]])))
    return slice(z_left, z_right, None), slice(y_left, y_right, None), slice(x_left, x_right, None)


def run_dbscan_on_chunk(df):
    points = df[["axis-0", "axis-1", "axis-2"]].to_numpy()
    print("Points shape", points.shape)

    eps = 2
    min_samples = 1

    # create a DBSCAN object
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # fit the points to the model
    dbscan.fit(points)

    # get the cluster assignments for each point
    labels = dbscan.labels_

    # get the unique cluster labels
    cluster_labels = np.unique(labels)
    print("Total clusters:", len(cluster_labels))

    # calculate the centroid of each cluster
    cluster_centroids = []
    for label in cluster_labels:
        # get the points in the current cluster
        points_in_cluster = points[labels == label]

        # calculate the mean of the points in the cluster to get the centroid
        centroid = np.mean(points_in_cluster, axis=0)

        # add the centroid to the list of cluster centroids
        cluster_centroids.append(centroid)

    cluster_centroids = np.array(cluster_centroids)
    df = pd.DataFrame()
    df['axis-0'] = cluster_centroids[:, 0]
    df['axis-1'] = cluster_centroids[:, 1]
    df['axis-2'] = cluster_centroids[:, 2]
    return df


def remove_background_spots(points, nuclei_chunk_shape):
    # print("Removing BG")
    # print("Converting to numpy")
    detected_cells_np = np.floor(points).astype(int)
    cells_binary = np.zeros(nuclei_chunk_shape, dtype=np.uint8)
    print("Converting to binary", cells_binary.shape)
    np.put(cells_binary, np.ravel_multi_index(detected_cells_np.T, nuclei_chunk_shape), 1)
    print("Multiplying by mask")
    mask_folder = os.path.join(str(Path(spectral_info_folder).parent.parent), 'scale_x', 'mask_resized')
    mask_stack = tifffile.imread(os.path.join(mask_folder, f"chunk_{str(number).zfill(5)}.tif"))
    mask_stack = resize(mask_stack, nuclei_chunk_shape)
    cells_filtered = cells_binary * mask_stack
    print("Converting to coords")
    nz = np.nonzero(cells_filtered)
    zipped_nz = list(zip(*nz))
    filtered_cells_np = np.asarray(zipped_nz)
    print("filtered_cells_np", filtered_cells_np.shape)
    print("Generating csv")
    filtered_cells_df = pd.DataFrame(columns=['axis-0', 'axis-1', 'axis-2'])
    if filtered_cells_np.shape[0] > 0:
        filtered_cells_df['axis-0'] = list(filtered_cells_np[:, 0])
        filtered_cells_df['axis-1'] = list(filtered_cells_np[:, 1])
        filtered_cells_df['axis-2'] = list(filtered_cells_np[:, 2])
    print("Saving coords to csv")
    filtered_cells_df.to_csv(os.path.join(dbscan_folder, f"filtered_chunk_{str(number).zfill(5)}.csv"))
    return filtered_cells_df, filtered_cells_np


def remove_background(number, nuclei_chunk_shape):
    """
    Multiply nuclei segmentation mask for a specific chunk by its foreground/background mask.
    """
    fg_mask_folder = os.path.join(OUTPUT_DIR, 'scale_x', 'mask_resized')
    fg_mask_stack = tifffile.imread(os.path.join(fg_mask_folder, f"chunk_{str(number).zfill(5)}.tif"))
    fg_mask_stack = resize(fg_mask_stack, nuclei_chunk_shape).astype(np.uint8)
    nuclei_mask_path = os.path.join(OUTPUT_DIR, f'scale_{SCALE}', 'detection_masks', f"mask_chunk_{str(number).zfill(5)}.tif")
    nuclei_mask = tifffile.imread(nuclei_mask_path)
    nuclei_mask *= fg_mask_stack
    tifffile.imwrite(nuclei_mask_path, nuclei_mask)
    return np.any(nuclei_mask)


def extract_box_intensities_resolution_mismatch(points):
    """
    Extraction of boxes around nuclei centroids.
    The boxes are extracted separately for nuclei channel and color channels, with the assumption that
    resolution for nuclei channel is different (better) than resolution of color channels.
    """
    points[:, 2] = points[:, 2] / xy_factor
    points[:, 0] = points[:, 0] / zy_factor

    points_nuclei = np.round(points).astype(int)
    averages_nuclei = np.empty(shape=(points_nuclei.shape[0], 1), dtype=np.float32)
    for ip, point in enumerate(list(points_nuclei)):
        # extract box around each point
        slice_z, slice_y, slice_x = get_box_slicing(point[0], point[1], point[2], nuclei_chunk.shape[-3:], nuclei_box_size)
        box = nuclei_chunk[slice_z, slice_y, slice_x]
        # get the box average for each channel
        box_avgs = np.mean(box)
        averages_nuclei[ip, :] = box_avgs

    points_absolute = np.empty_like(points)
    chunk_origin = origin_coords[number]
    points_absolute[:, 0] = points[:, 0] + chunk_origin[0]  # nuclei channel
    points_absolute[:, 1] = points[:, 1] + chunk_origin[1]
    points_absolute[:, 2] = points[:, 2] + chunk_origin[2]
    points_um = np.empty_like(points_absolute)
    points_um[:, 0] = points_absolute[:, 0] * NUCLEI_RESOLUTION[0]  # nuclei channel
    points_um[:, 1] = points_absolute[:, 1] * NUCLEI_RESOLUTION[1]
    points_um[:, 2] = points_absolute[:, 2] * NUCLEI_RESOLUTION[2]
    print("=====points_um", points_um.dtype)

    z_min_um = np.min(points_um[:, 0])  # both channels
    z_max_um = np.max(points_um[:, 0])
    y_min_um = np.min(points_um[:, 1])
    y_max_um = np.max(points_um[:, 1])
    x_min_um = np.min(points_um[:, 2])
    x_max_um = np.max(points_um[:, 2])
    print("Z limits um", z_min_um, z_max_um)
    print("Y limits um", y_min_um, y_max_um)
    print("X limits um", x_min_um, x_max_um)
    color_info_z_min_px = int(round(z_min_um / COLOR_RESOLUTION[0] - color_info_box_size[0] // 2))
    if color_info_z_min_px < 0:
        color_info_z_min_px = 0
    color_info_z_max_px = int(round(z_max_um / COLOR_RESOLUTION[0] + color_info_box_size[0] // 2 + 1))  # TODO check
    if color_info_z_max_px > (color_info_shape[0] - 1):
        color_info_z_max_px = color_info_shape[0] - 1
    color_info_y_min_px = int(round(y_min_um / COLOR_RESOLUTION[1] - color_info_box_size[1] // 2))
    if color_info_y_min_px < 0:
        color_info_y_min_px = 0
    color_info_y_max_px = int(round(y_max_um / COLOR_RESOLUTION[1] + color_info_box_size[1] // 2 + 1))  # TODO check
    if color_info_y_max_px > (color_info_shape[1] - 1):
        color_info_y_max_px = color_info_shape[1] - 1
    color_info_x_min_px = int(round(x_min_um / COLOR_RESOLUTION[2] - color_info_box_size[2] // 2))
    if color_info_x_min_px < 0:
        color_info_x_min_px = 0
    color_info_x_max_px = int(round(x_max_um / COLOR_RESOLUTION[2] + color_info_box_size[2] // 2 + 1))  # TODO check
    if color_info_x_max_px > (color_info_shape[2] - 1):
        color_info_x_max_px = color_info_shape[2] - 1
    print("Z limits px", color_info_z_min_px, color_info_z_max_px)
    print("Y limits px", color_info_y_min_px, color_info_y_max_px)
    print("X limits px", color_info_x_min_px, color_info_x_max_px)
    color_info_chunk = color_info_zarray[
        0,
        :,
        color_info_z_min_px: color_info_z_max_px,
        color_info_y_min_px: color_info_y_max_px,
        color_info_x_min_px: color_info_x_max_px
    ]
    color_info_chunk_origin = [color_info_z_min_px, color_info_y_min_px, color_info_x_min_px]
    averages = []  # 4 colors
    dimensions_um = list(np.array(color_info_shape) * np.array(COLOR_RESOLUTION))
    print("Color channels dimensions um", dimensions_um)
    failed_point_indices = []
    for ip, point_um in enumerate(list(points_um)):
        if np.any(point_um < 0):
            print("Negative coordinates um", ip, point_um)
            failed_point_indices.append(ip)
            continue
        if (point_um[0] > dimensions_um[0]) or (point_um[1] > dimensions_um[1]) or (point_um[2] > dimensions_um[2]):
            print("Point out of bounds", ip, point_um)
            failed_point_indices.append(ip)
            continue
        try:
            # convert point to pixels (absolute) in color channels space
            point_px = np.round((point_um / np.array(COLOR_RESOLUTION))).astype(int)  # color channels
            if np.any(point_px < 0):
                raise RuntimeError("Negative coordinates")
            # get point coords relative to chunk origin
            point = point_px - np.array(color_info_chunk_origin)  # coords within chunk
            # extract box around each point
            slice_z, slice_y, slice_x = get_box_slicing(point[0], point[1], point[2], color_info_chunk.shape[-3:], color_info_box_size)
            box = color_info_chunk[:, slice_z, slice_y, slice_x]
            # get the box average for each channel
            box_avgs = np.mean(box, axis=(1,2,3))
            averages.append(box_avgs)
        except Exception as e:
            print("ERROR:", e)
            failed_point_indices.append(ip)
    averages = np.array(averages)
    print("Averages", averages.shape)
    print("Failed to extract:", len(failed_point_indices))
    # save the box average for each channel in the csv file
    spectral_df = pd.DataFrame()
    spectral_df['index'] = list(range(filtered_df.shape[0] - len(failed_point_indices)))
    z_points = points_um[:, 0]
    z_points = np.ma.array(z_points, mask=False)
    z_points.mask[failed_point_indices] = True
    z_points = z_points.compressed()
    y_points = points_um[:, 1]
    y_points = np.ma.array(y_points, mask=False)
    y_points.mask[failed_point_indices] = True
    y_points = y_points.compressed()
    x_points = points_um[:, 2]
    x_points = np.ma.array(x_points, mask=False)
    x_points.mask[failed_point_indices] = True
    x_points = x_points.compressed()
    averages_nuclei = np.ma.array(averages_nuclei, mask=False)
    averages_nuclei.mask[failed_point_indices] = True
    averages_nuclei = averages_nuclei.compressed()
    print("Total points", points_um.shape[0])
    print("Good points", z_points.shape)
    print("Good points", y_points.shape)
    print("Good points", x_points.shape)
    spectral_df['axis-0'] = z_points  # absolute
    spectral_df['axis-1'] = y_points
    spectral_df['axis-2'] = x_points
    spectral_df['color1'] = averages_nuclei
    spectral_df['color2'] = averages[:, 0]
    spectral_df['color3'] = averages[:, 1]
    spectral_df['color4'] = averages[:, 2]
    spectral_df['color5'] = averages[:, 3]
    return spectral_df


def extract_box_intensities(points, nuclei_box_size):
    """
    Extraction of boxes around nuclei centroids.
    The boxes are extracted assuming the same resolution and shape of nuclei and color channels.
    """
    points[:, 2] = points[:, 2] / xy_factor
    points[:, 0] = points[:, 0] / zy_factor

    points_nuclei = np.round(points).astype(int)
    points_colors = np.round(points).astype(int)
    points_colors[:, 0] += nuclei_box_size[0] // 2
    points_colors[:, 1] += nuclei_box_size[1] // 2
    points_colors[:, 2] += nuclei_box_size[1] // 2
    color_locations = list(range(2, 6))
    averages_nuclei = np.empty(shape=(points_nuclei.shape[0], 1, 1), dtype=np.float32)
    averages_colors = np.empty(shape=(points_colors.shape[0], len(color_locations), 1), dtype=np.float32)

    color_info_z_min_px = ind[0].start - nuclei_box_size[0] // 2
    if color_info_z_min_px < 0:
        color_info_z_min_px = 0
    color_info_z_max_px = ind[0].stop + nuclei_box_size[0] // 2 + 1
    if color_info_z_max_px > (color_info_shape[0] - 1):
        color_info_z_max_px = color_info_shape[0] - 1
    color_info_y_min_px = ind[1].start - nuclei_box_size[1] // 2
    if color_info_y_min_px < 0:
        color_info_y_min_px = 0
    color_info_y_max_px = ind[1].stop + nuclei_box_size[1] // 2 + 1
    if color_info_y_max_px > (color_info_shape[1] - 1):
        color_info_y_max_px = color_info_shape[1] - 1
    color_info_x_min_px = ind[2].start - nuclei_box_size[2] // 2
    if color_info_x_min_px < 0:
        color_info_x_min_px = 0
    color_info_x_max_px = ind[2].stop + nuclei_box_size[2] // 2 + 1
    if color_info_x_max_px > (color_info_shape[2] - 1):
        color_info_x_max_px = color_info_shape[2] - 1
    color_info_chunk = color_info_zarray[
        0,
        :,
        color_info_z_min_px: color_info_z_max_px,
        color_info_y_min_px: color_info_y_max_px,
        color_info_x_min_px: color_info_x_max_px
    ]

    for ip, point in enumerate(list(points_nuclei)):
        # extract boxes around each point
        slice_z, slice_y, slice_x = get_box_slicing(point[0], point[1], point[2], nuclei_chunk.shape[-3:], nuclei_box_size)
        box = nuclei_chunk[slice_z, slice_y, slice_x]
        # get the box average for each channel
        box_avgs = np.mean(box)
        averages_nuclei[ip, :, 0] = box_avgs
    print("averages_nuclei", averages_nuclei.shape)

    for ch, color_info_location in enumerate(color_locations):  # works only if resolution of nuclei and colors is the same
        stripe_file = os.path.join(color_info_location, f"stripe_y{str(stripe_number).zfill(2)}.tiff")
        print("reading", stripe_file)
        for ip, point in enumerate(list(points_colors)):
            # extract boxes around each point
            slice_z, slice_y, slice_x = get_box_slicing(point[0], point[1], point[2], color_info_chunk.shape[-3:], nuclei_box_size)
            box = color_info_chunk[slice_z, slice_y, slice_x]
            # get the box average for each channel
            box_avgs = np.mean(box)
            averages_colors[ip, ch, 0] = box_avgs
    print("averages_colors", averages_colors.shape)

    spectral_df = pd.DataFrame()
    chunk_origin = origin_coords[number]
    spectral_df['axis-0'] = points_nuclei[:, 0] + chunk_origin[0]  # global, px
    spectral_df['axis-1'] = points_nuclei[:, 1] + chunk_origin[1]  # global, px
    spectral_df['axis-2'] = points_nuclei[:, 2] + chunk_origin[2]  # global, px
    for ib, cube_width in enumerate(cube_widths):
        spectral_df[f'color1-{cube_width}um-box'] = averages_nuclei[:, :, ib]
        spectral_df[f'color2-{cube_width}um-box'] = averages_colors[:, 0, ib]
        spectral_df[f'color3-{cube_width}um-box'] = averages_colors[:, 1, ib]
        spectral_df[f'color4-{cube_width}um-box'] = averages_colors[:, 2, ib]
        spectral_df[f'color5-{cube_width}um-box'] = averages_colors[:, 3, ib]
    return spectral_df


def get_props(mask, image):
    print("===============mask image shape", mask.shape, image.shape)
    # Compute the connected components of the binary mask
    labels = measure.label(mask)

    # Compute properties of the above found components
    region_props = pd.DataFrame(
        measure.regionprops_table(
            labels,
            intensity_image=image,
            properties=['label',
                        'centroid',
                        'coords']
        )
    )
    region_props = region_props.round(2)
    region_props['label'] = list(range(1, labels.max() + 1))
    return region_props


def blowup_vol(coords_set):
    coords_array = np.array(list(coords_set))

    all_permutations = np.array(list(itertools.product([0, 1, -1], repeat=3)))

    new_coords_array = coords_array[:, np.newaxis, :] + all_permutations[np.newaxis, :, :]

    new_coords_set = set(map(tuple, new_coords_array.reshape(-1, 3)))

    return new_coords_set


def blowup_vol_mask(coords_set, shape, steps=1, connectivity=26, ring_only=False):
    """
    coords_set: set of (z,y,x) voxels
    shape: full volume shape (Z,Y,X)
    steps: how many dilation iterations (1 => neighbors)
    connectivity: 6, 18, or 26 connectivity
    ring_only: if True, return only the newly added voxels
    Returns: set of (z,y,x)
    """
    if not coords_set:
        return set()

    mask = np.zeros(shape, dtype=bool)
    pts = np.asarray(list(coords_set), dtype=np.int64)
    mask[pts[:,0], pts[:,1], pts[:,2]] = True

    conn_map = {6: 1, 18: 2, 26: 3}  # for generate_binary_structure(rank=3, connectivity)
    structure = generate_binary_structure(3, conn_map.get(connectivity, 3))

    dil = binary_dilation(mask, structure=structure, iterations=steps)
    if ring_only:
        dil = dil & ~mask

    return set(map(tuple, np.argwhere(dil)))


def coordinates_in_bounds(image_np, z, y, x):
    #Check whether the given (x, y, z) coordinates are within bounds of a 3D NumPy image.
    z_shape, y_shape, x_shape = image_np.shape

    if x < 0 or x >= x_shape:
        return False
    if y < 0 or y >= y_shape:
        return False
    if z < 0 or z >= z_shape:
        return False

    return True


def get_intensity(image_np, coords_list):
    # Create an array to store the intensities at the specified coordinates
    intensities = []

    # Iterate over the coordinates and get the intensity at each location
    for coord in coords_list:
        z, y, x = coord
        if coordinates_in_bounds(image_np, z, y, x):
            intensities.append(image_np[z, y, x])
        else:
            continue

    # Calculate the average intensity
    if len(intensities) == 0:
        average_intensity = 0
    else:
        average_intensity = np.mean(intensities)

    return average_intensity

def _ensure_label_image(mask_bool_or_label, connectivity=1):
    """Return a label image where 0=background, 1..N=objects."""
    """     if mask_bool_or_label.dtype.kind in "iu" and mask_bool_or_label.max() > 1:
        return mask_bool_or_label  # already labeled
    return label(mask_bool_or_label > 0, connectivity=connectivity) """

    arr = np.asarray(mask_bool_or_label)

    # If not integer, treat as mask
    if arr.dtype.kind not in "iu":
        return label(arr > 0, connectivity=connectivity)

    # Unique values tell us if it's binary vs labeled
    uniq = np.unique(arr)
    if uniq.size <= 3 and uniq.min() == 0:
        # Likely binary like {0,1} or {0,255} (or sparse)
        return label(arr != 0, connectivity=connectivity)

    # Looks like a real label field already (0..N with many IDs)
    return arr.astype(np.int32, copy=False)

def _mean_by_label(channel, labels_img, labels_list):
    """
    Fast per-label mean via bincount. SAFE even if labels_list contains ids
    greater than labels_img.max() (we size with both maxima).
    """
    lab = np.asarray(labels_img).ravel()
    val = np.asarray(channel).ravel().astype(np.float64, copy=False)
    labels_list = np.asarray(labels_list, dtype=int)

    # keep only foreground
    fg = lab > 0
    lab = lab[fg]; val = val[fg]

    if lab.size == 0 or labels_list.size == 0:
        return np.zeros(labels_list.shape, dtype=np.float64)

    max_img = int(lab.max())
    max_req = int(labels_list.max())
    minlength = max(max_img, max_req) + 1

    sums   = np.bincount(lab, weights=val, minlength=minlength)
    counts = np.bincount(lab,               minlength=minlength)
    means_all = sums / np.maximum(counts, 1)
    return means_all[labels_list]

""" def _counts_by_label(labels_img, labels_list):
    Voxel counts per label id in labels_list.
    lab = labels_img.ravel()
    counts = np.bincount(lab, minlength=int(lab.max()) + 1)
    return counts[labels_list].astype(np.int64) """

def _counts_by_label(labels_img_zyx, labels_list):
    """Voxel counts per label. Safe if labels_list contains ids not in labels_img."""
    lab = np.asarray(labels_img_zyx).ravel()
    labels_list = np.asarray(labels_list, dtype=int)
    max_img = int(lab.max()) if lab.size else 0
    max_req = int(labels_list.max()) if labels_list.size else 0
    minlength = max(max_img, max_req) + 1
    counts = np.bincount(lab, minlength=minlength)
    return counts[labels_list].astype(np.int64)

def extract_volume_intensities_fast(
    chunk_number,
    ind,
    chunk_indices_folder,
    centroids,                 # kept for signature parity (unused here)
    nuclei_mask,               # (Z,Y,X) binary or labeled
    vol_unmixed_chunk,         # iterable of channels, each (Z,Y,X)  OR  ndarray (C,Z,Y,X)
    save_mask_path=None    # path to save union L3 shell as binary mask (0/255 uint8)
):
    """
    Vectorized per-label intensities for L1..L4:

      L1  = original object voxels
      L2  = expand_labels(lbl, 1) minus L1
      L3  = expand_labels(lbl, 2) minus expand_labels(lbl, 1)
      L4  = expand_labels(lbl, 3) minus expand_labels(lbl, 2)

    Saves the **union L3 shell** (all labels) if save_l3_mask_path is given.

    Notes:
      - Expects nuclei_mask and channels to be aligned (same ZYX grid).
      - Relies on global names you were already using: `column_names`, `origin_coords`,
        `number`, `zy_factor`, `xy_factor`. This function preserves your post-processing.
    """
    print(f"Extracting intensities (fast) for chunk {chunk_number}")

    # If you still need the original regionprops-derived table (centroids, etc.), keep this:
    nuclei_df = get_props(nuclei_mask, vol_unmixed_chunk[0] if isinstance(vol_unmixed_chunk, (list, tuple, np.ndarray)) else vol_unmixed_chunk)
    spectral_info_df = nuclei_df.copy()

    # Normalize channels input to a list of (Z,Y,X) ndarrays
    if isinstance(vol_unmixed_chunk, np.ndarray) and vol_unmixed_chunk.ndim == 4:
        channels = [vol_unmixed_chunk[c] for c in range(vol_unmixed_chunk.shape[0])]
    else:
        channels = list(vol_unmixed_chunk)
    C = len(channels)

    # 1) Ensure label image
    lbl = _ensure_label_image(nuclei_mask)  # (Z,Y,X), ints
    labels_list = spectral_info_df['label'].to_numpy(dtype=int)  # order to return stats in

    # 2) Layer label images (computed sequentially to keep memory down)
    # L1 labels
    L1 = lbl.copy()
    # L2 cumulative then ring
    L2cum = expand_labels(L1, distance=1)
    L2 = L2cum.copy()
    L2[L1 != 0] = 0
    # L3 cumulative then ring
    L3cum = expand_labels(L1, distance=2)
    L3 = L3cum.copy()
    L3[L2cum != 0] = 0
    # L4 cumulative then ring
    L4cum = expand_labels(L1, distance=3)
    L4 = L4cum.copy()
    L4[L3cum != 0] = 0

    ############################################################
    ##### Check disjointness of shells
    ############################################################
    def _check_shell_disjoint(L1, L2, L3, L4, tag="pre-io"):
        pairs = [("L1","L2",(L1!=0)&(L2!=0)),
                ("L2","L3",(L2!=0)&(L3!=0)),
                ("L3","L4",(L3!=0)&(L4!=0)),
                ("L1","L3",(L1!=0)&(L3!=0)),
                ("L1","L4",(L1!=0)&(L4!=0)),
                ("L2","L4",(L2!=0)&(L4!=0))]
        msg = {f"{a}&{b}": int(np.count_nonzero(m)) for a,b,m in pairs}
        print(f"[{tag}] shell overlap voxels:", msg)

    # after constructing L1..L4:
    _check_shell_disjoint(L1, L2, L3, L4, tag=f"pre-io chunk {chunk_number}")

    ############################################################

    # 3) Volumes (voxel counts) per label for each layer
    spectral_info_df['vol_l1'] = _counts_by_label(L1, labels_list)
    spectral_info_df['vol_l2'] = _counts_by_label(L2, labels_list)
    spectral_info_df['vol_l3'] = _counts_by_label(L3, labels_list)
    spectral_info_df['vol_l4'] = _counts_by_label(L4, labels_list)

    # 4) Per-channel means per layer (vectorized)
    # Fill columns in blocks of 4 per channel: [L1, L2, L3, L4]
    for ci, ch in enumerate(channels):
        m1 = _mean_by_label(ch, L1, labels_list)
        m2 = _mean_by_label(ch, L2, labels_list)
        m3 = _mean_by_label(ch, L3, labels_list)
        m4 = _mean_by_label(ch, L4, labels_list)

        base = ci * 4
        spectral_info_df.loc[:, base + 0] = m1
        spectral_info_df.loc[:, base + 1] = m2
        spectral_info_df.loc[:, base + 2] = m3
        spectral_info_df.loc[:, base + 3] = m4

    # 5) Df assembling
    column_names = ['label', 'axis-0', 'axis-1', 'axis-2', 'coords',
                'vol_l1', 'vol_l2', 'vol_l3', 'vol_l4',
                'ch1_l1', 'ch1_l2', 'ch1_l3', 'ch1_l4',
                'ch2_l1', 'ch2_l2', 'ch2_l3', 'ch2_l4',
                'ch3_l1', 'ch3_l2', 'ch3_l3', 'ch3_l4',
                'ch4_l1', 'ch4_l2', 'ch4_l3', 'ch4_l4',
                'ch5_l1', 'ch5_l2', 'ch5_l3', 'ch5_l4',
                ]
    spectral_info_df.columns = column_names
    spectral_info_df = spectral_info_df.drop('coords', axis=1)

    # Rescale back to raw data space (your existing logic)
    chunk_origin = origin_coords[chunk_number]
    spectral_info_df['axis-0'] = spectral_info_df['axis-0'] / zy_factor + chunk_origin[0]  # global, px
    spectral_info_df['axis-1'] = spectral_info_df['axis-1'] + chunk_origin[1]  # global, px
    spectral_info_df['axis-2'] = spectral_info_df['axis-2'] / xy_factor + chunk_origin[2]  # global, px
    spectral_info_df = spectral_info_df.round(2)

    """     # 6) Save L1–L4 shell union masks (binary 0/255), if requested
    if save_mask_path:
        # Optional: zero-pad chunk numbers for nice sorting; change width if you like
        chunk_str = f"{chunk_number:05d}"
        chunk_dir = os.path.join(save_mask_path, f"chunk_{chunk_str}")
        os.makedirs(chunk_dir, exist_ok=True)

        tiff.imwrite(os.path.join(chunk_dir, 'mask_l1.tif'), (L1 > 0).astype(np.uint8) * 255, metadata={'axes': 'ZYX'})
        tiff.imwrite(os.path.join(chunk_dir, 'mask_l2.tif'), (L2 > 0).astype(np.uint8) * 255, metadata={'axes': 'ZYX'})
        tiff.imwrite(os.path.join(chunk_dir, 'mask_l3.tif'), (L3 > 0).astype(np.uint8) * 255, metadata={'axes': 'ZYX'})
        tiff.imwrite(os.path.join(chunk_dir, 'mask_l4.tif'), (L4 > 0).astype(np.uint8) * 255, metadata={'axes': 'ZYX'})

        print(f"Saved shell masks for chunk {chunk_number} to {chunk_dir}") """

    # 6) Save L1–L4 shell union masks (binary 0/255), by layer
    if save_mask_path:
        chunk_str = f"{chunk_number:05d}"  # zero-pad for nice sorting

        layers = {"L1": L1,"L2": L2,"L3": L3,"L4": L4,}
        layer_dirs = []

        for layer_name, layer_arr in layers.items():
            layer_dir = os.path.join(save_mask_path, layer_name)
            os.makedirs(layer_dir, exist_ok=True)
            layer_dirs.append(layer_dir)
            out_path = os.path.join(layer_dir, f"mask_chunk_{chunk_str}.tif")
            tiff.imwrite(out_path, (layer_arr > 0).astype(np.uint8) * 255, metadata={'axes': 'ZYX'})
        print(layer_dirs)
        for layer_dir in layer_dirs:
            combine_masks(layer_dir, vol_unmixed, chunk_indices_folder, layer_dir)
        print(f"Saved shell masks for chunk {chunk_number} into L1–L4 folders under {save_mask_path}")


    return spectral_info_df


def extract_volume_intensities(chunk_number, ind, centroids, nuclei_mask, vol_unmixed_chunk, save_l1_mask_path):
    """
    Extract average intensities based on nuclei segmentation mask.

    Assuming that resolutions are the same for nuclei and color channels
    """
    # Construct the centroid dataframe from ch1
    print(f'Extracting intensities from chunk {chunk_number}')
    nuclei_df = get_props(nuclei_mask, vol_unmixed_chunk[0])
    print("nuclei_df.shape", nuclei_df.shape)
    spectral_info_df = nuclei_df.copy()
    channels = vol_unmixed_chunk
    l1_mask = np.zeros_like(nuclei_mask, dtype=np.uint8)

    for label in spectral_info_df['label']:
        print(f'Working on label {label}')

        # Get coordinates of the initial region (nucleus) as well as the blown up regions
        l1_coords = spectral_info_df.loc[nuclei_df['label'] == label, 'coords'].values[0]
        

        ### Create mask ###
        # Mark layer-1 voxels in the mask (vectorized)
        l1_arr = np.asarray(l1_coords)
        if l1_arr.ndim != 2 or l1_arr.shape[1] != 3:
            raise ValueError(f"coords for label {label} should be Nx3 (z,y,x); got {l1_arr.shape}")
        z_idx = l1_arr[:, 0].astype(np.int64)
        y_idx = l1_arr[:, 1].astype(np.int64)
        x_idx = l1_arr[:, 2].astype(np.int64)
        l1_mask[z_idx, y_idx, x_idx] = 1
        ###################

        l1_coords = set(tuple(coords) for coords in l1_coords.tolist())
        #l2_coords = blowup_vol(l1_coords)
        #l3_coords = blowup_vol(l2_coords)
        #l4_coords = blowup_vol(l3_coords)

        #added_region_l2 = l2_coords - l1_coords
        #added_region_l3 = l3_coords - l2_coords
        #added_region_l4 = l4_coords - l3_coords

        vol_l1 = len(l1_coords)
        #vol_l2 = len(added_region_l2)
        #vol_l3 = len(added_region_l3)
        #vol_l4 = len(added_region_l4)

        spectral_info_df.loc[spectral_info_df['label'] == label, 'vol_l1'] = vol_l1
        #spectral_info_df.loc[spectral_info_df['label'] == label, 'vol_l2'] = vol_l2
        #spectral_info_df.loc[spectral_info_df['label'] == label, 'vol_l3'] = vol_l3
        #spectral_info_df.loc[spectral_info_df['label'] == label, 'vol_l4'] = vol_l4

        counter = 0
        for channel in channels:

            # Calculate their coordinates
            l1_intensity = get_intensity(channel, list(l1_coords))
            #l2_intensity = get_intensity(channel, list(added_region_l2))
            #l3_intensity = get_intensity(channel, list(added_region_l3))
            #l4_intensity = get_intensity(channel, list(added_region_l4))

            spectral_info_df.loc[spectral_info_df['label'] == label, counter] = l1_intensity
            #spectral_info_df.loc[spectral_info_df['label'] == label, counter + 1] = l2_intensity
            #spectral_info_df.loc[spectral_info_df['label'] == label, counter + 2] = l3_intensity
            #spectral_info_df.loc[spectral_info_df['label'] == label, counter + 3] = l4_intensity
            counter += 1 # if all channels +=4

    column_names = ['label', 'axis-0', 'axis-1', 'axis-2', 'coords',
                'vol_l1', 
                'ch1_l1', 
                'ch2_l2',
                'ch3_l1', 
                'ch4_l1', 
                'ch5_l1', 
                ]

    spectral_info_df.columns = column_names
    spectral_info_df = spectral_info_df.drop('coords', axis=1)
    chunk_origin = origin_coords[chunk_number]
    # rescale back to raw data space
    spectral_info_df['axis-0'] = spectral_info_df['axis-0'] / zy_factor + chunk_origin[0]  # global, px
    spectral_info_df['axis-1'] = spectral_info_df['axis-1'] + chunk_origin[1]  # global, px
    spectral_info_df['axis-2'] = spectral_info_df['axis-2'] / xy_factor + chunk_origin[2]  # global, px
    spectral_info_df = spectral_info_df.round(2)

    # --- save L1 mask if requested ---
    if save_l1_mask_path:
        # write as 0/255 uint8 so it’s visible in viewers; axes: ZYX
        tiff.imwrite(os.path.join(save_l1_mask_path, f'mask_{chunk_number}.tif'), (l1_mask > 0).astype(np.uint8) * 255,
                         metadata={'axes': 'ZYX'})
        print(f"Saved L1 binary mask to {save_l1_mask_path}")

    return spectral_info_df



def process_chunk(chunk_number, ind, chunk_indices_folder, centroids_folder, detection_masks_folder, vol_unmixed, spectral_info_folder):
    print(f"=========== Processing chunk {chunk_number} ===========")

    # Ensure output dir exists
    os.makedirs(spectral_info_folder, exist_ok=True)

    # Output path for this chunk
    out_csv = os.path.join(spectral_info_folder, f"spectral_chunk_{chunk_number:05d}.csv")

    # Skip if already done (unless overwrite=True)
    if  os.path.exists(out_csv) and os.path.getsize(out_csv) > 0:
        print(f"[SKIP] {out_csv} already exists and is non-empty.")
        return

    # Get chunk centroids
    fname = f"napari_chunk_{chunk_number:0{5}d}.csv"
    path = os.path.join(centroids_folder, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    centroids = pd.read_csv(path)

    # Get chunk masks
    fname = f"mask_chunk_{chunk_number:0{5}d}.tif"  # e.g. 7 -> mask_chunk_00007.tif
    path = os.path.join(detection_masks_folder, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    nuclei_mask = tiff.imread(path)

    # Get chunk from vil_unmixed
    print("--------------------nuclei_mask", nuclei_mask.shape)
    yx_ratio = float(NUCLEI_RESOLUTION[-1]) / NUCLEI_RESOLUTION[-2]
    yz_ratio = float(NUCLEI_RESOLUTION[-3]) / NUCLEI_RESOLUTION[-2]

    # target_shape = (int(round(nuclei_chunk_shape[0] * yz_ratio)), nuclei_chunk_shape[1], int(round(nuclei_chunk_shape[2] * yx_ratio)))
    # print("target_shape", target_shape)
    #ch1_np = (resize(vol_unmixed[int(0), int(ind[0]), int(ind[1]), int(ind[2])], nuclei_masks_np.shape) * 65535).astype('uint16')

    # !!! Assuming that resolutions are the same for nuclei and colors
    # rescaling color data to isotropic space
    #ch2_np = (resize(vol_unmixed[0, ind[0], ind[1], ind[2]], nuclei_mask.shape) * 65535).astype('uint16')
    #ch3_np = (resize(vol_unmixed[1, ind[0], ind[1], ind[2]], nuclei_mask.shape) * 65535).astype('uint16')
    #ch4_np = (resize(vol_unmixed[2, ind[0], ind[1], ind[2]], nuclei_mask.shape) * 65535).astype('uint16')
    #ch5_np = (resize(vol_unmixed[3, ind[0], ind[1], ind[2]], nuclei_mask.shape) * 65535).astype('uint16')
    #vol_unmixed_chunk = [ch1_np, ch2_np, ch3_np, ch4_np, ch5_np]
    location = os.path.join(vol_unmixed, 'omehans')
    dask_zarray = read_omehans(location)
    vol_unmixed_chunk = get_chunk(ind, dask_zarray, yz_ratio=yz_ratio, yx_ratio=yx_ratio, order=1, preserve_range=True)

    # Construct the centroid dataframe from ch1
    #nuclei_df = get_props(nuclei_masks_np, ch1_np)
    #print("nuclei_df.shape", nuclei_df.shape)
    #spectral_info_df = nuclei_df.copy()
    

    #nuclei_chunk_shape = [
    #    int(round(nuclei_chunk.shape[0] * zy_factor)), nuclei_chunk.shape[1], int(round(nuclei_chunk.shape[2] * xy_factor))
    #]  # ONLY for removing bg

    #nonzero = remove_background(number, nuclei_chunk_shape)
    
    #if nonzero:
    #    print("Extracting intensities")
    #    spectral_df = extract_volume_intensities(nuclei_chunk_shape, number)
    #else:
    #    print("This chunk is background only. Creating empty dataframe")
      #  spectral_df = pd.DataFrame(columns=column_names)

    spectral_df = extract_volume_intensities_fast(chunk_number, ind, chunk_indices_folder, centroids, nuclei_mask, vol_unmixed_chunk, spectral_info_folder)
    spectral_df.to_csv(out_csv)


print("------------- EXTRACTING SPECTRAL INFO ------------")
chunk_number = sys.argv[1]
chunk_indices_folder = sys.argv[2]
centroids_folder = sys.argv[3]
detection_masks_folder = sys.argv[4]
vol_unmixed = sys.argv[5]
spectral_info_folder = sys.argv[6]
try:
    os.makedirs(spectral_info_folder)
except FileExistsError:
    pass

#chunk_indices_folder = Path(spectral_info_folder).resolve().parent

#dbscan_folder = os.path.join(OUTPUT_DIR, f'scale_{SCALE}', "dbscan")  # for background filtered csv files TODO
#try:
#    os.makedirs(dbscan_folder)
#except FileExistsError:
#    pass

xy_factor = float(NUCLEI_RESOLUTION[-1]) / NUCLEI_RESOLUTION[-2]
zy_factor = float(NUCLEI_RESOLUTION[-3]) / NUCLEI_RESOLUTION[-2]
location = os.path.join(vol_unmixed, 'omehans')
dask_zarray = read_omehans(location)
lazy_tiff_stack = dask_zarray# [0, :, :, :]
print(f'lazy_tiff_stack {lazy_tiff_stack.shape}')
#nuclei_dimensions_um = np.array(lazy_tiff_stack.shape) * np.array(NUCLEI_RESOLUTION)
#print("Nuclei channel dimensions um", nuclei_dimensions_um)
ratios = (np.array(lazy_tiff_stack[0, :, :, :].shape) / np.array(CHUNK_SIZE)).astype('int') + 1
patchify_chunks_shape = (*list(ratios), *CHUNK_SIZE)
origin_coords = get_origin_coords(3, patchify_chunks_shape, CHUNK_SIZE)
#chunk_indices = get_chunk_indices(origin_coords, CHUNK_SIZE)

# Chat gpt check of valid chunk indices
#-----------------------------------------------------
chunk_indices_path = os.path.join(chunk_indices_folder, 'chunk_indices.npy')
if os.path.exists(chunk_indices_path):
    chunk_indices = np.load(chunk_indices_path, allow_pickle=True)
else:
    chunk_indices = get_chunk_indices(origin_coords, CHUNK_SIZE)

# Early exit if requested chunk is outside available range so SLURM reports the skipped chunk.
if int(chunk_number) >= len(chunk_indices) or int(chunk_number) < 0:
    print(f"Chunk {chunk_number} is invalid; available chunks: {len(chunk_indices)}")
    sys.exit(1)
#-----------------------------------------------------

#lazy_data = dask_zarray[0, 1:, :, :, :]
nuclei_box_size = np.round(CUBE_SIZE / np.array(NUCLEI_RESOLUTION)).astype(int)  # 10 um box
ind = chunk_indices[int(chunk_number)]
color_info_box_size = np.round(CUBE_SIZE / np.array(COLOR_RESOLUTION)).astype(int)  # 10 um box
print("Box size", color_info_box_size)

""" column_names = ['label', 'axis-0', 'axis-1', 'axis-2', 'coords',
                'vol_l1', 'vol_l2', 'vol_l3', 'vol_l4',
                'ch1_l1', 'ch1_l2', 'ch1_l3', 'ch1_l4',
                'ch2_l1', 'ch2_l2', 'ch2_l3', 'ch2_l4',
                'ch3_l1', 'ch3_l2', 'ch3_l3', 'ch3_l4',
                'ch4_l1', 'ch4_l2', 'ch4_l3', 'ch4_l4',
                'ch5_l1', 'ch5_l2', 'ch5_l3', 'ch5_l4',
                ] """

"""  column_names = ['label', 'axis-0', 'axis-1', 'axis-2', 'coords',
                'vol_l1', 
                'ch1_l1', 
                'ch3_l1', 
                'ch4_l1', 
                'ch5_l1', 
                ]   """             

process_chunk(int(chunk_number), ind, chunk_indices_folder, centroids_folder, detection_masks_folder, vol_unmixed, spectral_info_folder)
