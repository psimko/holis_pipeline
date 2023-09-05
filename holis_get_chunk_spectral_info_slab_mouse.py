import itertools
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import dask.array as da
import tifffile
import zarr
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import DBSCAN
from stack_to_multiscale_ngff.archived_nested_store import Archived_Nested_Store
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store

from utils.settings import *


def get_origin_coords(ndim, patchify_chunks_shape, chunk_size):
    """
    Get coordinates of each chunk origin.

    TODO: only 3D now, make compatible with 2D

    :param ndim:
    :param chunk_shape:
    :param patches_shape:
    :return:
    """
    coords_shape = list(patchify_chunks_shape[:ndim]) + [ndim]
    coords = np.empty(coords_shape, dtype=np.uint16)
    print(" coords shape", coords.shape)
    for z in range(coords.shape[0]):
        for y in range(coords.shape[1]):
            for x in range(coords.shape[2]):
                coords[z, y, x, :] = np.array((
                    z * chunk_size[0],
                    y * chunk_size[1],
                    x * chunk_size[2]
                ))
    coords = np.reshape(coords, (np.prod(coords.shape[:ndim]), ndim))
    print("final coords shape", coords.shape)
    return coords


def get_chunk_indices(origin_coords, chunk_size):
    indices = []
    for origin in list(origin_coords):
        indices.append([
            slice(origin[0], origin[0] + chunk_size[0], 1),
            slice(origin[1], origin[1] + chunk_size[1], 1),
            slice(origin[2], origin[2] + chunk_size[2], 1)
        ])
    return indices


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


def convert_to_napari_format(chunk_file):
    """
    Change columns in csv file to make it readable with napari.

    :param chunks_folder:
    :return:
    """
    csv_file = chunk_file.replace('.tif', '.csv')
    print("Converting", csv_file)
    napari_csv_file_path = os.path.join(os.path.dirname(csv_file), f"napari_{os.path.basename(csv_file)}")
    if os.path.exists(napari_csv_file_path):
        return pd.read_csv(napari_csv_file_path)
    try:
        df = pd.read_csv(csv_file)
    except pd.errors.EmptyDataError as e:
        print("Warning: ", e)
        return
    df2 = pd.DataFrame()
    df2['index'] = list(range(df.shape[0]))  # TODO
    try:
        zvals = df['z'].tolist()
        yvals = df['y [px]'].tolist()
        xvals = df['x [px]'].tolist()
    except KeyError as e:
        print(e)
        return

    df2['axis-0'] = zvals
    df2['axis-1'] = xvals
    df2['axis-2'] = yvals
    df2.to_csv(napari_csv_file_path)
    return df2


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
    filtered_cells_df = pd.DataFrame()
    filtered_cells_df['axis-0'] = list(filtered_cells_np[:, 0])
    filtered_cells_df['axis-1'] = list(filtered_cells_np[:, 1])
    filtered_cells_df['axis-2'] = list(filtered_cells_np[:, 2])
    print("Saving coords to csv")
    filtered_cells_df.to_csv(os.path.join(dbscan_folder, f"filtered_chunk_{str(number).zfill(5)}.csv"))
    return filtered_cells_df, filtered_cells_np


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
    return region_props


def blowup_vol(coords_set, image_np):
    # Convert the set of coordinates to a list
    coords_list = list(coords_set)

    # Convert coordinates to tuples
    new_coords = [tuple(coords) for coords in coords_list]

    # Generate all possible permutations of the 1D array
    all_permutations = np.array(list(itertools.product([0, 1, -1], repeat=3)))

    # Create a set to store the new coordinates
    new_coords_set = set(new_coords)

    for coords in coords_list:
        for vector in all_permutations:
            new_coord = coords + vector
            if tuple(new_coord) not in new_coords_set and coordinates_in_bounds(image_np, new_coord[0], new_coord[1],
                                                                                new_coord[2]):
                new_coords_set.add(tuple(new_coord))  # Add the new_coord as a tuple to the set

    # Convert the set of new coordinates back to a set
    return new_coords_set


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
    """
    # Calculates average intensity over a list of coordinates in a tiff image
    """
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


def extract_volume_intensities():
    """
    Extract average intensities based on nuclei segmentation mask.

    Assuming that resolutions are the same for nuclei and color channels
    """
    # find mask by chunk number
    NUCLEI_MASK_PATH = os.path.join(os.path.dirname(chunk_file), f"mask_{os.path.basename(chunk_file)}")
    # rescale points to ome-zarr data space

    nuclei_masks_np = tifffile.imread(NUCLEI_MASK_PATH)  # mask is in the isotropic space
    # ch1_np = zarray[0, 0, ind[0], ind[1], ind[2]]
    ch1_np = tifffile.imread(chunk_file)  # isotropic
    # !!! Assuming that resolutions are the same for nuclei and colors
    # rescaling color data to isotropic space
    ch2_np = resize(color_info_zarray[0, 0, ind[0], ind[1], ind[2]], ch1_np.shape)
    ch3_np = resize(color_info_zarray[0, 1, ind[0], ind[1], ind[2]], ch1_np.shape)
    ch4_np = resize(color_info_zarray[0, 2, ind[0], ind[1], ind[2]], ch1_np.shape)
    ch5_np = resize(color_info_zarray[0, 3, ind[0], ind[1], ind[2]], ch1_np.shape)

    # Construct the centroid dataframe from ch1
    nuclei_df = get_props(nuclei_masks_np, ch1_np)
    spectral_info_df = nuclei_df
    channels = [ch1_np, ch2_np, ch3_np, ch4_np, ch5_np]

    for label in spectral_info_df['label']:
        print(f'Working on label {label}')

        # Get coordinates of the initial region (nucleus) as well as the blown up regions
        l1_coords = spectral_info_df.loc[nuclei_df['label'] == label, 'coords'].values[0]
        l1_coords = set(tuple(coords) for coords in l1_coords.tolist())
        l2_coords = blowup_vol(l1_coords, ch1_np)
        l3_coords = blowup_vol(l2_coords, ch1_np)
        l4_coords = blowup_vol(l3_coords, ch1_np)

        added_region_l2 = l2_coords - l1_coords
        added_region_l3 = l3_coords - l2_coords
        added_region_l4 = l4_coords - l3_coords

        vol_l1 = len(l1_coords)
        vol_l2 = len(added_region_l2)
        vol_l3 = len(added_region_l3)
        vol_l4 = len(added_region_l4)

        spectral_info_df.loc[spectral_info_df['label'] == label, 'vol_l1'] = vol_l1
        spectral_info_df.loc[spectral_info_df['label'] == label, 'vol_l2'] = vol_l2
        spectral_info_df.loc[spectral_info_df['label'] == label, 'vol_l3'] = vol_l3
        spectral_info_df.loc[spectral_info_df['label'] == label, 'vol_l4'] = vol_l4

        counter = 0
        for channel in channels:

            # Calculate their coordinates
            l1_intensity = get_intensity(channel, list(l1_coords))
            l2_intensity = get_intensity(channel, list(added_region_l2))
            l3_intensity = get_intensity(channel, list(added_region_l3))
            l4_intensity = get_intensity(channel, list(added_region_l4))

            spectral_info_df.loc[spectral_info_df['label'] == label, counter] = l1_intensity
            spectral_info_df.loc[spectral_info_df['label'] == label, counter + 1] = l2_intensity
            spectral_info_df.loc[spectral_info_df['label'] == label, counter + 2] = l3_intensity
            spectral_info_df.loc[spectral_info_df['label'] == label, counter + 3] = l4_intensity
            counter += 4

    column_names = ['label', 'axis-0', 'axis-1', 'axis-2','coords',
                    'vol_l1', 'vol_l2', 'vol_l3', 'vol_l4',
                    'ch1_l1', 'ch1_l2', 'ch1_l3', 'ch1_l4',
                    'ch2_l1', 'ch2_l2', 'ch2_l3', 'ch2_l4',
                    'ch3_l1', 'ch3_l2', 'ch3_l3', 'ch3_l4',
                    'ch4_l1', 'ch4_l2', 'ch4_l3', 'ch4_l4',
                    'ch5_l1', 'ch5_l2', 'ch5_l3', 'ch5_l4',
                    ]
    spectral_info_df.columns = column_names
    spectral_info_df = spectral_info_df.drop('coords', axis=1)
    chunk_origin = origin_coords[number]
    # rescale back to raw data space
    spectral_info_df['axis-0'] = spectral_info_df['axis-0'] + chunk_origin[0]  # global, px
    spectral_info_df['axis-1'] = spectral_info_df['axis-1'] + chunk_origin[1]  # global, px
    spectral_info_df['axis-2'] = spectral_info_df['axis-2'] + chunk_origin[2]  # global, px
    spectral_info_df = spectral_info_df.round(2)
    return spectral_info_df


def process_chunk(chunk_file, number):
    #print("Processing chunk", number)
    nuclei_chunk = zarray[0, 0, ind[0], ind[1], ind[2]]
    napari_csv = os.path.join(os.path.dirname(chunk_file), f"napari_{os.path.basename(chunk_file).replace('.tif', '.csv')}")
    points_df = pd.read_csv(napari_csv)
    points_df = points_df[["axis-0", "axis-1", "axis-2"]]
    points = points_df.to_numpy()
    print("Points", points.shape)
    nuclei_chunk_shape = [
        int(round(nuclei_chunk.shape[0] * zy_factor)), nuclei_chunk.shape[1], int(round(nuclei_chunk.shape[2] * xy_factor))
    ]  # ONLY for removing bg
    filtered_df, points = remove_background_spots(points, nuclei_chunk_shape)  # points are in chunk (isotropic) space
    points = points.astype('float32')
    spectral_df = extract_volume_intensities()
    spectral_df.to_csv(os.path.join(spectral_info_folder, f"spectral_chunk_{str(number).zfill(5)}.csv"))


chunk_file = sys.argv[1]
NUCLEI_DIR = sys.argv[2]

chunks_folder = str(Path(chunk_file).parent)
spectral_info_folder = os.path.join(str(Path(chunks_folder).parent), "spectral_info")
try:
    os.makedirs(spectral_info_folder)
except FileExistsError:
    pass
dbscan_folder = os.path.join(str(Path(chunks_folder).parent), "dbscan")  # for background filtered csv files
if not os.path.exists(dbscan_folder):
     os.makedirs(dbscan_folder)
xy_factor = float(NUCLEI_RESOLUTION[-1]) / NUCLEI_RESOLUTION[-2]
zy_factor = float(NUCLEI_RESOLUTION[-3]) / NUCLEI_RESOLUTION[-2]
number = int(re.findall(r"\d+", os.path.basename(chunk_file))[-1])
location = os.path.join(NUCLEI_DIR, 'scale0')
store = H5_Nested_Store(location)
zarray = zarr.open(store)
dask_zarray = da.array(zarray)
lazy_tiff_stack = dask_zarray[0, 0, :, :, :]
nuclei_dimensions_um = np.array(lazy_tiff_stack.shape) * np.array(NUCLEI_RESOLUTION)
print("Nuclei channel dimensions um", nuclei_dimensions_um)
ratios = (np.array(lazy_tiff_stack.shape) / np.array(CHUNK_SIZE)).astype('int') + 1
patchify_chunks_shape = (*list(ratios), *CHUNK_SIZE)
origin_coords = get_origin_coords(3, patchify_chunks_shape, CHUNK_SIZE)
chunk_indices = get_chunk_indices(origin_coords, CHUNK_SIZE)
lazy_data = dask_zarray[0, 1:, :, :, :]
nuclei_box_size = np.round(CUBE_SIZE / np.array(NUCLEI_RESOLUTION)).astype(int)  # 10 um box
ind = chunk_indices[number]

color_info_location = os.path.join(COLORS_DIR, 'scale0')
color_info_store = H5_Nested_Store(color_info_location)
color_info_zarray = zarr.open(color_info_store)
color_info_shape = color_info_zarray.shape[-3:]
color_info_box_size = np.round(CUBE_SIZE / np.array(COLOR_RESOLUTION)).astype(int)  # 10 um box
print("Box size", color_info_box_size)

process_chunk(chunk_file, number)
