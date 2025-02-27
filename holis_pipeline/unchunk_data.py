from holis_pipeline.settings import *


def merge_df_fix_wrong_scaling_no_bg(chunks_folder, origin_coords, xy_factor):
    df_column_names = ['index', 'axis-0', 'axis-1', 'axis-2']  # TODO: ndim
    df = pd.DataFrame(columns=df_column_names)
    csv_files = sorted(glob(os.path.join(chunks_folder, 'napari*.csv')))
    bg_chunks = set(np.load(os.path.join(OUTPUT_DIR, f'scale_{SCALE}', 'zero_chunks.npy')))
    csv_files = [x for x in csv_files if int(re.findall(r"\d+", os.path.basename(x))[-1]) not in bg_chunks]
    print("CSV files", len(csv_files))

    def read_df(chunk_file):
        print("reading", chunk_file)
        return pd.read_csv(chunk_file)

    def process_df(chunk_file, chunk_df):
        current_chunk = int(re.findall(r"\d+", os.path.basename(chunk_file))[-1])
        print("processing", current_chunk)
        chunk_df_corrected = pd.DataFrame()
        z_values = chunk_df[['axis-0']].to_numpy()
        y_values = chunk_df[['axis-1']].to_numpy()
        x_values = chunk_df[['axis-2']].to_numpy()
        x_values = x_values / xy_factor
        z_values += origin_coords[current_chunk, 0]
        y_values += origin_coords[current_chunk, 1]
        x_values += origin_coords[current_chunk, 2]
        chunk_df_corrected['index'] = list(range(chunk_df.shape[0]))
        chunk_df_corrected['axis-0'] = z_values
        chunk_df_corrected['axis-1'] = y_values
        chunk_df_corrected['axis-2'] = x_values
        return chunk_df_corrected

    dfs = [dask.delayed(read_df)(f) for f in csv_files]
    processed = [dask.delayed(process_df)(f, d_f) for f, d_f in zip(csv_files, dfs)]
    processed = dask.compute(processed)
    df = pd.concat(*processed)
    print("Saving df")
    df['index'] = list(range(df.shape[0]))
    return df


# def merge_spectral_info_df(origin_coords, bg_chunks):
#     spectral_info_folder = os.path.join(OUTPUT_DIR, f"scale_{SCALE}", "spectral_info")
#     # get all csv in spectral info folder, check their number
#     dfs = sorted(glob(os.path.join(spectral_info_folder, "spectral*.csv")))
#     print("total dfs", len(dfs))
#     # check they do not belong to bg
#     dfs = [x for x in dfs if int(re.findall(r"\d+", os.path.basename(x))[-1]) not in bg_chunks]
#     print("Dfs to merge", len(dfs))
#     # read them (without dask)
#     print("reading")
#     to_merge = []
#     labels_max = 0
#     for df_file in dfs:
#         current_chunk = int(re.findall(r"\d+", os.path.basename(df_file))[-1])
#         print("processing", current_chunk)
#         df = pd.read_csv(df_file)
#         df['label'] += labels_max
#         labels_max = df['label'].max()
#         to_merge.append(df)
#
#     print("done reading. Concatenating")
#     # concatenate them (without rescaling)
#     df = pd.concat(to_merge, ignore_index=True)
#     print("Concatenated. Saving")
#     # save new df
#     df.to_csv(os.path.join(OUTPUT_DIR, f"scale_{SCALE}", "detected_cells_pytorch_unet_bg_removed_px_w_color_info.csv"))
#     return df


def combine_masks(nuclei_dir):
    print("Merging masks")
    chunk_indices = np.load(
        os.path.join(OUTPUT_DIR, f'scale_{SCALE}', "chunk_indices.npy"),
        allow_pickle=True
    )
    store_nuclei = H5_Nested_Store(f"{nuclei_dir}/scale{SCALE}")
    zarray_nuclei = zarr.open(store_nuclei)
    raw_img_shape = zarray_nuclei.shape[-3:]
    combined_mask = np.zeros(raw_img_shape, dtype=np.uint8)
    masks = sorted(
        glob(
            os.path.join(OUTPUT_DIR, f"scale_{SCALE}", "detection_masks", "mask*.tif")
        )
    )
    for mask in masks:
        chunk_number = int(re.findall(r"\d+", os.path.basename(mask))[-1])
        print("Adding chunk", chunk_number)
        chunk_slices = chunk_indices[chunk_number]
        edge_flag = False
        edge_chunk_shape = CHUNK_SIZE
        if chunk_slices[0].stop >= raw_img_shape[0]:
            edge_flag = True
            chunk_slices[0] = slice(chunk_slices[0].start, raw_img_shape[0], chunk_slices[0].step)
            edge_chunk_shape = (raw_img_shape[0] - chunk_slices[0].start, edge_chunk_shape[1], edge_chunk_shape[2])
        if chunk_slices[1].stop >= raw_img_shape[1]:
            edge_flag = True
            chunk_slices[1] = slice(chunk_slices[1].start, raw_img_shape[1], chunk_slices[1].step)
            edge_chunk_shape = (edge_chunk_shape[0], raw_img_shape[1] - chunk_slices[1].start, edge_chunk_shape[2])
        if chunk_slices[2].stop >= raw_img_shape[2]:
            edge_flag = True
            chunk_slices[2] = slice(chunk_slices[2].start, raw_img_shape[2], chunk_slices[2].step)
            edge_chunk_shape = (edge_chunk_shape[0], edge_chunk_shape[1], raw_img_shape[2] - chunk_slices[2].start)
        mask_resized_space = tifffile.imread(mask)
        print("Read mask of shape", mask_resized_space.shape)
        if edge_flag:
            print("Edge chunk")
            mask_raw_space = resize(mask_resized_space, edge_chunk_shape) > 0
        else:
            mask_raw_space = resize(mask_resized_space, CHUNK_SIZE) > 0
        combined_mask[chunk_slices[0], chunk_slices[1], chunk_slices[2]] = mask_raw_space

    mask_location = os.path.join(OUTPUT_DIR, f"scale_{SCALE}", "combined_mask_raw_space.tif")
    tifffile.imwrite(mask_location, combined_mask)  # TODO zarr?
    return mask_location


def remove_chunking_artifacts(location):
    # from large slab pipeline
    no_artifact_location = oas.path.join(os.path.dirname(location), f"{os.path.basename(location)}_artifact_removed")
    try:
        os.makedirs(no_artifact_location)
    except:
        pass
    return no_artifact_location


def extract_coords(mask_location, output_dir):
    # from current pipeline
    # using regionprops
    # TODO: what format do we save it in?
    coords_file = os.path.join(output_dir, "nuclei_coords.csv")
    return coords_file
