from holis_pipeline.settings import *
import numpy as np
import os
import pandas as pd
import tifffile
import zarr
from skimage.transform import resize
from stack_to_multiscale_ngff.h5_nested_store3 import H5_Nested_Store
from holis_pipeline.utils.zarr_related import *
from glob import glob
import re
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops_table


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


    """ def combine_masks(detection_masks_folder, vol_unmixed_folder, output_folder_scale, output_combined_mask):
    print("Merging masks")
    mask_location = os.path.join(output_combined_mask, "combined_mask_raw_space.tif")
    if os.path.exists(mask_location):
        print(f"Mask already exists at {mask_location}, skipping write.")
        return mask_location
    else:
        chunk_indices = np.load(os.path.join(output_folder_scale, 'chunk_indices.npy'), allow_pickle=True)
        location = os.path.join(vol_unmixed_folder, 'omehans')
        dask_zarray = read_omehans(location)
        print(f'lazy_tiff_stack {dask_zarray.shape}')  # this is coming in as (X,Z,Y) here
        #lazy_tiff_stack = dask_zarray[0, :, :, :]
        if dask_zarray.ndim == 4:
            lazy_tiff_stack = dask_zarray[0, :, :, :]   # take first channel: (X,Z,Y),   ( (Z, Y, X) )
        elif dask_zarray.ndim == 3:
            lazy_tiff_stack = dask_zarray               # already (X,Z,Y)         ((Z, Y, X))
        else:
            raise ValueError(f"Unexpected volume ndim={dask_zarray.ndim}, shape={dask_zarray.shape}")
        
        ###    
        lazy_tiff_stack = lazy_tiff_stack.transpose(1, 2, 0)   #to (Z,Y,X)
        ###

        raw_img_shape = lazy_tiff_stack.shape 
        combined_mask = np.zeros(raw_img_shape, dtype=np.uint8)
        masks = sorted(
            glob(
                os.path.join(detection_masks_folder, "mask*.tif")
            )
        )
            
        for mask in masks:
            chunk_number = int(re.findall(r"\d+", os.path.basename(mask))[-1])
            print(f"Adding chunk {chunk_number}")
            chunk_slices = chunk_indices[chunk_number]
            print(f"Chunk slices: {chunk_slices}")
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
            print(f"Edge chunk shape: {edge_chunk_shape}")
            mask_resized_space = tifffile.imread(mask)
            print("Read mask of shape", mask_resized_space.shape)

            mask_bin = (mask_resized_space != 0)
            
            ###
            mask_bin = mask_bin.transpose(1, 2, 0) #to (Z,Y,X)
            ###

            if edge_flag:
                print("Edge chunk")
                mask_raw_space = resize(
                    mask_bin.astype(np.uint8), edge_chunk_shape,
                    order=0, preserve_range=True, anti_aliasing=False
                ).astype(bool)
            else:
                mask_raw_space = mask_bin
            combined_mask[chunk_slices[0], chunk_slices[1], chunk_slices[2]] |= mask_raw_space

            ###
            combined_mask =  combined_mask.transpose(2, 0, 1)     #back to (X,Z,Y)
            ###

            print(f"Final combined mask shape , {combined_mask.shape}")
        tifffile.imwrite(mask_location, (combined_mask > 0).astype(np.uint8) * 255, metadata={'axes': 'XZY'})
        return mask_location  """

def _to_zyx_from_xzy(arr):
    """
    Convert:
      - (X, Z, Y)       -> (Z, Y, X)
      - (C, X, Z, Y)    -> (C, Z, Y, X)
    """
    if arr.ndim == 3:
        # (X, Z, Y) -> (Z, Y, X)
        return arr.transpose(1, 2, 0)

    elif arr.ndim == 4:
        # (C, X, Z, Y) -> (C, Z, Y, X)
        return arr.transpose(0, 2, 3, 1)

    else:
        raise ValueError(
            f"Expected 3D or 4D array, got ndim={arr.ndim}, shape={arr.shape}"
        )


def combine_masks(detection_masks_folder, vol_unmixed_folder, output_folder_scale, output_combined_mask):
    print("Merging masks")

    mask_location = os.path.join(output_combined_mask, "combined_mask_raw_space.tif")
    if os.path.exists(mask_location):
        print(f"Mask already exists at {mask_location}, skipping write.")
        return mask_location

    # ---- Load chunk indices (assumed ZYX)
    chunk_indices_path = os.path.join(output_folder_scale, "chunk_indices.npy")
    chunk_indices = np.load(chunk_indices_path, allow_pickle=True)

    # ---- Load volume just to get shape; convert to ZYX if needed
    location = os.path.join(vol_unmixed_folder, "omehans")
    dask_zarray = read_omehans(location)

    print("read_omehans shape:", dask_zarray.shape, "ndim:", dask_zarray.ndim)

    # Select channel if present
    lazy = dask_zarray[0] if dask_zarray.ndim > 3 else dask_zarray
    if lazy.ndim != 3:
        raise ValueError(f"Expected 3D after selecting channel, got shape={lazy.shape}")

    # You said lazy comes in as (X,Z,Y). Convert to (Z,Y,X).
    lazy_zyx = _to_zyx_from_xzy(lazy)
    raw_img_shape = lazy_zyx.shape
    print("Using raw_img_shape (ZYX):", raw_img_shape)

    combined_mask = np.zeros(raw_img_shape, dtype=np.uint8)

    # ---- Iterate masks
    masks = sorted(glob(os.path.join(detection_masks_folder, "mask*.tif")))
    if not masks:
        print("No masks found.")
        return mask_location

    for mask_path in masks:
        chunk_number = int(re.findall(r"\d+", os.path.basename(mask_path))[-1])
        print(f"Adding chunk {chunk_number}")

        if chunk_number < 0 or chunk_number >= len(chunk_indices):
            print(f"Skipping mask {mask_path}: chunk_number out of range")
            continue

        # chunk_slices is [sliceZ, sliceY, sliceX] (ZYX convention)
        chunk_slices = list(chunk_indices[chunk_number])

        # Clamp slices to raw_img_shape (ZYX)
        chunk_slices[0] = slice(chunk_slices[0].start, min(chunk_slices[0].stop, raw_img_shape[0]), chunk_slices[0].step)
        chunk_slices[1] = slice(chunk_slices[1].start, min(chunk_slices[1].stop, raw_img_shape[1]), chunk_slices[1].step)
        chunk_slices[2] = slice(chunk_slices[2].start, min(chunk_slices[2].stop, raw_img_shape[2]), chunk_slices[2].step)

        block_shape = (
            chunk_slices[0].stop - chunk_slices[0].start,
            chunk_slices[1].stop - chunk_slices[1].start,
            chunk_slices[2].stop - chunk_slices[2].start,
        )

        # Read mask (you said these are XZY on disk)
        mask = tifffile.imread(mask_path)
        mask_bin = (mask != 0)

        if mask_bin.ndim != 3:
            raise ValueError(f"Mask is not 3D: {mask_path} shape={mask_bin.shape}")

        # Convert mask XZY -> ZYX to match chunk_slices
        mask_bin_zyx = _to_zyx_from_xzy(mask_bin)

        # Resize if needed to match destination block
        if mask_bin_zyx.shape != block_shape:
            mask_raw = resize(
                mask_bin_zyx.astype(np.uint8),
                block_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(bool)
        else:
            mask_raw = mask_bin_zyx

        # Union into combined mask
        combined_mask[chunk_slices[0], chunk_slices[1], chunk_slices[2]] |= mask_raw

        mask_xzy = combined_mask.transpose(2, 0, 1)  # (Z,Y,X) -> (X,Z,Y)

    # Save combined mask in ZYX
    os.makedirs(output_combined_mask, exist_ok=True)
    tifffile.imwrite(
        mask_location,
        (mask_xzy > 0).astype(np.uint8) * 255,
        metadata={"axes": "XZY"},
    )

    print("Saved:", mask_location)
    return mask_location


def remove_chunking_artifacts(location):
    # from large slab pipeline
    no_artifact_location = os.path.join(os.path.dirname(location), f"{os.path.basename(location)}_artifact_removed")
    try:
        os.makedirs(no_artifact_location)
    except:
        pass
    return no_artifact_location


def extract_coords(mask_location, output_dir,
                   coord_order="xyz",      # "xyz" or "zyx" in the output CSV
                   connectivity=1,         # 1=faces-only (6-connectivity in 3D); use 2 or 3 for more
                   min_size=4,             # drop components with fewer than this many voxels
                   include_area=False,     # also save component voxel counts
                   include_label=False,    # also save connected-component labels
                   float_dtype=np.float32  # dtype for centroid coords in CSV
                   ):
    """
    Extract centroids from a 3D binary mask (saved as TIFF) using regionprops.

    Assumes mask array axes are (Z, Y, X). Outputs a CSV with centroid coordinates in the
    desired order. Components smaller than `min_size` are optionally removed.
    """
    os.makedirs(output_dir, exist_ok=True)
    coords_file = os.path.join(output_dir, "nuclei_coords.csv")

    # Load and binarize
    vol = tifffile.imread(mask_location)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {vol.shape}")
    mask = vol > 0

    # Optional size filtering before labeling (works on boolean mask)
    if min_size and min_size > 0:
        mask = remove_small_objects(mask, min_size=min_size)

    # Label connected components (on (Z,Y,X))
    lbl = label(mask, connectivity=connectivity)

    # If empty, write headers and return
    if lbl.max() == 0:
        cols = ["axis-2","axis-1","axis-0"] if coord_order.lower() == "xyz" else ["axis-0","axis-1","axis-2"]
        if include_area: cols.append("area")
        if include_label: cols.append("label")
        pd.DataFrame(columns=cols).to_csv(coords_file, index=False)
        return coords_file

    # Extract centroids (and area/label if requested)
    props = ["centroid"]
    if include_area:  props.append("area")
    if include_label: props.append("label")
    tbl = regionprops_table(lbl, properties=tuple(props))

    # regionprops_table returns centroid-0=z, centroid-1=y, centroid-2=x
    z = np.asarray(tbl["centroid-0"], dtype=float_dtype)
    y = np.asarray(tbl["centroid-1"], dtype=float_dtype)
    x = np.asarray(tbl["centroid-2"], dtype=float_dtype)

    if coord_order.lower() == "xyz":
        data = {"axis-2": x, "axis-1": y, "axis-0": z}
    elif coord_order.lower() == "zyx":
        data = {"axis-0": z, "axis-1": y, "axis-2": x}
    else:
        raise ValueError("coord_order must be 'xyz' or 'zyx'")

    if include_area:
        data["area"] = np.asarray(tbl["area"], dtype=np.int64)
    if include_label:
        data["label"] = np.asarray(tbl["label"], dtype=np.int64)

    df = pd.DataFrame(data)
    df.to_csv(coords_file, index=False)
    return coords_file


from skimage.transform import resize

def _shape_from_indices(chunk_indices):
    # Compute (Z,Y,X) from the max stop across slices
    max_z = max(s[0].stop for s in chunk_indices)
    max_y = max(s[1].stop for s in chunk_indices)
    max_x = max(s[2].stop for s in chunk_indices)
    return (int(max_z), int(max_y), int(max_x))

# --- helper: detect coords in CSV and return (z_local, y_local, x_local) ---
    """ def _get_local_zyx(df, coord_order="zyx"):
    cols = {c.lower(): c for c in df.columns}
    # Support both x/y/z and axis-0/1/2 (axis-0=z, axis-1=y, axis-2=x)
    zcol = cols.get("z") or cols.get("axis-0")
    ycol = cols.get("y") or cols.get("axis-1")
    xcol = cols.get("x") or cols.get("axis-2")
    if not (zcol and ycol and xcol):
        raise ValueError(f"Could not find z/y/x (or axis-0/1/2) in columns: {list(df.columns)}")

    if coord_order.lower() == "zyx":
        z = df[zcol].to_numpy(); y = df[ycol].to_numpy(); x = df[xcol].to_numpy()
    elif coord_order.lower() == "xyz":
        # CSV is x,y,z → map to (z,y,x)
        x = df[xcol].to_numpy(); y = df[ycol].to_numpy(); z = df[zcol].to_numpy()
    elif coord_order.lower() == "xzy":
        x = df[xcol].to_numpy(); z = df[zcol].to_numpy(); y = df[ycol].to_numpy()
    else:
        raise ValueError("coord_order must be 'zyx' or 'xyz' or 'xzy'")
    return z, y, x """

def _get_local_zyx(df, coord_order="zyx"):
    """
    Return (z, y, x) arrays regardless of how the CSV columns are ordered.

    Supports two column naming styles:
      - axis-0, axis-1, axis-2  (in the CSV's coord_order)
      - X, Y, Z                 (explicit)

    coord_order examples: "zyx", "xzy", "xyz", "yxz", etc.
    """
    coord_order = coord_order.lower()

    # Case 1: explicit columns exist
    if all(c in df.columns for c in ["X", "Y", "Z"]):
        x = df["X"].to_numpy()
        y = df["Y"].to_numpy()
        z = df["Z"].to_numpy()
        return z, y, x

    # Case 2: axis-* columns exist, interpret them in coord_order
    if not all(c in df.columns for c in ["axis-0", "axis-1", "axis-2"]):
        raise KeyError(f"CSV must have either X/Y/Z or axis-0/1/2 columns. Got: {list(df.columns)}")

    a0 = df["axis-0"].to_numpy()
    a1 = df["axis-1"].to_numpy()
    a2 = df["axis-2"].to_numpy()

    mapping = {coord_order[0]: a0, coord_order[1]: a1, coord_order[2]: a2}

    # Always return in ZYX semantic order
    return mapping["z"], mapping["y"], mapping["x"]

def _parse_chunk_num(path, prefix=None):
    base = os.path.basename(path)
    if prefix:
        m = re.search(rf"{re.escape(prefix)}\D*(\d+)", base)
        if m:
            return int(m.group(1))
    return int(re.findall(r"\d+", base)[-1])

def combine_centroids_csv(
    centroids_folder,
    #detection_masks_folder,
    vol_unmixed_folder,
    output_folder_scale,
    chunk_indices_name="chunk_indices.npy",
    csv_pattern="centroids*.csv",
    #mask_pattern="mask*.tif",
    centroid_prefix="centroids",   # anchor for parsing chunk id from CSV names
    #mask_prefix="mask",            # anchor for parsing chunk id from mask names
    coord_order="zyx",             # your CSVs appear to be (z,y,x)
    dedupe=True,
    output_file_name="combined_centroids.csv"
    ):
    """
    Mirror the combine_masks placement for centroids:
      1) read raw_img_shape from vol_unmixed_folder/omehans
      2) trim each chunk slice to raw bounds (edge crop)
      3) read the corresponding mask tile to get mask tile shape
      4) scale local CSV coords from mask_shape -> target_chunk_shape
      5) offset by trimmed slice starts
    Writes combined_centroids.csv with columns: z, y, x, chunk
    """
    os.makedirs(output_folder_scale, exist_ok=True)
    out_path = os.path.join(output_folder_scale, output_file_name)

    # 1) load chunk indices & raw image shape EXACTLY like combine_masks
    chunk_indices = np.load(os.path.join(output_folder_scale, chunk_indices_name), allow_pickle=True)

    location = os.path.join(vol_unmixed_folder, "omehans")
    dask_zarray = read_omehans(location)             # <- same call you use
    #lazy_tiff_stack = dask_zarray[0, :, :, :]        # Z,Y,X
    if dask_zarray.ndim == 4:
        lazy_tiff_stack = dask_zarray[0, :, :, :]   # take first channel: (Z, Y, X)
    elif dask_zarray.ndim == 3:
        lazy_tiff_stack = dask_zarray               # already (Z, Y, X)
    else:
        raise ValueError(f"Unexpected volume ndim={dask_zarray.ndim}, shape={dask_zarray.shape}")
    raw_img_shape = tuple(lazy_tiff_stack.shape)     # (Z,Y,X)
    Zfull, Yfull, Xfull = raw_img_shape
    print("raw_img_shape (Z,Y,X):", raw_img_shape)

    # 2) collect CSVs & Masks
    csv_paths = sorted(glob(os.path.join(centroids_folder, csv_pattern)))
    #mask_paths = sorted(glob(os.path.join(detection_masks_folder, mask_pattern)))
    print(f"Found {len(csv_paths)} centroid CSVs and {len(mask_paths)} mask tiles")

    if not csv_paths:
        pd.DataFrame(columns=["axis-0","axis-1","axis-2","chunk"]).to_csv(out_path, index=False)
        return out_path

    # Build map: chunk_id -> mask_path  (anchor to mask_prefix to avoid wrong number)
    mask_for_chunk = {}
    for mp in mask_paths:
        cid = _parse_chunk_num(mp, prefix=mask_prefix)
        mask_for_chunk[cid] = mp

    rows = []
    for cp in csv_paths:
        chunk_id = _parse_chunk_num(cp, prefix=centroid_prefix)
        try:
            z0, y0, x0 = chunk_indices[chunk_id]
        except Exception:
            raise KeyError(f"chunk_indices missing entry for chunk {chunk_id}")

        # 2) TRIM slices to raw bounds (edge crop) – exact mirror of combine_masks
        z_s = slice(z0.start, min(z0.stop, Zfull), z0.step)
        y_s = slice(y0.start, min(y0.stop, Yfull), y0.step)
        x_s = slice(x0.start, min(x0.stop, Xfull), x0.step)

        tz = z_s.stop - z_s.start
        ty = y_s.stop - y_s.start
        tx = x_s.stop - x_s.start
        if tz <= 0 or ty <= 0 or tx <= 0:
            print(f"Skipping chunk {chunk_id}: zero-sized after trim")
            continue
        target_shape = (tz, ty, tx)

        # 3) read the corresponding mask tile to get MASK SHAPE (source space)
        mp = mask_for_chunk.get(chunk_id, None)
        if mp is None:
            raise FileNotFoundError(f"No mask tile found for chunk {chunk_id} using pattern/prefix in {detection_masks_folder}")
        m = tifffile.imread(mp)
        if m.ndim != 3:
            raise ValueError(f"Mask {mp} is not 3D; got shape {m.shape}")
        mask_shape = tuple(m.shape)   # assume Z,Y,X on disk
        # (If your masks were Y,X,Z, add a small axis-order fix here)

        # 4) scale local CSV coords from mask_shape -> target_chunk_shape (mirrors resize in combine_masks)
        sz = target_shape[0] / mask_shape[0]
        sy = target_shape[1] / mask_shape[1]
        sx = target_shape[2] / mask_shape[2]

        df = pd.read_csv(cp)
        z_loc, y_loc, x_loc = _get_local_zyx(df, coord_order=coord_order)  # local coords in MASK SPACE

        # scale then offset by trimmed starts (mirror of mask resizing + paste)
        zg = np.rint(z_loc * sz + z_s.start).astype(np.int64)
        yg = np.rint(y_loc * sy + y_s.start).astype(np.int64)
        xg = np.rint(x_loc * sx + x_s.start).astype(np.int64) 

        """         zg = np.rint(z_loc + z_s.start).astype(np.int64)
        yg = np.rint(y_loc + y_s.start).astype(np.int64)
        xg = np.rint(x_loc + x_s.start).astype(np.int64) """

        # clip to bounds defensively
        zg = np.clip(zg, 0, Zfull - 1)
        yg = np.clip(yg, 0, Yfull - 1)
        xg = np.clip(xg, 0, Xfull - 1)

        rows.append(pd.DataFrame({"axis-0": zg, "axis-1": yg, "axis-2": xg, "chunk": chunk_id}))

        print(f"Chunk {chunk_id}: mask_shape={mask_shape}, target_shape={target_shape}, "
              f"offsets=({z_s.start},{y_s.start},{x_s.start}), placed={len(df)}")

    combined = pd.concat(rows, ignore_index=True)

    if dedupe:
        before = len(combined)
        combined.drop_duplicates(subset=["axis-0","axis-1","axis-2"], inplace=True, ignore_index=True)
        print(f"Deduped exact (z,y,x): {before - len(combined)} removed")

    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} global centroids to {out_path}")
    return out_path

def combine_centroids_noMasks_csv(
    centroids_folder,
    vol_unmixed_folder,
    output_folder_scale,
    chunk_indices_name="chunk_indices.npy",
    csv_pattern="centroids*.csv",
    centroid_prefix="centroids",   # anchor for parsing chunk id from CSV names
    coord_order="zyx",             
    dedupe=True,
    output_file_name="combined_centroids.csv",
    ):
    """
    Combine per-chunk centroids into global (raw-image) coordinates.

    Assumes:
      - chunk_indices[chunk_id] gives (z_slice, y_slice, x_slice) for that chunk
      - centroids in each CSV are already in the SAME resolution as the raw volume
        (i.e. no mask/resize scaling needed)

    Writes combined_centroids.csv with columns: axis-0 (z), axis-1 (y), axis-2 (x), chunk
    """
    os.makedirs(output_folder_scale, exist_ok=True)
    out_path = os.path.join(output_folder_scale, output_file_name)

    # 1) load chunk indices & raw image shape
    chunk_indices = np.load(
        os.path.join(output_folder_scale, chunk_indices_name),
        allow_pickle=True,
    )

    location = os.path.join(vol_unmixed_folder, "omehans")
    dask_zarray = read_omehans(location)      # expect (C, X, Z, Y) or (X, Z, Y)
    if dask_zarray.ndim == 4:
        lazy_tiff_stack = dask_zarray[0, :, :, :]   # take first channel: (X, Z, Y)
    elif dask_zarray.ndim == 3:
        lazy_tiff_stack = dask_zarray               # already (X, Z, Y)
    else:
        raise ValueError(f"Unexpected volume ndim={dask_zarray.ndim}, shape={dask_zarray.shape}")


    #coord_order="xzy"
    lazy_tiff_stack = _to_zyx_from_xzy(lazy_tiff_stack)

    raw_img_shape = tuple(lazy_tiff_stack.shape)    # (Z, Y, X)
    Zfull, Yfull, Xfull = raw_img_shape
    print("raw_img_shape (Z,Y,X):", raw_img_shape)

    # 2) collect CSVs
    csv_paths = sorted(glob(os.path.join(centroids_folder, csv_pattern)))
    print(f"Found {len(csv_paths)} centroid CSVs")

    if not csv_paths:
        pd.DataFrame(columns=["axis-0", "axis-1", "axis-2", "chunk"]).to_csv(out_path, index=False)
        return out_path

    rows = []
    for cp in csv_paths:
        chunk_id = _parse_chunk_num(cp, prefix=centroid_prefix)
        try:
            z0, y0, x0 = chunk_indices[chunk_id]
        except Exception:
            raise KeyError(f"chunk_indices missing entry for chunk {chunk_id}")

        # trim slices to raw bounds (edge crop)
        z_s = slice(z0.start, min(z0.stop, Zfull), z0.step)
        y_s = slice(y0.start, min(y0.stop, Yfull), y0.step)
        x_s = slice(x0.start, min(x0.stop, Xfull), x0.step)

        tz = z_s.stop - z_s.start
        ty = y_s.stop - y_s.start
        tx = x_s.stop - x_s.start
        if tz <= 0 or ty <= 0 or tx <= 0:
            print(f"Skipping chunk {chunk_id}: zero-sized after trim")
            continue
        target_shape = (tz, ty, tx)

        df = pd.read_csv(cp)
        z_loc, y_loc, x_loc = _get_local_zyx(df, coord_order="xzy")

        # no scaling, just offset
        zg = np.rint(z_loc + z_s.start).astype(np.int64)
        yg = np.rint(y_loc + y_s.start).astype(np.int64)
        xg = np.rint(x_loc + x_s.start).astype(np.int64)

        # clip to bounds
        zg = np.clip(zg, 0, Zfull - 1)
        yg = np.clip(yg, 0, Yfull - 1)
        xg = np.clip(xg, 0, Xfull - 1)

        rows.append(
            pd.DataFrame(
                {
                    "axis-0": zg,
                    "axis-1": yg,
                    "axis-2": xg,
                    "chunk": chunk_id,
                }
            )
        )

        print(
            f"Chunk {chunk_id}: target_shape={target_shape}, "
            f"offsets=({z_s.start},{y_s.start},{x_s.start}), placed={len(df)}"
        )

    combined = pd.concat(rows, ignore_index=True)

    if dedupe:
        before = len(combined)
        combined.drop_duplicates(subset=["axis-0", "axis-1", "axis-2"], inplace=True, ignore_index=True)
        print(f"Deduped exact (z,y,x): {before - len(combined)} removed")

    combined_xzy = combined[["axis-2", "axis-0", "axis-1"]]
    combined_xzy.columns = ["axis-0", "axis-1", "axis-2"]

    combined_xzy.to_csv(out_path, index=False)

    #combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} global centroids to {out_path}")
    return out_path




def _parse_chunk_num(path, prefix="spectral_chunk"):
    # Parses e.g. ".../spectral_chunk_00017.csv" -> 17
    bn = os.path.basename(path)
    # Prefer anchored prefix if present
    if prefix and bn.startswith(prefix):
        m = re.search(r"(\d+)", bn[len(prefix):])
    else:
        m = re.search(r"(\d+)", bn)
    if not m:
        raise ValueError(f"Cannot parse chunk id from {bn} with prefix='{prefix}'")
    return int(m.group(1))

def combine_spectral_info_csv(
    spectral_folder,
    output_path,
    csv_pattern="spectral_chunk_*.csv",
    spectral_prefix="spectral_chunk",
    coord_cols=("axis-0","axis-1","axis-2"),
    round_coords=True,        # rint to voxel indices if you want strict grid coords
    dedupe=True,
    dedupe_strategy="max_vol_l1",  # 'first' | 'max_vol_l1' | 'mean'
):
    """
    Combine spectral chunk CSVs that already contain GLOBAL coordinates (axis-0/1/2).

    - Adds a 'chunk' column derived from filename if missing.
    - Optionally rounds coords to integer voxels.
    - Optional deduplication for overlap regions:
        * 'first': keep first occurrence
        * 'max_vol_l1': keep row with largest vol_l1 (requires 'vol_l1' column)
        * 'mean': group by coords and average numeric columns
    """
    paths = sorted(glob(os.path.join(spectral_folder, csv_pattern)))
    if not paths:
        raise FileNotFoundError(f"No spectral CSVs matched {csv_pattern} in {spectral_folder}")

    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        # Ensure coord columns exist
        for c in coord_cols:
            if c not in df.columns:
                raise KeyError(f"{c} missing in {p}")

        # Add chunk id column (if not present)
        if "chunk" not in df.columns:
            df["chunk"] = _parse_chunk_num(p, prefix=spectral_prefix)

        # Optional rounding to integer grid
        if round_coords:
            df[list(coord_cols)] = np.rint(df[list(coord_cols)].to_numpy()).astype(np.int64)

        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if dedupe and not combined.empty:
        if dedupe_strategy == "first":
            combined = combined.drop_duplicates(subset=list(coord_cols), keep="first", ignore_index=True)

        elif dedupe_strategy == "max_vol_l1":
            if "vol_l1" in combined.columns:
                # Keep row with largest vol_l1 per voxel coordinate
                combined = (combined
                            .sort_values("vol_l1", ascending=False)
                            .drop_duplicates(subset=list(coord_cols), keep="first")
                            .reset_index(drop=True))
            else:
                # Fallback to 'first' if vol_l1 is not present
                combined = combined.drop_duplicates(subset=list(coord_cols), keep="first", ignore_index=True)

        elif dedupe_strategy == "mean":
            # Average numeric columns per voxel coordinate
            num_cols = combined.select_dtypes(include=[np.number]).columns.tolist()
            # Preserve coords as int
            grp = (combined
                   .groupby(list(coord_cols), as_index=False)[num_cols]
                   .mean(numeric_only=True))
            # If 'chunk' exists, drop it or keep a representative:
            if "chunk" in grp.columns:
                grp = grp.drop(columns=["chunk"], errors="ignore")
            combined = grp

        else:
            raise ValueError(f"Unknown dedupe_strategy: {dedupe_strategy}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"[OK] Wrote {len(combined)} rows to {output_path}")
    return output_path
