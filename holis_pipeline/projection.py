import tifffile
import numpy as np
import os

vol = tifffile.imread('/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_composites/crop_tests/composite_colors_shifted_originalCropMinusYadded_ZTopcrop30_ZBottomcrop30_YRightcrop44_YLeftcrop44.tif')  # loads as numpy array
output_dir = '/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_composites/ZTopcrop30_ZBottomcrop30_YRightcrop44_YLeftcrop44_transformed_projections/'
os.makedirs(output_dir, exist_ok=True)

for ax in range(3):
    max_proj = np.max(vol, axis=ax)
    mean_proj = np.mean(vol, axis=ax)
    sum_proj = np.sum(vol, axis=ax)
    median_proj = np.median(vol, axis=ax)

    np.save(os.path.join(output_dir, f'median_projection_axis{ax}.npy'), median_proj)
    np.save(os.path.join(output_dir, f'max_projection_axis{ax}.npy'), max_proj)
    np.save(os.path.join(output_dir, f'median_projection_axis{ax}.npy'), median_proj)
    np.save(os.path.join(output_dir, f'mean_projection_axis{ax}.npy'), mean_proj)
    np.save(os.path.join(output_dir, f'median_projection_axis{ax}.npy'), median_proj)
    np.save(os.path.join(output_dir, f'sum_projection_axis{ax}.npy'), sum_proj)