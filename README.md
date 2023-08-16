# holis_pipeline
HOLiS pipeline for nuclei detection and extraction of spectral information

Current version expects to have foreground/background mask and mask of very bright signal areas (like cerebellum in the mouse brain) that distort the detection.
1) Fuse stripes into composite tiff planes, using `fuse_tiffs.py` script (`python fuse_tiffs.py <path_to_tiff_stacks> <path_to_output_folder>`)
2) Rearrange the composite tiffs into folders (nuclei/1, colors/2, colors/3, colors/4, colors/5).
3) Build ome-zarr stores for nuclei and colors separately using `stack_to_multiscale_ngff` package.
4) From ome-zarr, extract low resolution level (typically scale 2-4), using `extract_low_resolution` function from `holis_segment_foreground_and_cerebellum.py` script.
5) Create the foreground/background mask and mask of very bright signal areas on low-resolution version of the data, using napari_apoc plugin and save to the output folder with names "bg_fg_mask.tif" and "bright_mask.tif".
6) Generate masks for all the chunks by running the `holis_segment_foreground_and_cerebellum.py` script.
It will create the masks in the output folder, in the subfolder called scale_x
7) Change the holis_pipeline/utils/settings.py file to specify the required parameters
8) run `interact` to get to a large-memory-node, then `module load miniconda3`
9) Activate your large-memory-node environment
10) Run the pipeline: `python_holis_pipeline_slab_mouse.py`
11) Re-run if not all spectral information jobs got submitted
12) Once it's built the final csv file, find it at your output directory/scale_0/
