# holis_pipeline
HOLiS pipeline for nuclei detection and extraction of spectral information

Current version expects to have foreground/background mask and mask of very bright signal areas (like cerebellum in the mouse brain) that distort the detection.
1) Create these masks on low-resolution version of the data, using napari_apoc plugin and save to the output folder with names "bg_fg_mask.tif" and "bright_mask.tif".
2) Generate masks for all the chunks by modifying and running the `holis_segment_foreground_and_cerebellum.py` script.
It will create the masks in the output folder, in the subfolder called scale_x
3) Change the holis_pipeline/utils/settings.py file to specify the required parameters
4) run `interact` to get to a large-memory-node, then `module load miniconda3`
5) Activate your large-memory-node environment
6) Run the pipeline: `python_holis_pipeline_slab_mouse.py`
7) Re-run if not all spectral information jobs got submitted
8) Once it's built the final csv file, find it at your output directory/scale_0/
