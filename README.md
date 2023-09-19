# holis_pipeline
HOLiS pipeline for nuclei detection and extraction of spectral information

This branch is for analyzing mouse brains.<br/>
This branch is for ome-zarr data.<br/>
This branch introduces the new method for extracting
spectral information based on nuclei segmentation mask and
adding volumetric layers around it.

Current version expects to have foreground/background mask and mask of very bright signal areas (like cerebellum in the mouse brain) that distort the detection.
1) Allocate resources for fusing tiffs and creating zarr (steps 2-4) using `salloc --ntasks=1 --mem 2900G --cpus-per-task 80`. Load miniconda3 module, activate the virtual environment for conversion.
2) Fuse stripes into composite tiff planes, using `fuse_tiffs.py` script (`python fuse_tiffs.py <path_to_tiff_stacks> <path_to_output_folder>`)
3) Make sure the composite tiffs are arranged into folders (nuclei/1, colors/2, colors/3, colors/4, colors/5).
4) Build ome-zarr stores for nuclei and colors separately using `stack_to_multiscale_ngff` package.
Example command for nuclei: `python -i builder.py /bil/proj/rf1hillman/results/2023_08_15_combinatorialSlides2_AI7_EH5f/composites/nuclei /bil/proj/rf1hillman/results/2023_08_15_combinatorialSlides2_AI7_EH5f/omezarr/nuclei.omehans -s 1 1 1.34 1.54 2.0 --clevel 5 -ft tif -tmp '/scratch/tmp_convert' -sk --colors green --channelLabels SytoG24 --name AI7_EH5f3 -mem 2900`<br/>
Example command for colors: `python -i builder.py /bil/proj/rf1hillman/results/2023_08_15_combinatorialSlides2_AI7_EH5f/composites/colors /bil/proj/rf1hillman/results/2023_08_15_combinatorialSlides2_AI7_EH5f/omezarr/colors.omehans -s 1 1 1.34 1.54 2.0 --clevel 5 -ft tif -tmp '/scratch/tmp_convert' -sk --colors green yellow orange red --channelLabels NeuN GAD1_ACTA2 PV_GFAP nNOS_lba1 --name AI7_EH5f3 -mem 2900`
5) From ome-zarr, extract low resolution level (typically scale 2-4), using `extract_low_resolution` function from `holis_segment_foreground_and_cerebellum.py` script.
6) Create the foreground/background mask and mask of very bright signal areas on low-resolution version of the data, using napari_apoc plugin and save to the output folder with names "bg_fg_mask.tif" and "bright_mask.tif".
7) Generate masks for all the chunks by running the `holis_segment_foreground_and_cerebellum.py` script.
It will create the masks in the output folder, in the subfolder called scale_x
8) Change the holis_pipeline/utils/settings.py file to specify the required parameters
9) run `interact` to get to a large-memory-node, then `module load miniconda3`
10) Activate your large-memory-node environment
11) Run the pipeline: `python_holis_pipeline_slab_mouse.py`
12) Re-run if not all spectral information jobs got submitted
13) Once it's built the final csv file, find it at your output directory/scale_0/
