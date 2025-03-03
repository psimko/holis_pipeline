# holis_pipeline
HOLiS pipeline for nuclei detection and extraction of spectral information

This branch is for analyzing the human hemibrain.<br/>
This branch is for HICAM data.<br/>
This branch introduces pre-processing that is run before nuclei detection and spectral extraction.

The pipeline works on one pair of spool (.fli) files

Changes need to be made in the settings.py file to set paths and file naming patterns.

Takes in one spool file for nuclei and corresponding spool file for colors.
Outputs detected nuclei and extracted spectral information.

inputs:
- 1 hicam (.fli) file (nuclei - camera the suffix is 272)
- desired output location

Corresponding hicam (.fli) file for colors (camera the suffix is 088) is found automatically

1) read fli files to zarr (nuclei, colors)
2) subtract background (nuclei, colors)
3) split color channels (colors)
4) do laser correction (nuclei, colors)
5) unmixing (nuclei, colors)
6) chunk (nuclei)
7) run pytorch unet on all chunks
8) fix chunking artifacts
9) unchunk (nuclei, colors)
10) save coordinates (nuclei)
11) color_registration (colors)
12) get spectral information on all chunks (nuclei, colors)
13) delete intermediate files (zarr, preprocessed zarr, any chunks)

outputs:
- 1 combined mask of nuclei
- 1 csv file with coordinates
- 1 csv file with spectral information

To run the pipeline:

1) Change the holis_pipeline/utils/settings.py file to specify other required parameters
2) run `interact` to get to a large-memory-node, then `module load miniconda3`
3) Activate your large-memory-node environment
4) Run the pipeline: `python holis_pipeline_hemibrain.py`
