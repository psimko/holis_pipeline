import os
import shutil
from pathlib import Path

# Set source and destination folders
source_folder = Path("/bil/proj/rf1hillman/2025_06_28_NPBB328_surface_processedTiffs_fused_SLURM")         
destination_folder = Path("/bil/proj/rf1hillman/2025_06_28_NPBB328_surface_processedTiffs_fused_SLURM_organized/")    

# Ensure the destination folder exists
destination_folder.mkdir(parents=True, exist_ok=True)

# Accepted channel suffixes
valid_suffixes = ["_c1.tiff", "_c2.tiff", "_c3.tiff", "_c4.tiff", "_c5.tiff"]

# Loop through all files in the source folder
for file in source_folder.iterdir():
    if file.is_file():
        for suffix in valid_suffixes:
            if file.name.endswith(suffix):
                channel = suffix[2]  # e.g., '1' from ".c1.tiff"
                target_subfolder = destination_folder / channel
                target_subfolder.mkdir(parents=True, exist_ok=True)

                shutil.copy2(file, target_subfolder / file.name)
                print(f"Copied {file.name} -> {target_subfolder}")
                break  # stop checking other suffixes

print("Done.")

