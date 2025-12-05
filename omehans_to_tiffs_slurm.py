import os, re, subprocess
from pathlib import Path
import argparse, os, re
from pathlib import Path
import numpy as np
from tifffile import imwrite
from holis_pipeline.utils.zarr_related import read_omehans
from dataclasses import dataclass

##########################################################################################
@dataclass(frozen=True) # this disallows mutability
class Config:

    channel     = 0                   # 0 for nuclei, else splitter
    processing  = "bg_subtracted"               # or "bg_subtracted" / "laser_corrected" / "registered" / "unmixed"
    partition   = "compute"           # "compute" or "gpu"
    mem_gb      = 512                 # adjust for your strip size
    cpus        = 16
    python_path: str = "/bil/users/psimko/.conda/envs/stack_to_multiscale_ngff/bin/python"

CFG = Config()
##########################################################################################

def _y_key(p: Path) -> int:
    """Numeric sort by yNNN anywhere in the path/name; non-matches last."""
    m = re.search(r'y(\d+)', p.as_posix(), flags=re.IGNORECASE)
    return int(m.group(1)) if m else 10**9

def slurm_omehans_as_tiffs(path_ome, tiffs_location):
        path_ome = Path(path_ome)
        tiffs_location = Path(tiffs_location)
        image_dask = read_omehans(path_ome)  # (37000, 1024, 1280)
        print("Shape of the strip", image_dask.shape)
        X, Z, Y = image_dask.shape
        y_name = path_ome.parent.name
        out_dir = os.path.join(tiffs_location, y_name)
        os.makedirs(out_dir, exist_ok=True)
        for z in range(Z):
            if not os.path.exists(f"{out_dir}/stripe_{z:04d}.tif"):
                target_z = 800
                if z == target_z: #if z % 100 == 0:
                    print(f'Saving plane={z}')
                    print("Reading into memory")
                    plane = image_dask[:, z, :].astype("float32").compute() 
                    print("done")
                    imwrite(f"{out_dir}/stripe_{z:04d}.tif", plane)
                else:
                    continue

##########################################################################################
# SLURM template
slurm_template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --mem={mem_gb}G
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH -o {stdout_log}
#SBATCH -e {stderr_log}
set -euo pipefail

"{python_path}" "{this_script}" --path-ome "{path_ome}" --tiffs-location "{tiffs_location}"
"""

##########################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-ome", default=None)     # worker mode if set
    parser.add_argument("--tiffs-location", default=None)

    parser.add_argument("channel", type=int, nargs="?", default=0)
    parser.add_argument("processing", nargs="?", default=None)
    parser.add_argument("source_dir", nargs="?")
    parser.add_argument("output_dir", nargs="?")
    args = parser.parse_args()

    # ---- WORKER MODE: called by Slurm ----
    if args.path_ome is not None:
        if args.tiffs_location is None:
            raise SystemExit("Worker mode: --tiffs-location is required")
        slurm_omehans_as_tiffs(args.path_ome, args.tiffs_location)
        raise SystemExit(0)

    # ---- LAUNCHER MODE: called by you ----
    channel = args.channel
    processing = args.processing
    omehans_location = args.source_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Map the processing stage to the folder suffix you need to open

    nuc_file_pattern = "*-272.fli*"
    color_file_pattern = "*-088.fli*"

    if channel==0:
        suffix_map = {
            None:              nuc_file_pattern,
            "bg_subtracted":   nuc_file_pattern + "_bg_subtracted",
            "laser_corrected": nuc_file_pattern + "_laser_corrected",
            "registered":      color_file_pattern + "_registered",
            "unmixed":         color_file_pattern + "_unmixed",
        }
    else:
        suffix_map = {
            None:              color_file_pattern,
            "bg_subtracted":   color_file_pattern + "_bg_subtracted",
            "laser_corrected": color_file_pattern + "_laser_corrected",
            "registered":      color_file_pattern + "_registered",
            "unmixed":         color_file_pattern + "_unmixed",
        }
    wanted_name = suffix_map.get(processing)
    print(f'Wanted name: {wanted_name}')
    if wanted_name is None:
        raise ValueError(f"Unknown processing='{processing}'. Expected one of {list(suffix_map)}")

    paths = sorted(Path(args.source_dir).glob(wanted_name), key=_y_key)
    paths_ome = [p / "omehans" for p in paths]
    print("files found:", len(paths_ome))

    this_script = os.path.realpath(__file__)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    for path_ome in paths_ome:
        root = path_ome.parent.name.replace(" ", "_")
        job_name = f"tiff_{root}"

        stdout_log = os.path.join(logs_dir, f"{root}.out")
        stderr_log = os.path.join(logs_dir, f"{root}.err")

        job_script_content = slurm_template.format(
            job_name=job_name,
            mem_gb=CFG.mem_gb,
            cpus=CFG.cpus,
            partition=CFG.partition,
            python_path=CFG.python_path,
            this_script=this_script,
            path_ome=str(path_ome),
            tiffs_location=output_dir,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )

        job_script_path = os.path.join(output_dir, f"job_{root}.sh")
        print(f'job_script_path: {job_script_path}')
        with open(job_script_path, "w") as f:
            f.write(job_script_content)

        subprocess.run(["sbatch", job_script_path])
        print(f"Submitting {job_name} -> {path_ome}")

        




