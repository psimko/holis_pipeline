import os, re, subprocess
from pathlib import Path
import argparse, os, re
from pathlib import Path
import numpy as np
from tifffile import imwrite
from holis_pipeline.utils.zarr_related import read_omehans
from dataclasses import dataclass

##########################################################################################
@dataclass(frozen=True)
class Config:

    channel     = 1
    processing  = "transformed"
    partition   = "compute"
    mem_gb      = 1024
    cpus        = 16
    python_path: str = "/bil/users/psimko/.conda/envs/stack_to_multiscale_ngff/bin/python"

CFG = Config()
##########################################################################################

def _y_key(p: Path) -> int:
    """Numeric sort by yNNN anywhere in the path/name; non-matches last."""
    m = re.search(r'y(\d+)', p.as_posix(), flags=re.IGNORECASE)
    return int(m.group(1)) if m else 10**9


def slurm_omehans_as_tiffs(channel, path_ome, tiffs_location, as_volume, target_z=800):
    path_ome = Path(path_ome)
    tiffs_location = Path(tiffs_location)
    image_dask = read_omehans(path_ome)
    print("Shape of the strip", image_dask.shape)
    ndim = image_dask.ndim

    y_name = path_ome.parent.name
    out_dir = os.path.join(tiffs_location, y_name)
    os.makedirs(out_dir, exist_ok=True)

    if ndim == 3:
        X, Z, Y = image_dask.shape
        if not as_volume:
            for z in range(Z):
                if z != target_z:
                    continue
                out_path = os.path.join(out_dir, f"stripe_{z:04d}.tif") if channel == 0 else os.path.join(out_dir, f"stripe_z{z:04d}_colors.tif")
                if os.path.exists(out_path):
                    continue
                print(f'Saving plane={z}')
                plane = image_dask[:, z, :].astype("float32").compute()
                imwrite(out_path, plane)
        else:
            out_path = os.path.join(out_dir, "vol.tif") if channel == 0 else os.path.join(out_dir, "vol_colors.tif")
            if not os.path.exists(out_path):
                print(f'Saving volume {y_name}')
                vol = image_dask[:, :, :].astype("float32").compute()
                imwrite(out_path, vol)

    elif ndim == 4:
        # shape: (C, X, Z, Y)
        C, X, Z, Y = image_dask.shape
        if not as_volume:
            for z in range(Z):
                if z != target_z:
                    continue
                for c in range(C):
                    out_path = os.path.join(out_dir, f"stripe_c{c:02d}_z{z:04d}.tif")
                    if os.path.exists(out_path):
                        continue
                    print(f"Saving plane z={z}, channel={c}")
                    print("Reading into memory")
                    plane = image_dask[c, :, z, :].astype("float32").compute()
                    print("done")
                    imwrite(out_path, plane)
        else:
            for c in range(C):
                if channel == 0:
                    out_path = os.path.join(out_dir, f"vol.tif")
                else:
                    out_path = os.path.join(out_dir, f"vol_colors_ch{c}.tif")
                if os.path.exists(out_path):
                    continue
                print(f'Saving volume channel{c} to {y_name}')
                print("Reading into memory")
                vol = image_dask[c, :, :, :].astype("float32").compute()
                print("Done")
                imwrite(out_path, vol)
    else:
        raise ValueError(f"Unsupported image shape {image_dask.shape}")


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

"{python_path}" "{this_script}" --channel "{channel}" --path-ome "{path_ome}" --tiffs-location "{tiffs_location}" --as-volume {as_volume} --target-z {target_z}
"""

##########################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=int, default=None)
    parser.add_argument("--path-ome", default=None)         # worker mode if set
    parser.add_argument("--tiffs-location", default=None)
    parser.add_argument("--as-volume", default="False")
    parser.add_argument("--target-z", type=int, default=800)

    parser.add_argument("channel", type=int, nargs="?", default=1)
    parser.add_argument("processing", nargs="?", default=None,
                        type=lambda s: None if s is not None and s.lower() in ("none", "null", "") else s)
    parser.add_argument("as_volume_pos", nargs="?", default="False")
    parser.add_argument("source_dir", nargs="?")
    parser.add_argument("output_dir", nargs="?")
    args = parser.parse_args()

    # ---- WORKER MODE: called by Slurm ----
    if args.path_ome is not None:
        if args.tiffs_location is None:
            raise SystemExit("Worker mode: --tiffs-location is required")
        as_volume = args.as_volume.lower() == "true"
        slurm_omehans_as_tiffs(args.channel, args.path_ome, args.tiffs_location, as_volume, args.target_z)
        raise SystemExit(0)

    # ---- LAUNCHER MODE: called by you ----
    channel    = args.channel
    processing = args.processing
    as_volume  = args.as_volume_pos
    omehans_location = args.source_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    nuc_file_pattern   = "*-272.fli*"
    color_file_pattern = "*-088.fli*"

    if channel == 0:
        suffix_map = {
            None:              nuc_file_pattern,
            "bg_subtracted":   nuc_file_pattern + "_bg_subtracted",
            "laser_corrected": nuc_file_pattern + "_laser_corrected",
            "transformed":     nuc_file_pattern + "_transformed",
            "registered":      color_file_pattern + "_registered",
            "unmixed":         color_file_pattern + "_unmixed",
        }
    else:
        suffix_map = {
            None:              color_file_pattern,
            "bg_subtracted":   color_file_pattern + "_bg_subtracted",
            "laser_corrected": color_file_pattern + "_laser_corrected",
            "transformed":     color_file_pattern + "_transformed",
            "registered":      color_file_pattern + "_registered",
            "unmixed":         color_file_pattern + "_unmixed",
        }

    wanted_name = suffix_map.get(processing)
    print(f'Wanted name: {wanted_name}')
    if wanted_name is None:
        raise ValueError(f"Unknown processing='{processing}'. Expected one of {list(suffix_map)}")

    paths     = sorted(Path(omehans_location).glob(wanted_name), key=_y_key)
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
            channel=channel,
            job_name=job_name,
            mem_gb=CFG.mem_gb,
            cpus=CFG.cpus,
            partition=CFG.partition,
            python_path=CFG.python_path,
            this_script=this_script,
            path_ome=str(path_ome),
            tiffs_location=output_dir,
            as_volume=as_volume,
            target_z=800,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )

        job_script_path = os.path.join(output_dir, f"job_{root}.sh")
        print(f'job_script_path: {job_script_path}')
        with open(job_script_path, "w") as f:
            f.write(job_script_content)

        subprocess.run(["sbatch", job_script_path])
        print(f"Submitting {job_name} -> {path_ome}")
        




