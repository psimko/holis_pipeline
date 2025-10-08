import os
import subprocess
from pathlib import Path
from glob import glob


def extract_small_tiffs():
    stripe_paths = [f"/h20/CBI/Iana/projects/hillman/hemibrain/slab2/out/NPBB328-Cortex-Slab02-run{str(i+1).zfill(3)}-z01-y{str(i).zfill(3)}-Exc-488nm-561nm-594nm-660nm_HiCAM FLUO_1875-ST-272.fli.zst" for i in range(1, 71)]
    stripe_paths = sorted([x for x in stripe_paths if os.path.exists(x)])
    print(len(stripe_paths))
    print(*stripe_paths, sep="\n")

    # Configuration
    # input_dir = "/h20/Public/holis/2025_06_15_NPBB328_surface"  # Replace with your folder path
    output_dir = "/h20/CBI/Iana/projects/hillman/hemibrain/slab2/out_tiffs"  # Replace with your desired output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'scripts'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)

    slurm_template = """#!/bin/bash
#SBATCH --job-name=holis_job
#SBATCH --output="{output_log}"
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu

source ~/.bashrc
source /h20/home/lab/miniconda3/bin/activate holis
python /h20/CBI/Iana/src/holis_pipeline/zarr_to_tiffs.py "{input_file}" "{output_dir}" "{stripe_number}"
"""

    # Main script
    for y, stripe_path in enumerate(stripe_paths):
        input_file = stripe_path
        # input_file = os.path.join(stripe_path, 'array.zarr')
        output_log = os.path.join(output_dir, 'logs', f"{y:04d}.out")
        job_script_content = slurm_template.format(input_file=input_file, output_dir=output_dir, output_log=output_log, stripe_number=y)

        job_script_path = os.path.join(output_dir, 'scripts', f"job_{y:04d}.sh")
        with open(job_script_path, "w") as f:
            f.write(job_script_content)

        subprocess.run(["sbatch", job_script_path])
        print(f"Submitted job for {input_file}")


def build_composite_tiffs():
    source_dir = Path("/h20/CBI/Iana/projects/hillman/hemibrain/slab2/out_tiffs")
    output_dir = Path("/h20/CBI/Iana/projects/hillman/hemibrain/slab2/out_composites")
    os.makedirs(output_dir, exist_ok=True)
    script_path = Path("/h20/CBI/Iana/src/holis_pipeline/tiffs_to_composites.py")

    crop_top = 395
    crop_bottom = 5
    num_z_planes = 1024
    # num_z_planes = 2

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir/"logs", exist_ok=True)

    for z in range(num_z_planes):
        output_tif = output_dir / f"composite_z{z:04d}.tif"
        job_name = f"holis_stitch_z{z:04d}"

        cmd = [
            "sbatch",
            "--job-name", job_name,
            "--output", f"{output_dir}/logs/{job_name}_%j.out",
            "--error", f"{output_dir}/logs/{job_name}_%j.err",
            "--ntasks=1",
            "--cpus-per-task=4",
            "--mem=128G",
            "--partition=gpu",
            "--wrap", ( 
                f"/h20/home/iana/anaconda3/envs/holis/bin/python {script_path} {z} {source_dir} {output_tif} "
                f"--crop_top {crop_top} --crop_bottom {crop_bottom}"
            )
        ]

        print(f"Submitting Z={z}...")
        subprocess.run(cmd)


def read_corrections_to_zarr():
    from holis_pipeline.read_data import read_fli_as_zarr
    correction_path = "/h20/Public/holis/2025_06_15_NPBB328_surface_corrections/"
    # print("Reading FF")
    ffNuclei_path = correction_path + 'externalEpoxy-FF-run001-LEDblue_HiCAM FLUO_1875-ST-272.fli'
    read_fli_as_zarr(ffNuclei_path, os.path.join(os.path.dirname(correction_path), os.path.basename(ffNuclei_path).replace('.fli', '.omehans')))

    # print("Reading FFbg")
    ffbgNuclei_path = correction_path + "externalEpoxy-Laser-run001-darkFrames_HiCAM FLUO_1875-ST-272.fli"
    read_fli_as_zarr(ffbgNuclei_path, os.path.join(os.path.dirname(correction_path), os.path.basename(ffbgNuclei_path).replace('.fli', '.omehans')))

    lasers = [488, 561, 594, 660]
    for laser in lasers:
        corr = glob(correction_path + f'NPBB328-corrections-epoxy-run*{laser}nm_HiCAM FLUO_1875-ST-272.fli')[0]
        read_fli_as_zarr(corr, os.path.join(os.path.dirname(correction_path), os.path.basename(corr).replace('.fli', '.omehans')))

    corrbgNuclei = correction_path + "NPBB328-corrections-run001-darkFrames_HiCAM FLUO_1875-ST-272.fli"
    read_fli_as_zarr(corrbgNuclei, os.path.join(os.path.dirname(correction_path), os.path.basename(corrbgNuclei).replace('.fli', '.omehans')))


if __name__ == "__main__":
    # extract_small_tiffs()
    build_composite_tiffs()
