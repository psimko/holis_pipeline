"""
This script finds new .fli files and sends them to the hemibrain pipeline.

- run on a folder (assume one folder for all files)
- find all fli file pairs
- exclude the ones already processed
- exclude the ones currently being processed
- send SLURM CPU task for remaining pairs
"""
import os
import sys
from datetime import datetime
from glob import glob

import os
import subprocess

# Configuration
input_dir = "/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Slab6/2025_08_22_HOLiS_NPBB328_Cortex_Slab06/"
output_dir = "/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/"
slurm_template = """#!/bin/bash
#SBATCH --job-name=holis_job
#SBATCH --output={output_log}
#SBATCH --mem=512G
#SBATCH --cpus-per-task=16
#SBATCH --partition=compute

/bil/users/noisysky/.conda/envs/holis-pytorch/bin/python /bil/users/noisysky/code/holis_pipeline/holis_pipeline_hicam.py "{input_file}" "{output_dir}"
"""


# Main script
file_glob = glob(os.path.join(input_dir, "*-z01-y*-Exc-*-272.fli.zst"))
print(">>>>>>>>>>> Files:", len(file_glob))
for filename in file_glob:
    input_file = os.path.join(input_dir, filename)
    output_log = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.out")
    job_script_content = slurm_template.format(input_file=input_file, output_dir=output_dir, output_log=output_log)

    job_script_path = os.path.join(output_dir, f"job_{os.path.splitext(filename)[0]}.sh")
    with open(job_script_path, "w") as f:
        f.write(job_script_content)

    subprocess.run(["sbatch", job_script_path])
    print(f"Submitted job for {filename}")
