"""
Save all composite planes for a given z_scan via SLURM.
One job per z_scan, each job loops over all z_opts internally.
 
LAUNCHER MODE (called by you):
    python slurm_save_composites.py <root_dir> <output_dir>
 
WORKER MODE (called by SLURM):
    python slurm_save_composites.py --z-scan Z --root-dir DIR --output-dir DIR
"""
 
import os
import subprocess
import argparse
from dataclasses import dataclass
 
 
##########################################################################################
@dataclass(frozen=True)
class Config:
    channel:     int = 0
    n_zscan:     int = 13
    n_zopt:      int = 508
    folder_glob: str = "*-088.fli*_transformed*"
    partition:   str = "compute"
    mem_gb:      int = 1024
    cpus:        int = 16
    python_path: str = "/bil/users/psimko/.conda/envs/stack_to_multiscale_ngff/bin/python"
 
CFG = Config()
##########################################################################################
 
slurm_template = """#!/bin/bash
#SBATCH --job-name=composite_z{z_scan:02d}
#SBATCH --mem={mem_gb}G
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH -o {stdout_log}
#SBATCH -e {stderr_log}
set -euo pipefail
 
"{python_path}" "{this_script}" --z-scan {z_scan} --root-dir "{root_dir}" --output-dir "{output_dir}" --channel {channel} --folder-glob "{folder_glob}" --n-zopt {n_zopt}
"""
 
##########################################################################################
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    # Worker mode args
    parser.add_argument("--z-scan",      type=int, default=None)
    parser.add_argument("--root-dir",    default=None)
    parser.add_argument("--output-dir",  default=None)
    parser.add_argument("--channel",     type=int, default=CFG.channel)
    parser.add_argument("--folder-glob", default=CFG.folder_glob)
    parser.add_argument("--n-zopt",      type=int, default=CFG.n_zopt)
 
    # Launcher mode positional args
    parser.add_argument("root_dir_pos",   nargs="?", default=None)
    parser.add_argument("output_dir_pos", nargs="?", default=None)
 
    args = parser.parse_args()
 
    # ---- WORKER MODE: called by SLURM, loops over all z_opts ----
    if args.z_scan is not None:
        for z_opt in range(args.n_zopt):
            print(f"z_scan={args.z_scan}  z_opt={z_opt}")
            subprocess.run([
                args.python_path if hasattr(args, 'python_path') else "python",
                "save_composite_plane.py",
                "--root_dir",    args.root_dir,
                "--output_dir",  args.output_dir,
                "--z_scan",      str(args.z_scan),
                "--z_opt",       str(z_opt),
                "--channel",     str(args.channel),
                "--folder_glob", args.folder_glob,
            ], check=True)
        raise SystemExit(0)
 
    # ---- LAUNCHER MODE: submits one job per z_scan ----
    root_dir   = args.root_dir_pos
    output_dir = args.output_dir_pos
 
    if root_dir is None or output_dir is None:
        raise SystemExit("Usage: python slurm_save_composites.py <root_dir> <output_dir>")
 
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
 
    this_script = os.path.realpath(__file__)
 
    for z_scan in range(1, CFG.n_zscan + 1):
        job_script_content = slurm_template.format(
            z_scan      = z_scan,
            mem_gb      = CFG.mem_gb,
            cpus        = CFG.cpus,
            partition   = CFG.partition,
            python_path = CFG.python_path,
            this_script = this_script,
            root_dir    = root_dir,
            output_dir  = output_dir,
            channel     = CFG.channel,
            folder_glob = CFG.folder_glob,
            n_zopt      = CFG.n_zopt,
            stdout_log  = os.path.join(logs_dir, f"z{z_scan:02d}.out"),
            stderr_log  = os.path.join(logs_dir, f"z{z_scan:02d}.err"),
        )
 
        job_script_path = os.path.join(logs_dir, f"job_z{z_scan:02d}.sh")
        with open(job_script_path, "w") as f:
            f.write(job_script_content)
 
        subprocess.run(["sbatch", job_script_path], check=True)
        print(f"Submitted z_scan={z_scan}")
 
    print(f"Done — submitted {CFG.n_zscan} jobs")