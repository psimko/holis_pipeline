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
import re
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass
from get_correction_masks_hicam_hemibrain1 import normalize_patterns, mix_patterns
import json
import base64
from holis_pipeline.preprocessing_functions import infer_z, infer_y



# ---------- Config (edit all inputs here) ----------
@dataclass(frozen=True) # this disallows mutability
class Config:
    # What to do
    convert_to_ome: bool = True
    get_bg_masks: bool = False
    get_patterns: bool = False
    norm_and_mix_patterns: bool = False
    process: bool = False
    segment: bool = False

    # Save intermediate results
    save_bg_subtracted: bool = False
    save_laser_corrected: bool = False
    save_transformed: bool = False

    # Transforms and other .mat info path
    transforms_matlab_path = "/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Matlab_info/NPBB328_colorMerge_transforms.mat"   
    matrices_matlab_path = "/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Matlab_info/NPBB328_SimulationMatrices_equalPower.mat"

    # Don't forget to change the dates too (in the write_slurm script as well) (not just the slab number)
    # IO — convert fli to omehans
    input_dir_fli: str = "/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Slab16/2026_03_12_HOLiS_NPBB328_Cortex_Slab16/"        #"/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Slab01_rerun/2025_09_23_HOLiS_NPBB328_Cortex_Slab01/"               #"/bil/proj/rf1hillman/2025_09_04_HOLiS_NPBB328_Cortex_Slab7_test_tissueXYZ/"                            
    output_dir_ome: str = "/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab16/"                                            #"/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab01/"                                                  #"/bil/proj/rf1hillman/results_peter/results_Slab7_test/"  should be the same as input for processing
    file_pattern_fli: str = '*NPBB328-Cortex-Slab16-run*-z*-y*-Exc-488nm-561nm-594nm-660nm_HiCAM FLUO_1875-ST-032.fli.zst' #'*NPBB328-Cortex-Slab01-run*-z*-y*-Exc-488nm-561nm-594nm-660nm_HiCAM FLUO_1875-ST-272.fli.zst'       #'NPBB328-Cortex-Slab07-ROI1-run*-z*-y*-Exc-488nm-561nm-594nm-660nm_HiCAM FLUO_1875-ST-272.fli' 
    
    # IO — background masks
    input_dir_bg: str = "/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Slab16/2026_03_12_HOLiS_NPBB328_Cortex_Slab16/" #"/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Slab01_rerun/2025_09_23_HOLiS_NPBB328_Cortex_Slab01/"              #"/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Slab6/2025_08_22_HOLiS_NPBB328_Cortex_Slab06/"  # where .fli files are, should be in the same folder as the raw files
    output_dir_bg: str = "/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab16/correction_files/"                                 #"/bil/proj/rf1hillman/results_peter/results_Slab7_test/correction_files/"
    file_pattern_bg: str = "NPBB328-Cortex-Slab16-run*-z*-y000-darkFrames_HiCAM FLUO_1875-ST-032.fli.zst"         #"NPBB328-Cortex-Slab01-run*-z*-y000-darkFrames_HiCAM FLUO_1875-ST-272.fli.zst"                      #"NPBB328-Cortex-Slab07-run*-z*-y000-darkFrames_HiCAM FLUO_1875-ST-272.fli.zst"  

    # IO — laser patterns
    input_dir_lasers: str = "/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Slab01_rerun/"                                                  #"/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Slab6/" # directory in which subdirectories for different zs are located
    output_dir_lasers: str = "/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab01/correction_files/laser_patterns/"               #"/bil/proj/rf1hillman/results_peter/results_Slab7_test/correction_files/laser_patterns/"
    subfolder_pattern_lasers: str = "2025_09_23_HOLiS_NPBB328_Cortex_Slab01_corrections*"                                        #"2025_09_02_HOLiS_NPBB328_Cortex_Slab07_corrections*"
    file_pattern_lasers: str = "epoxy-mix-run*HiCAM FLUO_1875-ST-272.fli.zst"                                                   #"epoxy-mix-run*HiCAM FLUO_1875-ST-272.fli.zst" 

    # IO — processing
    input_dir_process: str = "/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab01/"                                          #"/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/" "/bil/proj/rf1hillman/results_peter/results_Slab7_test/" 
    output_dir_process: str = "/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab01/out_processed_z05/"                    #"/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/out_processed_bg_dec1/" "/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_processed/"
    file_pattern_process = "*-272.fli*"

    # IO — segmentation
    input_dir_segment: str = "/bil/proj/rf1hillman/results_peter/results_Slab7_test/out_processed/"                             #*"/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/out_processed_bg_dec1/"
    output_dir_segment: str = "/bil/proj/rf1hillman/results_peter/results_Slab7_test/segmentation/"                             #"/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/out_processed_bg_dec1/segmentation/"
    file_pattern_segment = "*-272.fli_laser_corrected"

    # SLURM
    partition: str = "compute"
    mem_gb: int = 1024
    cpus: int = 16
    python_path: str = "/bil/users/psimko/.conda/envs/stack_to_multiscale_ngff/bin/python"
    convert_script_path: str = "/bil/users/psimko/holis/holis_pipeline/convert_scan.py"  
    get_bg_masks_script_path: str = "/bil/users/psimko/holis/holis_pipeline/get_bg_masks.py"
    get_laser_pattern_script_path: str = "/bil/users/psimko/holis/holis_pipeline/get_laser_patterns.py"
    process_script_path: str = "/bil/users/psimko/holis/holis_pipeline/process_scan.py"
    segment_script_path: str = "/bil/users/psimko/holis/holis_pipeline/segment_scan.py"

CFG = Config()
# ----------------------------------------

slurm_template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --mem={mem_gb}G
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH -o {stdout_log}
#SBATCH -e {stderr_log}
set -euo pipefail

"{python_path}" "{script_path}" "{input_file}" "{output_dir}" "{save_flags_json}"
"""
# ----------------------------------------

def write_slurm_script(cfg: Config, input_dir: str, output_dir: str, file_pattern: str, script_path: str, job_prefix: str, subfolder_pattern_lasers: str = None):
    os.makedirs(output_dir, exist_ok=True)
    if job_prefix == 'laser':
        file_glob = glob(os.path.join(input_dir, subfolder_pattern_lasers, file_pattern)) 
    elif job_prefix == 'proc' or job_prefix == 'segment':
        file_glob = glob(os.path.join(input_dir, file_pattern))  
        print(file_glob)
    else:
        file_glob = glob(os.path.join(input_dir, "**", file_pattern), recursive=True) 
    print(">>>>>>>>>>> Files:", len(file_glob))
    for filename in file_glob:
        allowed_z = {"05"}
        #allowed_y = {"001", "002", "003", "004", "005", "006", "007", "008", "009", "010"}
        base = os.path.basename(filename)                                   # e.g. NPBB328-...-272.fli.zst
        #root = os.path.splitext(os.path.splitext(base)[0])[0]               # strip .zst then .fli 
        if base.endswith(".fli.zst"):               # take care of both .fli and .zst.fli extensions
            root = base[:-len(".fli.zst")]
        elif base.endswith(".fli"):
            root = base[:-len(".fli")]
        else:
            root = os.path.splitext(base)[0]
        print(root)

        z = infer_z(root)   
        if z not in allowed_z:
            continue

        root = re.sub(r'\s+', '_', root)          # replace spaces
        root = re.sub(r'[^A-Za-z0-9._-]+', '_', root)  # extra safety
        job_name = f"{job_prefix}_{root}"

        # Set up logs
        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        stdout_log = os.path.join(logs_dir, f"{root}.out")
        stderr_log = os.path.join(logs_dir, f"{root}.err")

        config_dict = {
            "save_bg_subtracted": cfg.save_bg_subtracted,
            "save_laser_corrected": cfg.save_laser_corrected,
            "save_transformed": cfg.save_transformed,
        }

        # Encode as base64 to avoid bash escaping issues
        config_json = json.dumps(config_dict)
        config_b64 = base64.b64encode(config_json.encode()).decode()

        job_script_content = slurm_template.format(
            job_name=job_name,
            mem_gb=cfg.mem_gb,
            cpus=cfg.cpus,
            partition=cfg.partition,
            python_path=cfg.python_path,
            script_path=script_path, 
            input_file=filename, 
            output_dir=output_dir,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            save_flags_json=config_b64
        ) 
        
        job_script_path = os.path.join(output_dir, f"job_{root}.sh")
        print(f'job_script_path: {job_script_path}')
        with open(job_script_path, "w") as f:
            f.write(job_script_content)

        subprocess.run(["sbatch", job_script_path])
        print(f"Submitting {job_name} -> {filename}")


# ----------------------------------------
def main(cfg: Config):
    # Write SLURM script to convert flis to omehans
    if cfg.convert_to_ome:
        write_slurm_script(cfg, cfg.input_dir_fli, cfg.output_dir_ome, cfg.file_pattern_fli, cfg.convert_script_path, "convert")
    else:
        print("convert_to_ome=False — skipping conversion to ome")
    
    # Write SLURM script to extract BG masks
    if cfg.get_bg_masks:
        write_slurm_script(cfg, cfg.input_dir_bg, cfg.output_dir_bg, cfg.file_pattern_bg, cfg.get_bg_masks_script_path, "bg")
    else:
        print("get_bg_masks=False — skipping bg mask extraction")
    
    # Write SLURM script to extract LASER PATTERN masks
    if cfg.get_patterns:
        write_slurm_script(cfg, cfg.input_dir_lasers, cfg.output_dir_lasers, cfg.file_pattern_lasers, cfg.get_laser_pattern_script_path,"laser", cfg.subfolder_pattern_lasers)
        # No parallelization necessary here, just working with 2D masks. Returns 2 dictionaries {laser: pattern [Z,Y]}
        #normalize_patterns(cfg, cfg.output_dir_lasers, cfg.output_dir_lasers)         # Input and output are the same here
        #mix_patterns(cfg, cfg.output_dir_lasers, cfg.output_dir_lasers)
    else:
        print("get_patterns=False — skipping laser pattern extraction")

    if cfg.norm_and_mix_patterns:
        # No parallelization necessary here, just working with 2D masks. Returns 2 dictionaries {laser: pattern [Z,Y]}
        normalize_patterns(cfg, cfg.output_dir_lasers, cfg.output_dir_lasers)         # Input and output are the same here
        mix_patterns(cfg, cfg.output_dir_lasers, cfg.output_dir_lasers)
    else:
        print("norm_and_mix_patterns=False — skipping laser pattern normalization and mixing")

    # Write SLURM script to PROCESS
    if cfg.process:
        write_slurm_script(cfg, cfg.input_dir_process, cfg.output_dir_process, cfg.file_pattern_process, cfg.process_script_path, "proc")
    else:
        print("process=False — skipping processing")

    # Write SLURM script to SEGMENT
    if cfg.segment:
        write_slurm_script(cfg, cfg.input_dir_segment, cfg.output_dir_segment, cfg.file_pattern_segment, cfg.segment_script_path, "segment")
    else:
        print("segment=False — skipping segmentation")

# ----------------------------------------
if __name__ == "__main__":
    main(CFG)


""" # Configuration
# For correction masks
input_dir = "/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Slab6/2025_09_02_HOLiS_NPBB328_Cortex_Slab06/"       #the datee changes too!
output_dir = '/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab6/correction_files/'

# For processing
#input_dir =  "/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab7/" # use for corrections "/bil/proj/rf1hillman/HOLiS_NPBB328_Cortex/Slab6/"
#output_dir = '/bil/proj/rf1hillman/results/NPBB328_Cortex/Slab7/out_processed/'
#os.makedirs(output_dir, exist_ok=True)

slurm_template =#!/bin/bash
#SBATCH --job-name=holis_job
#SBATCH --mem=1024G
#SBATCH --cpus-per-task=16
#SBATCH --partition=compute

/bil/users/psimko/.conda/envs/stack_to_multiscale_ngff/bin/python /bil/users/psimko/holis/holis_pipeline/process_correction_omehans.py "{input_file}" "{output_dir}"
""" """

# #SBATCH --partition=compute or --partition=gpu
# When running this to get correction masks use
#/bil/users/psimko/.conda/envs/stack_to_multiscale_ngff/bin/python /bil/users/psimko/holis/holis_pipeline/process_correction_omehans.py "{input_file}" "{output_dir}"

# When running this to process files use
#/bil/users/psimko/.conda/envs/stack_to_multiscale_ngff/bin/python /bil/users/psimko/holis/holis_pipeline/process_scan.py "{input_file}" "{output_dir}"

# Main script
file_glob = glob(os.path.join(input_dir, f'*-272.fli.zst'))          # Use for darkframe file pattern f'NPBB328-Cortex-Slab06-run*-z*-y000-darkFrames_HiCAM FLUO_1875-ST-272.fli.zst'
print(">>>>>>>>>>> Files:", len(file_glob))
for filename in file_glob:
    #input_file = os.path.join(input_dir, filename)
    input_file = filename  
    base = os.path.basename(filename)                                   # e.g. NPBB328-...-272.fli.zst
    root = os.path.splitext(os.path.splitext(base)[0])[0]               # strip .zst then .fli
    print(f'Input file: {input_file}')
    output_log = os.path.join(output_dir, f"{root}.out")
    print(f'Output log: {output_log}')
    job_script_content = slurm_template.format(input_file=input_file, output_dir=output_dir) #,output_log=output_log)
    job_script_path = os.path.join(output_dir, "job_" + f'{root}.sh')
    print(f'job_script_path: {job_script_path}')
    with open(job_script_path, "w") as f:
        f.write(job_script_content)

    subprocess.run(["sbatch", job_script_path])
    print(f"Submitted job for {filename}")


#SBATCH --output={output_log} """
