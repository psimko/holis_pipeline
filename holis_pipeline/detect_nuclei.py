import os
import subprocess
from pathlib import Path

from holis_pipeline.settings import *


def write_detection_task_for_slurm(chunk_number, output_path):
    work_dir = str(Path(os.path.dirname(__file__)).parent)
    with open(output_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('module load miniconda3\n')
        f.write(f'source activate {GPU_ENV_NAME}')
        f.write('\n')
        f.write(f'python {work_dir}/predict_dynamicThreshold_fast.py ')  # TODO
        f.write(MODEL_PATH)
        f.write(' ')
        f.write(str(chunk_number))
        f.write('\n')


def submit_slurm_task_gpu(path_to_task):
    command = ['sbatch', '-p', 'gpu', '--gres=gpu:1', '--mem=64Gb', '-n8', path_to_task]
    subprocess.run(command)


def detect_cells_deepblink_slurm(chunk_numbers, jobs_folder):
    for chunk_number in chunk_numbers:
        detection_folder = os.path.join(OUTPUT_DIR, f'scale_{SCALE}', 'detection')
        if not os.path.exists(detection_folder):
            os.makedirs(detection_folder)
        detections_file_name = os.path.join(detection_folder, f"napari_chunk_{str(chunk_number).zfill(5)}.csv")
        if os.path.exists(detections_file_name):
            print(f"Skipping chunk {chunk_number}")
            continue
        print(f"Submitting gpu task for chunk {chunk_number}")
        task_path = os.path.join(jobs_folder, f"detect_chunk_{str(chunk_number).zfill(5)}.sh")
        write_detection_task_for_slurm(chunk_number, task_path)
        submit_slurm_task_gpu(task_path)
