import os

from holis_pipeline import settings


def create_folders(output_folder):
    # create folders for chunks and for SLURM jobs
    output_folder_scale = os.path.join(output_folder, f"scale_{settings.SCALE}")
    if not os.path.exists(output_folder_scale):
        os.makedirs(output_folder_scale)
    detection_folder = os.path.join(output_folder_scale, 'detection')
    if not os.path.exists(detection_folder):
        os.makedirs(detection_folder)
    jobs_folder = os.path.join(output_folder_scale, 'slurm_jobs')
    if not os.path.exists(jobs_folder):
        os.makedirs(jobs_folder)
    spectral_info_folder = os.path.join(output_folder_scale, 'spectral_info')
    if not os.path.exists(spectral_info_folder):
        os.makedirs(spectral_info_folder)
