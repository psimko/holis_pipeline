def submit_slurm_task_compute(task_path):
    command = ['sbatch', '-p', 'compute', '--mem=128Gb', '-n4', task_path]
    subprocess.run(command)


def write_spectral_extraction_script_for_slurm(chunk_number, task_path):
    with open(task_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('module load miniconda3\n')
        f.write(f'source activate {LNODE_ENV_NAME}')
        f.write('\n')
        f.write(f'python {work_dir}/holis_get_chunk_spectral_info_slab_mouse.py ')
        f.write(str(chunk_number))
        f.write(' ')
        f.write(NUCLEI_DIR)
        f.write('\n')


def get_spectral_info_slurm(chunk_numbers, jobs_folder, spectral_info_folder):
    # exclude already saved chunks
    # write bash scripts
    # submit to compute partition
    for chunk_number in chunk_numbers:
        if os.path.exists(os.path.join(spectral_info_folder, f"spectral_chunk_{str(chunk_number).zfill(5)}.csv")):
            print(f"Skipping chunk {chunk_number}")
            continue
        task_path = os.path.join(jobs_folder, f"get_color_info_chunk_{str(chunk_number).zfill(5)}.sh")
        write_spectral_extraction_script_for_slurm(chunk_number, task_path)
        submit_slurm_task_compute(task_path)


def get_spectral_info(coords_file, colors_dir):
    pass
    # ================= Create spectral extraction jobs =================

    # check which csv files have been generated
    # send these chunks for spectral information
    # spectral_info_folder = os.path.join(output_folder_scale, 'spectral_info')
    # sent_tasks = set()
    # remaining_chunks = fg_chunks.copy()
    # while len(remaining_chunks):
    #     print("Chunks remaining to do nuclei detection", len(remaining_chunks))
    #     detection_done = set([
    #         int(re.findall(r"\d+", os.path.basename(x))[-1]) for x in glob(os.path.join(detection_folder, "*.csv"))
    #     ])
    #     chunk_numbers_set = detection_done - sent_tasks
    #     # actual submission happens here:
    #     get_spectral_info_slurm(list(chunk_numbers_set), jobs_folder, spectral_info_folder)
    #     sent_tasks.update(chunk_numbers_set)
    #     remaining_chunks = fg_chunks - sent_tasks
    #     time.sleep(2)
