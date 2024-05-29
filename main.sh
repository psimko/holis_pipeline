# parse settings
# create all folders
# call chunking script
# call nuclei detection script
# call spectral extraction script
# call finalization script



# --------- parse settings in utils/settings.py --------

# Loop through lines in the file
while IFS= read -r line; do
  # Skip empty lines and comments
  if [[ -z "$line" || "$line" =~ ^# ]]; then
    continue
  fi

  # Extract key-value pair (assuming format KEY=VALUE)
  key=${line%%=*}
  key=$(echo $key | tr -d ' ')

  value=${line#*=}
  value=${value%%#*}
  value=$(echo $value | tr -d ' ')
  value=$(echo $value | tr -d '"')
  value=$(echo $value | tr -d "'")

  # Export the variable
#  echo "$key=$value"
  export "$key=$value"
done < "./utils/settings.py"

echo "The value of NUCLEI_DIR is: $NUCLEI_DIR"
echo "The value of OUTPUT_DIR is: $OUTPUT_DIR"


# --------- create all folders ---------

# output folder
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/scale_x
mkdir -p $OUTPUT_DIR/scale_x/mask_resized
mkdir -p $OUTPUT_DIR/scale_x/bright_spots_mask_resized
mkdir -p $OUTPUT_DIR/scale_$SCALE
mkdir -p $OUTPUT_DIR/scale_$SCALE/spectral_info
mkdir -p $OUTPUT_DIR/scale_$SCALE/slurm_jobs
mkdir -p $OUTPUT_DIR/scale_$SCALE/detection_masks
mkdir -p $OUTPUT_DIR/scale_$SCALE/detection
mkdir -p $OUTPUT_DIR/scale_$SCALE/dbscan


# --------- start pipeline ---------

module load anaconda3
conda activate $LNODE_ENV_NAME
python calculate_chunking.py

SCALE_DIR=$OUTPUT_DIR/scale_$SCALE
DETECTION_DIR=$OUTPUT_DIR/scale_$SCALE/detection
JOBS_DIR=$OUTPUT_DIR/scale_$SCALE/slurm_jobs
SPECTRAL_INFO_DIR=$OUTPUT_DIR/scale_$SCALE/spectral_info

FG_CHUNKS_FILE=$SCALE_DIR/foreground_chunks.txt
FG_CHUNKS_STR="$(cat $FG_CHUNKS_FILE)"
IFS=' ' read -r -a FG_CHUNKS <<< "$FG_CHUNKS_STR"
FG_CHUNKS_NUMBER=`echo ${FG_CHUNKS[@]} | wc -w`
echo "FG chunks number: $FG_CHUNKS_NUMBER"

for i in "${FG_CHUNKS[@]}"
do
  CHUNK_NUMBER=$i
  CHUNK_NUMBER_PADDED=$(printf "%05d" $CHUNK_NUMBER)
  DETECTION_FILE=$DETECTION_DIR/napari_chunk_$CHUNK_NUMBER_PADDED.csv
  if [ -f "$DETECTION_FILE" ]; then
    echo "skipping detection for chunk $CHUNK_NUMBER - output file already exists"
  else
    TASK_FILE=$JOBS_DIR/detect_chunk_$CHUNK_NUMBER_PADDED.sh
    # check whether job exists
    if [[ -f "$TASK_FILE" ]]; then
      echo "skipping detection for chunk $CHUNK_NUMBER - job already running"
    else
      echo "submitting GPU detection task for chunk $CHUNK_NUMBER"
      echo "#!/bin/bash" > $TASK_FILE
      echo "module load miniconda3" >> $TASK_FILE
      echo "source activate $GPU_ENV_NAME" >> $TASK_FILE
      echo "python predict_dynamicThreshold_fast.py $MODEL_PATH $CHUNK_NUMBER" >> $TASK_FILE
      #sbatch -p gpu --gres=gpu:1 --mem=64Gb -n8 $TASK_FILE
      sbatch -p GPU -N 1 --gpus=v100-32:16 -t 1:00:00 $TASK_FILE
    fi
  fi
done

# Empty array to store extracted numbers
finished_detection_jobs=()
sent_spectral_info_jobs=()

# Check what detection jobs finished (repeat until all are finished)
#while [ ${#finished_detection_jobs[@]} -lt ${#FG_CHUNKS[@]} ]
while true
do
  finished_detection_jobs=()

  # Loop through all .csv files in the output directory
  for filename in $DETECTION_DIR/*.csv; do
    # Check if it's a regular file (avoid hidden files, etc.)
    if [[ -f "$filename" ]]; then
      # Extract number from filename (assuming format "result_<number>.csv")
      number_str=${filename##*_}  # Remove everything before the last "_"
      number_str=${number_str%.csv}  # Remove ".csv" extension
      number=${number_str//[^0-9]/}  # Extract only digits
      number=$(echo $number | sed 's/^0*//')

      finished_detection_jobs+=($number)
      # Check that spectral info job hasn't been created for this chunk yet
      if [[ ! " ${sent_spectral_info_jobs[*]} " =~ [[:space:]]${number}[[:space:]] ]]; then
        # check whether output file exists
        CHUNK_NUMBER_PADDED=$(printf "%05d" $number)
        spectral_extraction_file=$SPECTRAL_INFO_DIR/spectral_chunk_$CHUNK_NUMBER_PADDED.scv
        if [ -f "$spectral_extraction_file" ]; then
          echo "Skipping spectral extraction for chunk $number - output already exists"
        else
          TASK_FILE=$JOBS_DIR/get_color_info_chunk_$CHUNK_NUMBER_PADDED.sh
          # check whether job exists
          if [[ -f "$TASK_FILE" ]]; then
            echo "Skipping spectral extraction for chunk $number - job already running"
          else
            echo "Submitting spectral extraction job for chunk $number"
            echo "#!/bin/bash" > $TASK_FILE
            echo "module load miniconda3" >> $TASK_FILE
            echo "source activate $LNODE_ENV_NAME" >> $TASK_FILE
            echo "python holis_get_chunk_spectral_info_slab_mouse.py $number $NUCLEI_DIR" >> $TASK_FILE
            #sbatch -p compute --mem=128Gb -n4 $TASK_FILE
            sbatch -p EM -t 1:00:00 --ntasks-per-node=96 $TASK_FILE
          fi
        fi
        sent_spectral_info_jobs+=($number)  # whether output file exists or not
      fi
    fi
  done
  finished_jobs_number=${#finished_detection_jobs[@]}
  sent_jobs_number=${#sent_spectral_info_jobs[@]}
  echo "Finished detection jobs: $finished_jobs_number"
  echo "Submitted spectral extraction jobs: $sent_jobs_number"

  if [ $sent_jobs_number -eq $FG_CHUNKS_NUMBER ]; then
    break
  fi
  sleep 2
done

finished_spectral_extraction_jobs=()

# Check what spectral extraction jobs finished (repeat until all are finished)
while true
do
  finished_spectral_extraction_jobs=()
  for filename in $SPECTRAL_INFO_DIR/*.csv; do
    # Check if it's a regular file (avoid hidden files, etc.)
    if [[ -f "$filename" ]]; then
      # Extract number from filename
      number_str=${filename##*_}  # Remove everything before the last "_"
      number_str=${number_str%.csv}  # Remove ".csv" extension
      number=${number_str//[^0-9]/}  # Extract only digits
      number=$(echo $number | sed 's/^0*//')

      finished_spectral_extraction_jobs+=($number)
    fi
  done
  finished_jobs_number=${#finished_spectral_extraction_jobs[@]}
  echo "Finished spectral extraction jobs: $finished_jobs_number"
  if [ $finished_jobs_number -eq $FG_CHUNKS_NUMBER ]; then
    break
  fi
  sleep 2
done

# Finalize: merge all csv and masks
python merge_csv_and_masks.py

# Check for SLURM errors
chmod a+x ./check_slurm_errors.sh
./check_slurm_errors.sh
