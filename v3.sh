#!/bin/bash
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --account=project_2009199
#SBATCH --mail-user=pytorchregisterlabeling@gmail.com 
#SBATCH --mail-type=ALL

if [[ -z "$SLURM_JOB_ID" ]]; then
  PARTITION="gpusmall"
  if [[ $1 == "m" ]]; then
    NUM_GPUS=4
    PARTITION="gpumedium"
    shift
  elif [[ $1 == "2" ]]; then
    NUM_GPUS=2
    shift 
  else
    NUM_GPUS=1 
  fi

  # Set the dynamic GPU requirement
  GRES_GPU="gpu:a100:$NUM_GPUS"
  DYNAMIC_JOBNAME="$1"
  shift  # Shift command-line arguments to remove the first one
  # Submit the job to slurm with the dynamic job name and capture the output
  JOB_SUBMISSION_OUTPUT=$(sbatch --job-name="$DYNAMIC_JOBNAME" --gres="$GRES_GPU" --partition="$PARTITION" -o "logs/${DYNAMIC_JOBNAME}-%j.log" "$0" "$@")
  echo "Submission output: $JOB_SUBMISSION_OUTPUT"
  # Extract the job ID from the submission output
  JOB_ID=$(echo "$JOB_SUBMISSION_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
  LOG_FILE="logs/${DYNAMIC_JOBNAME}-${JOB_ID}.log"
  touch $LOG_FILE
  echo "Tailing log file: $LOG_FILE"
  # Use tail -f to follow the log file
  tail -f "$LOG_FILE"
  exit $?
else
  # Actual job script starts here
  source venv/bin/activate
  srun python3 "$@"
fi