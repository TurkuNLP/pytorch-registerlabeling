#!/bin/bash
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpumedium
#SBATCH --time=12:00:00
#SBATCH --account=project_2009199


if [[ -z "$SLURM_JOB_ID" ]]; then
  # Check if the first argument is a number (for GPUs), else default to 1
  if [[ $1 =~ ^[0-9]+$ ]]; then
    NUM_GPUS="$1"
    shift  # Remove the GPU count from the arguments
  else
    NUM_GPUS=1  # Default to 1 GPU
  fi

  # Set the dynamic GPU requirement
  GRES_GPU="gpu:a100:$NUM_GPUS"
  DYNAMIC_JOBNAME="$1"
  shift  # Shift command-line arguments to remove the first one
  # Submit the job to slurm with the dynamic job name and capture the output
  JOB_SUBMISSION_OUTPUT=$(sbatch --job-name="$DYNAMIC_JOBNAME" --gres="$GRES_GPU"  -o "logs/${DYNAMIC_JOBNAME}-%j.log" "$0" "$@")
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