#!/bin/bash
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gputest
#SBATCH --time=0:15:00
#SBATCH --account=project_2009199
#SBATCH --mail-user=pytorchregisterlabeling@gmail.com 
#SBATCH --mail-type=ALL

if [[ -z "$SLURM_JOB_ID" ]]; then
  DYNAMIC_JOBNAME="$1"
  shift  # Shift command-line arguments to remove the first one
  # Submit the job to slurm with the dynamic job name and capture the output
  JOB_SUBMISSION_OUTPUT=$(sbatch --job-name="$DYNAMIC_JOBNAME" -o "logs/${DYNAMIC_JOBNAME}-%j.log" "$0" "$@")
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
  # Reconstruct log file path
  LOG_FILE="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log"

  # Ensure you replace "my@email.com" with the actual recipient's email address
  /usr/bin/mail -s "Job $SLURM_JOB_NAME Ended id=$SLURM_JOB_ID" pytorchregisterlabeling@gmail.com < "$LOG_FILE"
fi