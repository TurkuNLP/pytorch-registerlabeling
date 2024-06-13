#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --account=project_2009056
#SBATCH --mail-user=pytorchregisterlabeling@gmail.com 
#SBATCH --mail-type=ALL

if [[ -z "$SLURM_JOB_ID" ]]; then
  PARTITION="gpusmall"
  TIME="12:00:00"
  NUM_GPUS=1
  MEM=32
  if [[ $1 == "4gpu" ]]; then
    NUM_GPUS=4
    PARTITION="gpumedium"
    shift
  elif [[ $1 == "2gpu" ]]; then
    NUM_GPUS=2
    shift 
  elif [[ $1 == "64G" ]]; then
    MEM=64
    shift 
  elif [[ $1 == "24h" ]]; then
    TIME="24:00:00"
    shift 
   elif [[ $1 == "24h2gpu" ]]; then
    TIME="24:00:00"
    NUM_GPUS=2
    shift 
   elif [[ $1 == "36h2gpu" ]]; then
    TIME="36:00:00"
    NUM_GPUS=2
    shift 
  elif [[ $1 == "1h" ]]; then
    TIME="1:00:00"
    MEM=8
    shift 
  elif [[ $1 == "2h" ]]; then
    TIME="2:00:00"
    MEM=8
    shift 
  elif [[ $1 == "6h" ]]; then
    TIME="6:00:00"
    MEM=8
    shift 
  elif [[ $1 == "30min" ]]; then
    TIME="0:30:00"
    MEM=8
    shift 
  fi

  # Set the dynamic GPU requirement
  GRES_GPU="gpu:a100:$NUM_GPUS"
  DYNAMIC_JOBNAME="$1"
  shift  # Shift command-line arguments to remove the first one
  # Submit the job to slurm with the dynamic job name and capture the output
  JOB_SUBMISSION_OUTPUT=$(sbatch --job-name="$DYNAMIC_JOBNAME" --time="$TIME" --gres="$GRES_GPU" --mem="$MEM"G --partition="$PARTITION" -o "logs/${DYNAMIC_JOBNAME}-%j.log" "$0" "$@")
  echo "Submission output: $JOB_SUBMISSION_OUTPUT"
  # Extract the job ID from the submission output
  JOB_ID=$(echo "$JOB_SUBMISSION_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
  LOG_FILE="logs/${DYNAMIC_JOBNAME}-${JOB_ID}.log"
  touch $LOG_FILE
  echo "Log file: $LOG_FILE"
  # Use tail -f to follow the log file
  #tail -f "$LOG_FILE"
  exit $?
else
  # Actual job script starts here
  source venv/bin/activate
  srun python3 "$@"
fi
