#!/bin/bash
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpusmall
#SBATCH --time=12:00:00
#SBATCH --account=project_2009199


if [[ -z "$SLURM_JOB_ID" ]]; then
  JOB_NAME="$1"
  shift
  sbatch --job-name="$JOB_NAME" -o "logs/${JOB_NAME}-%j.log" "$0" "$@"
  exit $?
else
  source venv/bin/activate
  python3 v3.py "$@"
fi