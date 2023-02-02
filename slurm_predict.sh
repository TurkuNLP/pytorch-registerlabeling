#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH -p small
#SBATCH -t 10:15:00
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2005092
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

module purge
module load pytorch
#source venv/bin/activate
export PYTHONPATH=venv/lib/python3.8/site-packages:$PYTHONPATH
export TRANSFORMERS_CACHE=cachedir
#pip3 install --upgrade pip
python3 -m pip install --upgrade pip
pip3 install transformers
#pip3 install datasets

srun python3 predict.py --text $1 --load_model $2 | gzip > $1_preds.txt.gz
