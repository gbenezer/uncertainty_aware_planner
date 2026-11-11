#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=dlinear_quantile_training
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --output=experiments/gil/job_logs/dlinear_quantile_training.%j.out
#SBATCH --error=experiments/gil/job_logs/dlinear_quantile_training.%j.err

source .venv/bin/activate
python experiments/gil/dlinear_quantile_training.py