#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=imputation_eval_long
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --output=experiments/gil/job_logs/imputation_eval_long.%j.out
#SBATCH --error=experiments/gil/job_logs/imputation_eval_long.%j.err

source .venv/bin/activate
python experiments/gil/imputation_model_evaluation_long.py