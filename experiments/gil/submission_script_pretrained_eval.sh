#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=pretrained_eval
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --output=experiments/gil/job_logs/pretrained_eval.%j.out
#SBATCH --error=experiments/gil/job_logs/pretrained_eval.%j.err

source .venv/bin/activate
python experiments/gil/pretrained_model_eval.py