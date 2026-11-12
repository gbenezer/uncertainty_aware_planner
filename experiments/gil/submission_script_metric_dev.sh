#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=metric_dev
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --output=experiments/gil/job_logs/metric_dev_1.%j.out
#SBATCH --error=experiments/gil/job_logs/metric_dev_1.%j.err

source .venv/bin/activate
python experiments/gil/metric_development_experimentation.py