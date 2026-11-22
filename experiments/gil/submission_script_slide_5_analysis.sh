#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --job-name=slide_5
#SBATCH --mem=8GB
#SBATCH --ntasks=1
#SBATCH --output=experiments/gil/job_logs/slide_5.%j.out
#SBATCH --error=experiments/gil/job_logs/slide_5.%j.err

source .venv/bin/activate
python experiments/gil/slide_5_analysis.py