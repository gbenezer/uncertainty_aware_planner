#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=dlinear_test_script_1
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --output=experiments/gil/job_logs/dlinear_test_script_1.%j.out
#SBATCH --error=experiments/gil/job_logs/dlinear_test_script_1.%j.err

cd ondemand/dev/uncertainty_aware_planner
source .venv/bin/activate
python experiments/gil/data_pipeline_experimentation.py