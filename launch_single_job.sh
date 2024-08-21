#!/bin/bash
#SBATCH --job-name=single_job
#SBATCH -o output/single_job_%j.out
#SBATCH -e output/single_job_%j.err
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

python yolo_finetune_gen_data.py --seed 2025 --n_obs 400
