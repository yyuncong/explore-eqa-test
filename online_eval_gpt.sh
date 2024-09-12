#!/bin/bash
#SBATCH --job-name=eval_gpt
#SBATCH -o output/gpt_%j.out
#SBATCH -e output/gpt_%j.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint="vram12"


module load miniconda/22.11.1-1
conda activate explore-eqa


#python gpt_evaluation_dynamic_continuous.py -cf cfg/gpt_eval_continuous.yaml
#python gpt_evaluation_dynamic.py -cf cfg/gpt_eval_dynamic.yaml
#python gpt_evaluation_dynamic_continuous.py -cf cfg/gpt_eval_dynamic.yaml
python gpt_evaluation_goatbench.py -cf cfg/gpt_eval_goatbench.yaml --start_ratio 0.1 --end_ratio 0.3