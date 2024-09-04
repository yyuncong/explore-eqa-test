#!/bin/bash
#SBATCH --job-name=eval_cluster
#SBATCH -o output/dynamic_openeqa_%j.out
#SBATCH -e output/dynamic_openeqa_%j.err
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --constraint="vram40"

module load miniconda/22.11.1-1
conda activate explore-eqa

# python online_evaluation_ours_new_1.py -cf cfg/online_eval_ours_new.yaml
# python online_evaluation_ours_3.py -cf cfg/online_eval_ours_2.yaml
# python online_evaluation_openeqa_3.py -cf cfg/online_eval_openeqa.yaml
# python online_evaluation_openeqa_new.py -cf cfg/online_eval_openeqa_new.yaml
#python run_openeqa_clustering.py -cf cfg/openeqa_clustering.yaml
#python online_evaluation_ours_new_1.py -cf cfg/online_eval_ours_new.yaml
python online_evaluation_openeqa_dynamic.py -cf cfg/online_eval_openeqa_dynamic.yaml