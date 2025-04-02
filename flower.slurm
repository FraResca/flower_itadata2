#!/bin/bash

#SBATCH --job-name=flower
#SBATCH --nodes=1
#SBATCH --error=flower.err
#SBATCH --output=flower.out
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1

module load miniconda3/
module load cuda/12.2

export PYTHONUNBUFFERED=1

conda run python3 finetune.py