#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=runs/dove_eval-%j.log
#SBATCH --job-name=dove_eval


module purge
module load Anaconda3/2024.02-1
module load CUDA/12.8.0

# create conda environment 
conda activate V3VSR


python eval_imgs.py --time-scale=1  /home2/s3591077/scratch/datasets/scisr-img/test/GT  /home2/s3591077/scratch/models/v3/dove_results
