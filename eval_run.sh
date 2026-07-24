#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=1:00:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --job-name=eval_run
#SBATCH --output=eval_run-%j.log
# start environments and load existing modules
module purge
module load Anaconda3/2024.02-1
module load CUDA/12.8.0

# create conda environment 
conda activate V3VSR
# pip install -r requirements.txt
export JAX_COORDINATOR_ADDRESS=127.0.0.1:1234
export JAX_PROCESS_ID=0
export JAX_NUM_PROCESSES=2  # or your actual process count
export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
export WANDB_API_KEY="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"

# run 
# python inference_tile.py -i /scratch/s3591077/mthesis/datasets/volume_static_short -o volume_static_short.mp4 --outputimage_path ./pngdump

python run_inference.py --data-dir /home2/s3591077/scratch/datasets/scisr --checkpoint-path /home2/s3591077/scratch/models/logs/params_latest-v3-pre.pkl --eval-sets test/LQ --space-scale 4 --time-scale 2
    
