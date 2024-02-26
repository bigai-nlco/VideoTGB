#!/bin/bash
#SBATCH --job-name=LSTP-Chat-flant5-xl
#SBATCH --partition=xxx
#SBATCH --account=xxx
#SBATCH --qos=xxx
#SBATCH --time=xxx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=./slurm_logs/train-LSTP-videoinstruct.out
#SBATCH --error=./slurm_logs/train-LSTP-videoinstruct.error.out

# Schedule execution of many runs
# unset LOCAL_RANK
# CUDA_LAUNCH_BLOCKING=1 \
# NCCL_DEBUG=INFO \
# NCCL_P2P_DISABLE=1 \
HYDRA_FULL_ERROR=1 \
srun python src/train.py experiment=LSTP_SF_blip2flant5xl_videoinstruct