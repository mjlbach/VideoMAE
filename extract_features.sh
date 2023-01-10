#!/bin/bash           
#
#SBATCH --job-name=procthor_extract_vmae_features
#SBATCH --partition=svl
#SBATCH --account=vision
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=m.j.lbach@gmail.com
#SBATCH --array=0-29

# DATASET_PATH='/svl/data/procthor_search_dataset/v1.3/train/'
MANIFEST_PATH='/svl/u/mjlbach/Repositories/video_memory_project/video-memory-retrieval/data/manifest.csv'

# path to pretrain model
MODEL_PATH='/svl/u/mjlbach/Repositories/video_memory_project/VideoMAE/checkpoint-2400.pth'
export PATH=/svl/u/mjlbach/mambaforge/envs/mvae/bin:$PATH

python3 extract_videomae_features.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    --manifest_path $MANIFEST_PATH \
    --group $SLURM_ARRAY_TASK_ID \
    --model_path $MODEL_PATH
