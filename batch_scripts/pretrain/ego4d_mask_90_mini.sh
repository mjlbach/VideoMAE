#!/bin/bash           

#SBATCH --job-name=mvae_ego_single_node_90_mini
#SBATCH --partition=macondo
#SBATCH --time=124:00:00
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=m.j.lbach@gmail.com

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

cd ~/Repositories/VideoMAE

source ~/.bashrc  
conda activate mvae

OUTPUT_DIR='/viscam/u/mjlbach/video_memory_project/ego4d_90_mini'
DATA_PATH='/svl/u/mjlbach/Repositories/VideoMAE/batch_scripts/pretrain/experiments_jc/train_mini.csv'

srun python run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 28 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}
wait
