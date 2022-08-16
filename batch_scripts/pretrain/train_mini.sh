#!/bin/bash           

#SBATCH --job-name=mvae_ego4d_mini
#SBATCH --partition=viscam,svl
#SBATCH --time=128:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=m.j.lbach@gmail.com

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='/svl/u/mjlbach/ego_vae_mini'
SAVE_DIR='/svl/u/mjlbach/ego_vae_mini_save'
MODEL_PATH='/svl/u/mjlbach/checkpoint.pth'
DATA_PATH='/svl/u/mjlbach/train_mini.csv'

source ~/.bashrc
conda activate mvae

python -u run_mae_pretraining.py \
  --data_path ${DATA_PATH} \
  --data_manifest /svl/u/mjlbach/train_mini.csv \
  --mask_type tube \
  --mask_ratio 0.9 \
  --model pretrain_videomae_base_patch16_224 \
  --decoder_depth 4 \
  --batch_size 8 \
  --num_frames 16 \
  --sampling_rate 4 \
  --opt adamw \
  --opt_betas 0.9 0.95 \
  --warmup_epochs 40 \
  --save_ckpt_freq 2 \
  --epochs 801 \
  --log_dir ${OUTPUT_DIR} \
  --output_dir ${OUTPUT_DIR}
