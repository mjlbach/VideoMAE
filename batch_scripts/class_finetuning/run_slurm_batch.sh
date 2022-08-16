#!/bin/bash           

#SBATCH --job-name=mvae_ego4d
#SBATCH --partition=svl,viscam
#SBATCH --time=128:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:3090:4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=m.j.lbach@gmail.com

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

OUTPUT_DIR='/svl/u/mjlbach/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800/eval_lr_1e-3_epoch_100'
DATA_PATH='/vision/group/ego4d/v1/full_scale'
MODEL_PATH='/svl/u/mjlbach/checkpoint.pth'

source ~/.bashrc
conda activate mvae

python -u run_class_finetuning.py \
  --model vit_base_patch16_224 \
  --data_set SSV2 \
  --nb_classes 400 \
  --data_path ${DATA_PATH} \
  --finetune ${MODEL_PATH} \
  --log_dir ${OUTPUT_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --batch_size 8 \
  --num_sample 1 \
  --input_size 224 \
  --short_side_size 224 \
  --save_ckpt_freq 10 \
  --num_frames 16 \
  --sampling_rate 4 \
  --opt adamw \
  --lr 1e-3 \
  --opt_betas 0.9 0.999 \
  --weight_decay 0.05 \
  --epochs 100 \
  --dist_eval \
  --test_num_segment 5 \
  --test_num_crop 3 \
  --enable_deepspeed
