#!/bin/bash           

#SBATCH --job-name=mvae_ssv2_single_node_10
#SBATCH --partition=viscam
#SBATCH --time=128:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a5000:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=m.j.lbach@gmail.com

source ~/.bashrc  
conda activate mvae

cd ~/Repositories/VideoMAE

OUTPUT_DIR='/viscam/u/mjlbach/video_memory_project/ssv2_10'
DATA_PATH='/viscam/data/SomethingSomethingV2/20bn-something-something-v2_sta_web_140w_320p/train.csv'

head_node_ip=$(hostname --ip-address)

python3 -m torch.distributed.launch --nproc_per_node=1 \
        --master_port 12320 --nnodes=1 \
        --node_rank=0 --master_addr="$head_node_ip" \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.1 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 28 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}
