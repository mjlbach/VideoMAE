#!/bin/bash
# NOTE : Quote it else use array to avoid problems #
shopt -s nullglob
video_directory=/viscam/data/SomethingSomethingV2/20bn-something-something-v2_sta_web_140w_320p
output_path=/viscam/u/mjlbach

# Activate virtual environment
source ~/.bashrc  
conda activate mvae

cd ~/Repositories/VideoMAE

# path to pretrain model
MODEL_PATH='/viscam/u/mjlbach/video_memory_project/pretrain_checkpoint_ssv2_2400.pth'

arr=($video_directory/*)
for i in "${arr[@]:0:10}"
do
    VIDEO_PATH=$i
    filename=$(basename $i)
    directory="${filename%%.*}"

    # Set the path to save video
    OUTPUT_DIR="/viscam/u/mjlbach/video_memory_project/ssv2_pretrained_90/$directory"

    python3 run_videomae_vis.py \
        --mask_ratio 0.9 \
        --mask_type tube \
        --decoder_depth 4 \
        --frame_offset 0 \
        --num_frames 16 \
        --model pretrain_videomae_base_patch16_224 \
        ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}

done
