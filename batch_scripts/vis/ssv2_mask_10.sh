#!/bin/bash

# Activate virtual environment
source ~/.bashrc  
conda activate mvae

cd ~/Repositories/VideoMAE

# NOTE : Quote it else use array to avoid problems #
FILES="/path/to/*"
for f in $FILES
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name
  cat "$f"
done

# Set the path to save video
OUTPUT_DIR='/viscam/u/mjlbach/video_memory_project/ssv2_10/vis'
# path to video for visualization
VIDEO_PATH='/viscam/data/SomethingSomethingV2/20bn-something-something-v2_sta_web_140w_320p/10279.mp4'
# path to pretrain model
MODEL_PATH='/viscam/u/mjlbach/video_memory_project/ssv2_10/checkpoint-239.pth'

python3 run_videomae_vis.py \
    --mask_ratio 0.1 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
