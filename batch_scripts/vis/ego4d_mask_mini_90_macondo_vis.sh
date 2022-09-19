# Activate virtual environment
source ~/.bashrc  
conda activate mvae

cd ~/Repositories/VideoMAE

VIDEO='257f685e-f4a9-46b1-819c-ee4499600436.mp4'
# Set the path to save video
OUTPUT_DIR="/viscam/u/mjlbach/video_memory_project/ego4d_90_mini_macondo/$VIDEO"
# path to video for visualization
VIDEO_PATH="/vision/group/ego4d/v1/clips/$VIDEO"
# path to pretrain model
MODEL_PATH='/viscam/u/mjlbach/video_memory_project/ego4d_90_mini_macondo/checkpoint-479.pth'

python3 run_videomae_vis.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
