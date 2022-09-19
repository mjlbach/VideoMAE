# Activate virtual environment
source ~/.bashrc  
conda activate mvae

cd ~/Repositories/VideoMAE

# Set the path to save video
OUTPUT_DIR='/viscam/u/mjlbach/video_memory_project/ego4d_90/vis'
# path to video for visualization
VIDEO_PATH='/vision/group/ego4d/v1/clips/065715ac-d305-435a-a68f-e65e09cd5ae3.mp4 '
# path to pretrain model
MODEL_PATH='/viscam/u/mjlbach/video_memory_project/ego4d_90/checkpoint-1.pth'

python3 run_videomae_vis.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
