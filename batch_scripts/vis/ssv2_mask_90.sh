# Activate virtual environment
source ~/.bashrc  
conda activate mvae

cd ~/Repositories/VideoMAE

# Set the path to save video
OUTPUT_DIR='/viscam/u/mjlbach/video_memory_project/ssv2_90/vis'
# path to video for visualization
VIDEO_PATH='/viscam/data/SomethingSomethingV2/20bn-something-something-v2_sta_web_140w_320p/102800.mp4'
# path to pretrain model
MODEL_PATH='/viscam/u/mjlbach/video_memory_project/ssv2_90/checkpoint-739.pth'

python3 run_videomae_vis.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --frame_offset 0 \
    --num_frames 16 \
    --model pretrain_videomae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
