# Set the path to save video
OUTPUT_DIR='VideoMAE/demo/test'
# path to video for visualization
VIDEO_PATH='/vision/group/ego4d/v1/full_scale/000786a7-3f9d-4fe6-bfb3-045b368f7d44.mp4'
# path to pretrain model
MODEL_PATH='/svl/u/mjlbach/checkpoint.pth'

python3 run_videomae_vis.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --sampling_rate 4 \
    --model pretrain_videomae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
