# Set the path to save video
OUTPUT_DIR='test_features'
# path to video for visualization
VIDEO_PATH='/home/michael/Repositories/igibson-dev/video_memory_project/trajectories/0ac68ba0-4e98-11ed-a406-2cf05da8a666'
# path to pretrain model
MODEL_PATH=$(pwd)/checkpoint-2400.pth

python3 extract_videomae_features.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}
