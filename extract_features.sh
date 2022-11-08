# path to video for visualization
DATASET_PATH='/home/michael/Repositories/video_memory_project/video-memory-retrieval/data/trajectories'
# path to pretrain model
MODEL_PATH=$(pwd)/checkpoint-2400.pth

python3 extract_videomae_features.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model pretrain_videomae_base_patch16_224 \
    --dataset_path $DATASET_PATH \
    --model_path $MODEL_PATH
