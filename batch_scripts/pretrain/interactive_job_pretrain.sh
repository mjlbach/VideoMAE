source ~/.bashrc  
conda activate mvae

cd ~/Repositories/VideoMAE

OUTPUT_DIR='/viscam/u/mjlbach/video_memory_project/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800'
DATA_PATH='/viscam/data/SomethingSomethingV2/20bn-something-something-v2_sta_web_140w_320p/train.csv'

head_node_ip=$(hostname --ip-address)

python3 -m torch.distributed.launch --nproc_per_node=2 \
        --master_port 12320 --nnodes=1 \
        --node_rank=0 --master_addr="$head_node_ip" \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
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
