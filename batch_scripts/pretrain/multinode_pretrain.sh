#!/bin/bash           

#SBATCH --job-name=mvae_ssv2_multinode_90
#SBATCH --partition=svl
#SBATCH --time=128:00:00
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:4
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=m.j.lbach@gmail.com

#export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
export NUM_GPUS_PER_NODE=4

source ~/.bashrc  
conda activate mvae

OUTPUT_DIR='/viscam/u/mjlbach/video_memory_project/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800'
DATA_PATH='/viscam/data/SomethingSomethingV2/20bn-something-something-v2_sta_web_140w_320p/train.csv'

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

for ((i = 0; i < $SLURM_JOB_NUM_NODES; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 --export="all" -w "$node_i" python3 -m torch.distributed.launch --nproc_per_node=$NUM_GPUS_PER_NODE \
        --master_port 12320 --nnodes=$SLURM_NTASKS \
        --node_rank=$i --master_addr="$head_node_ip" \
        run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 20 \
        --epochs 801 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}
done
