#!/bin/bash           

#SBATCH --job-name=test-nodes        # name
#SBATCH --partition=viscam           # partition
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=5            # number of cores per tasks
#SBATCH --gres=gpu:2                 # number of gpus
#SBATCH --mem-per-cpu=4G             # memory
#SBATCH --time 0:05:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name

cd ~/Repositories/VideoMAE
source ~/.bashrc
conda activate mvae 

export OMP_NUM_THREADS=1
export GPUS_PER_NODE=4
export NCCL_P2P_LEVEL=NVL
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
torch-distributed-gpu-test.py'
