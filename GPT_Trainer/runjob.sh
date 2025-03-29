#!/bin/bash

#SBATCH -A eclarson_protein_diffusion_0001
#SBATCH --job-name=Nya~GPT~^w^
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -o runjob.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=500G


# Specify node to run on
#SBATCH --nodelist=bcm-dgxa100-0002


# Number of nodes
nnodes=1
# Number of tasks per node
nproc_per_node=4



nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

cd /users/gmongaras/work/Cottention_Tests
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 srun /home/gmongaras/miniconda3/bin/torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
GPT_Trainer/train.py
