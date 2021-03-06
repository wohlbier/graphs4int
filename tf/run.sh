#!/bin/bash

# export CEREBRAS_DATA_DIR=/srv/scratch/ogb/datasets/cb/nodeproppred
# export CEREBRAS_SIF_DIR=/srv/scratch/cerebras
# cd /path/to/modelzoo
# module load singularity-3.7.0-gcc-10.2.0-7qnn4oi

# need to build cython libraries if they don't exist. don't do it every time.
# NB: different singularity image!
# singularity shell --bind $CEREBRAS_DATA_DIR:/data,`pwd`:/work $CEREBRAS_SIF_DIR/centos.sif
# cd /work
# python3.7 graphsaint/setup.py build_ext --inplace
# make -C graphsaint links

# singularity shell --bind $CEREBRAS_DATA_DIR:/data,`pwd`:/work $CEREBRAS_SIF_DIR/cbcore-latest.sif
# cd /work/graphsaint/tf
# ./run.sh

#DATA=./data/ogbn_arxiv
#DATA=./data/ogbn_products
DATA=/data/ogbn_arxiv/GraphSAINT
#DATA=/data/ogbn_products

rm -f ./checkpoint
rm -f ./tmp.chkpt.*

python \
    run.py \
    --data_prefix ${DATA} \
    --train_config \
    ./train_config/open_graph_benchmark/ogbn-products_3_e_gat.yml \
    --gpu 0 \
    --cpu_eval \
    --eval_val_every 10 \
    --loss_dim_expand \
    --train_log_freq 100 \
    --num_global_train_steps 60 \
    --model_dir ./model_dir \
    --mode compile_only


#--cs_ip 1.1.1.1
