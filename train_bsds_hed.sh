#!/usr/bin/bash

LEARNING_RATE=1e-6
WEIGHT_DECAY=2e-4
LABEL_GAMMA=0.3
LABEL_LAMBDA=1.1
BATCH_SIZE=4
NUM_EPOCHS=15
UPDATE_ITERS=8
OPTIMIZER="sgd"
EXPT_NAME="bsds_torch_lr_${LEARNING_RATE}_wd_${WEIGHT_DECAY}_gamma_${LABEL_GAMMA}_lambda_${LABEL_LAMBDA}_batch_${BATCH_SIZE}_opt_${OPTIMIZER}"
DATA_DIR="/mnt/cube/projects/bsds500/HED-BSDS/"
BASE_DIR="models/expt_checkpoints"

python main_bsds.py \
 --learning_rate=${LEARNING_RATE} \
 --weight_decay=${WEIGHT_DECAY} \
 --label_gamma=${LABEL_GAMMA} \
 --label_lambda=${LABEL_LAMBDA} \
 --batch_size=${BATCH_SIZE} \
 --num_epochs=${NUM_EPOCHS} \
 --update_iters=${UPDATE_ITERS} \
 --optimizer=${OPTIMIZER} \
 --expt_name=${EXPT_NAME} \
 --data_dir=${DATA_DIR} \
 --base_dir=${BASE_DIR}
