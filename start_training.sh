#!/bin/bash

DATA_DIR='/raid/data/bert/preproc_data'

OUTPUT_DIR="/raid/data/bert/run_output"

SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )
OUTPUT_DIR="${SCRIPT_DIR}/output"

NUM_GPUS=${1:-8}
CONTAINER_NAME=${2:-train_bert}
LOGGING_DIR=${3:-"$SCRIPT_DIR/output"}
DOCKER_IMAGE=${4:-"bert:loic"}
BATCH_SIZE=${5:-6}
NUM_STEPS=${6:-2400}
SAVE_CKPT_STEPS=${7:-2400}

docker run -it --rm --name=$CONTAINER_NAME --gpus all \
	-v /raid/data/bert/wiki:/wiki \
	-v /raid/data/bert/raw-data:/raw-data \
	-v ${DATA_DIR}:/data \
	-v ${OUTPUT_DIR}:/output \
	-v ${LOGGING_DIR}:/logging \
	$DOCKER_IMAGE /bin/bash ./train_model.sh $NUM_GPUS $BATCH_SIZE $NUM_STEPS $SAVE_CKPT_STEPS

