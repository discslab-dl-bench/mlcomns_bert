#!/bin/bash


OUTPUT_DIR="/raid/data/bert/run_output"
DATA_DIR='/raid/data/bert/short_old'

NUM_GPUS=${1:-8}
CONTAINER_NAME=${2:train_bert}
BATCH_SIZE=${3:-6}
DOCKER_IMAGE=${4:-"bert:horovod"}
NUM_STEPS=${5:-2400}
SAVE_CKPT_STEPS=${6:-2400}

docker run -it --rm --name=$CONTAINER_NAME --gpus all \
	-v /raid/data/bert/wiki:/wiki \
	-v /raid/data/bert/raw-data:/raw-data \
	-v ${DATA_DIR}:/data \
	-v ${OUTPUT_DIR}:/output \
	$DOCKER_IMAGE /bin/bash ./train_model.sh $NUM_GPUS $BATCH_SIZE $NUM_STEPS $SAVE_CKPT_STEPS

