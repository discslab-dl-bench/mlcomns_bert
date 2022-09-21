#!/bin/bash

# Server Constants
MAX_NUM_GPUS=8
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

NUM_GPUS=${1:-8}
CONTAINER_NAME=${2:-train_bert}
shift 2

echo $SCRIPT_DIR

# Defaults 
OUTPUT_DIR="${SCRIPT_DIR}/output"
DATA_DIR='/raid/data/bert/preproc_data'
NUM_GPUS=8
BATCH_SIZE=24
AMT_MEMORY=-1
DELETE_PREVIOUS_RUNS=0
WARMUP_AMOUNT=10000

# Getting each named parameter
while getopts g:b:o:d:m:r:w:h flag
do
	case "${flag}" in
		g) NUM_GPUS=${OPTARG};;
		b) BATCH_SIZE=${OPTARG};;
		o) OUTPUT_DIR=${OPTARG};;
		d) DATA_DIR=${OPTARG};;
		m) AMT_MEMORY=${OPTARG};;
		r) DELETE_PREVIOUS_RUNS=1;;
		w) WARMUP_AMOUNT=${OPTARG};;
		h) echo "-g: number of gpus"; echo "-b: global batch size"; echo "-o: checkpoint output dir"; echo "-d: TFRecord directory"; echo "-m: memory limit in GB"; echo "-h: this page"; exit 1;
	esac
done

# Variable validation
if [ $NUM_GPUS -gt $MAX_NUM_GPUS ]
then
		echo "Sorry you cannot have more gpus then $MAX_NUM_GPUS"
		exit -2
fi

if [ ! -d $OUTPUT_DIR ]
then
		echo $OUTPUT_DIR
		echo "You must provide a valid output directory"
		exit -3
fi

if [ ! -d $DATA_DIR ]
then
		echo "You must provide a valid input directory"
		exit -4
fi

if [[ $DELETE_PREVIOUS_RUNS -eq 1 ]]
then
		rm -rf $OUTPUT_DIR/*
fi

# Building up the command
COMMAND="docker run -it --rm --gpus all -v /raid/data/bert/wiki:/wiki -v /raid/data/bert/raw-data:/raw-data -v ${DATA_DIR}:/data -v ${OUTPUT_DIR}:/output" 


if [ $AMT_MEMORY -gt 0 ]
then
		COMMAND="${COMMAND} -m ${AMT_MEMORY}g"
fi

COMMAND="${COMMAND} --name $CONTAINER_NAME tf-bert:test-loic /bin/bash train_model.sh ${NUM_GPUS} ${BATCH_SIZE}"

exec $COMMAND
