#!/bin/bash

# Server Constants
MAX_NUM_GPUS=8

# Defaults 
OUTPUT_DIR='/raid/data/bert/output'
DATA_DIR='/raid/data/bert/preproc_data/original'
NUM_GPUS=8
AMT_MEMORY=-1

# Getting each named parameter
while getopts g:o:d:m:h flag
do
	case "${flag}" in
		g) NUM_GPUS=${OPTARG};;
		o) OUTPUT_DIR=${OPTARG};;
		d) DATA_DIR=${OPTARG};;
		m) AMT_MEMORY=${OPTARG};;
		h) echo "-g: number of gpus"; echo "-o: checkpoint output dir"; echo "-d: TFRecord directory"; echo "-m: memory limit in GB"; echo "-h: this page"; exit 1;
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
		echo "You must provide a valid output directory"
		exit -3
fi

if [ ! -d $DATA_DIR ]
then
		echo "You must provide a valid input directory"
		exit -4
fi

# Building up the command
COMMAND="docker run -it --rm --gpus all -v /raid/data/bert/wiki:/wiki -v /raid/data/bert/raw-data:/raw-data -v ${DATA_DIR}:/data -v ${OUTPUT_DIR}:/output" 


if [ $AMT_MEMORY -gt 0 ]
then
		COMMAND="${COMMAND} -m ${AMT_MEMORY}g"
fi

COMMAND="${COMMAND} tf-bert:latest /bin/bash train_model.sh ${NUM_GPUS}"

exec $COMMAND
