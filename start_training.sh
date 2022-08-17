#!/bin/bash

# Server Constants
MAX_NUM_GPUS=8

# Defaults 
OUTPUT_DIR='/raid/data/bert/output'
DATA_DIR='/raid/data/bert/preproc_data'
NUM_GPUS=8

# Getting each named parameter
while getopts g:o:d:h flag
do
	case "${flag}" in
		g) NUM_GPUS=${OPTARG};;
		o) OUTPUT_DIR=${OPTARG};;
		d) DATA_DIR=${OPTARG};;
		h) echo "-g: number of gpus"; echo "-o: checkpoint output dir"; echo "-d: TFRecord directory"; echo "-h: this page"; exit 1;
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

echo $OUTPUT_DIR
echo $NUM_GPUS
echo $DATA_DIR


#docker run -it --rm --gpus all -v /raid/data/bert/wiki:/wiki -v /raid/data/bert/raw-data:/raw-data -v DATA_DIR:/data -v $OUTPUT_DIR:/output tf-bert:latest /bin/bash train_model.sh $NUM_GPUS
