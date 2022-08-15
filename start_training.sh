#!/bin/bash

OUTPUTDIR='/raid/data/bert/output'

if [ -z $1 ]
then
	echo "Saving to default output directory"
else
	if [ -d $1 ] 
	then
		echo "Setting directory path to ${1}"
		OUTPUTDIR=$1
	else
		echo 'You must provide a valid directory path'
		exit 1
	fi
fi	


docker run -it --rm --gpus all -v /raid/data/bert/wiki:/wiki -v /raid/data/bert/raw-data:/raw-data -v /raid/data/bert/preproc_data:/data -v $OUTPUTDIR:/output tf-bert:latest /bin/bash train_model.sh
