#!/bin/bash

#TODO: Parametrize num_gpus and mount points
docker run -it --rm --gpus all -v /raid/data/bert/wiki:/wiki -v /raid/data/bert/raw-data:/raw-data -v /raid/data/bert/preproc_data:/data -v /raid/data/bert/output:/output tf-bert:latest /bin/bash train_model.sh
