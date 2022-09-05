#!/bin/bash
METHOD=$1
#TODO: Parametrize num_gpus and mount points
docker run -it --rm --gpus all -v /raid/data/bert/wiki:/wiki -v /raid/data/bert/augmentation:/augmentation -v /raid/data/bert/preproc_aug:/preproc_aug  tf-bert:latest /bin/bash preproc_data_parallel.sh ${METHOD}


