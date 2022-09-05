#/bin/bash

# docker run -it --gpus all -v /data/bert/wiki:/wiki -v /data/bert/raw-data:/raw-data -v /raid/data/bert/preproc_data:/data -v /raid/data/bert/output:/output tf-bert:latest
docker run -it --gpus all -v /data/bert/wiki:/wiki -v /data/bert/raw-data:/raw-data -v /dl-bench/ychen/mlcommons_bert/augmentation:/data -v /raid/data/bert/output:/output tf-bert:latest
