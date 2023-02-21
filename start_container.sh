#/bin/bash

OUTPUT_DIR="/raid/data/bert/run_output"
DATA_DIR='/raid/data/bert/preproc_data'

docker run -it --gpus all \
    -v /data/bert/wiki:/wiki \
    -v /data/bert/raw-data:/raw-data \
    -v /raid/data/bert/preproc_data:/data \
    -v /raid/data/bert/output:/output bert:original 
