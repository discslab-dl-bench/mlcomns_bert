#/bin/bash

# OUTPUT_DIR="/dl-bench/lhovon/mlcomns_bert/output"
OUTPUT_DIR="/raid/data/bert/run_output"
DATA_DIR='/raid/data/bert/preproc_data'

if [ $# -eq 1 ]
then
		OUTPUT_DIR=$1
fi

docker run -it --gpus all -v /raid/data/bert/wiki:/wiki -v /raid/data/bert/raw-data:/raw-data -v /raid/data/bert/preproc_data:/data -v $OUTPUT_DIR:/output bert:loic
