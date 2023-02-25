#/bin/bash

# OUTPUT_DIR="/dl-bench/lhovon/mlcomns_bert/output"
OUTPUT_DIR="/raid/data/bert/run_output"
DATA_DIR='/raid/data/bert/preproc_data'

SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )
# OUTPUT_DIR="${SCRIPT_DIR}/output"

NUM_GPUS=${1:-8}
CONTAINER_NAME=${2:-train_bert}
LOGGING_DIR=${3:-"$SCRIPT_DIR/logging"}

if [ $# -eq 1 ]
then
		OUTPUT_DIR=$1
fi

docker run -it --gpus all \
	-v /raid/data/bert/wiki:/wiki \
	-v /raid/data/bert/raw-data:/raw-data \
	-v /raid/data/bert/preproc_data:/data \
	-v $SCRIPT_DIR/output:/output \
	-v $SCRIPT_DIR/logging:/logging \
	bert:loic /bin/bash
