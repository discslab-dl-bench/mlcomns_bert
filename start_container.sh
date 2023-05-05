#/bin/bash

SCRIPT_DIR=$( dirname -- "$( readlink -f -- "$0"; )" )

DOCKER_IMAGE=${1:-"bert:original"}

docker run -it --gpus all \
	-v /raid/data/bert/wiki:/wiki \
	-v /raid/data/bert/raw-data:/raw-data \
	-v /raid/data/bert/preproc_data:/data \
	-v $SCRIPT_DIR/output:/output \
	-v $SCRIPT_DIR/logging:/logging \
	$DOCKER_IMAGE /bin/bash
