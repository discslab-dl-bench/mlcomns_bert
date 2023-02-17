#!/bin/bash

# To be run in the docker container
# Needs /data (containing tfrecords of data), /wiki and /output mounted

NUM_GPUS=${1:-8}
BATCH_SIZE=${2:-6}
NUM_STEPS_TRAIN=${3:-2400}
# ALways checkpoints once at the start as well
SAVE_CKPT_STEPS=${4:-2400}

GLOBAL_BATCH_SIZE=$(expr $BATCH_SIZE \* $NUM_GPUS)

DATA_DIR="/data"
WIKI_DIR="/wiki"
OUTPUT_DIR="/output"

TF_XLA_FLAGS='--tf_xla_auto_jit=2'

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

echo "Checking wiki dir: "
ls ${WIKI_DIR}/ckpt/

horovodrun -np $NUM_GPUS python run_pretraining.py \
  --bert_config_file=${WIKI_DIR}/bert_config.json \
  --output_dir=${OUTPUT_DIR} \
  --log_dir=${OUTPUT_DIR} \
  --input_file="${DATA_DIR}/part*" \
  --nodo_eval \
  --do_train \
  --learning_rate=0.0001 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_train_steps=$NUM_STEPS_TRAIN \
  --num_warmup_steps=0 \
  --optimizer=lamb \
  --save_checkpoints_steps=$SAVE_CKPT_STEPS \
  --start_warmup_step=0 \
  --num_gpus=$NUM_GPUS \
  --train_batch_size=$GLOBAL_BATCH_SIZE 2>&1 | tee ${OUTPUT_DIR}/app.log

# Perform an evaluation
horovodrun -np $NUM_GPUS python run_pretraining.py \
  --bert_config_file=${WIKI_DIR}/bert_config.json \
  --output_dir=${OUTPUT_DIR} \
  --log_dir=${OUTPUT_DIR} \
  --input_file="${DATA_DIR}/eval_10k" \
  --nodo_train \
  --do_eval \
  --eval_batch_size=8 \
  --learning_rate=0.0001 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_warmup_steps=0 \
  --optimizer=lamb \
  --save_checkpoints_steps=$SAVE_CKPT_STEPS \
  --start_warmup_step=0 \
  --max_eval_steps=100 \
  --num_gpus=$NUM_GPUS | tee -a ${OUTPUT_DIR}/app.log
  
  # --init_checkpoint=${WIKI_DIR}/ckpt/model.ckpt-28252 \
  # end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="BERT"

echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
