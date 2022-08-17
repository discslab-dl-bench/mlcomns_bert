#!/bin/bash

# To be run in the docker container
# Needs /data (containing tfrecords of data), /wiki and /output mounted

NUM_GPUS=8

if [ $# -eq 1 ] 
then
	NUM_GPUS=$1
fi

DATA_DIR="/data"
WIKI_DIR="/wiki"
OUTPUT_DIR="/output"

TF_XLA_FLAGS='--tf_xla_auto_jit=2'

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

python run_pretraining.py \
  --bert_config_file=${WIKI_DIR}/bert_config.json \
  --output_dir=${OUTPUT_DIR} \
  --input_file="${DATA_DIR}/*" \
  --nodo_eval \
  --do_train \
  --eval_batch_size=8 \
  --learning_rate=0.0001 \
  --init_checkpoint=${WIKI_DIR}/ckpt/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_train_steps=107538 \
  --num_warmup_steps=1562 \
  --optimizer=lamb \
  --save_checkpoints_steps=6250 \
  --start_warmup_step=0 \
  --num_gpus=$NUM_GPUS \
  --train_batch_size=24
  
  # end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="BERT"

echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
