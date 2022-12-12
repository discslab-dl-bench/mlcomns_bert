#!/bin/bash

# To be run in the docker container
# Needs /data (containing tfrecords of data), /wiki and /output mounted

NUM_GPUS=${1:-8}
BATCH_SIZE=${2:-48}

if [ $# -eq 1 ] 
then
	NUM_GPUS=$1
	BATCH_SIZE=$(expr 6 \* $NUM_GPUS)
fi

NUM_STEPS_TRAIN=300
SAVE_CKPT_STEPS=300

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

NUM_PROCS=$(( $NUM_GPUS - 1 ))

for i in $(seq 0 $NUM_PROCS); do
  worker_list[${i}]="\"localhost:1234${i}\""
done

worker_list=$(printf ",%s" "${worker_list[@]}")
worker_list=${worker_list:1}
echo $worker_list

for i in $(seq 0 $NUM_PROCS); do
  TF_CONFIG="{ \"cluster\": { \"worker\": [${worker_list}] }, \"task\": {\"type\": \"worker\", \"index\": ${i} }}"
  env TF_CONFIG="$TF_CONFIG" \
  CUDA_VISIBLE_DEVICES=$i \
  python run_pretraining.py \
    --bert_config_file=${WIKI_DIR}/bert_config.json \
    --output_dir=${OUTPUT_DIR} \
    --log_dir=${OUTPUT_DIR} \
    --input_file="${DATA_DIR}/part*" \
    --nodo_eval \
    --do_train \
    --eval_batch_size=$BATCH_SIZE \
    --learning_rate=0.0001 \
    --iterations_per_loop=1000 \
    --max_predictions_per_seq=76 \
    --max_seq_length=512 \
    --num_train_steps=$NUM_STEPS_TRAIN \
    --num_warmup_steps=0 \
    --optimizer=lamb \
    --save_checkpoints_steps=$SAVE_CKPT_STEPS \
    --start_warmup_step=0 \
    --num_gpus=1 \
    --train_batch_size=$BATCH_SIZE > ${OUTPUT_DIR}/job_${i}.log 2>&1 &
    # --train_batch_size=$BATCH_SIZE 2>&1 | tee ${OUTPUT_DIR}/job_${i}.log &

  workers[${i}]=$!
done

for pid in ${workers[*]}; do
  wait $pid
done

  # --init_checkpoint=${WIKI_DIR}/ckpt/model.ckpt-28252 \
  # end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="BERT"

echo "RESULT,$result_name,$SEED,$result,$USER,$start_fmt"
