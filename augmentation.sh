#!/bin/bash

method=$1
for num in {001..002}
do
    echo "method: ${method} case: ${num}"
   python3 augmentation.py \
   --input /raid/data/bert/raw-data/part-00${num}-of-00500 \
   --output /raid/data/bert/${method}/part-00${num}-of-00500 \
   --method ${method} \
   --number 0.5
done

