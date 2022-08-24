#!/bin/bash

python3 augmentation.py --input cleanup_scripts/sample_data/wiki_01 --output augmentation/wiki_01 --method random_insertion --number 5
python3 augmentation.py --input cleanup_scripts/sample_data/wiki_01 --output augmentation/wiki_01  --method random_swap --number 5
python3 augmentation.py --input cleanup_scripts/sample_data/wiki_01 --output augmentation/wiki_01  --method random_deletion --number 0.5
python3 augmentation.py --input cleanup_scripts/sample_data/wiki_01 --output augmentation/wiki_01  --method synonym_replacemnet --number 5
