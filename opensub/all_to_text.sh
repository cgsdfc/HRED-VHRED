#!/usr/bin/env bash


INPUT_DIR=/home/cgsdfc/OpenSubData
OUTPUT_DIR=/home/cgsdfc/Serban_OpenSubData
DICT_FILE=/home/cgsdfc/OpenSubData/movie_25000

python opensub/to_plain_text.py \
    --input-dir $INPUT_DIR \
    --output-dir $OUTPUT_DIR \
    --dict-file $DICT_FILE
