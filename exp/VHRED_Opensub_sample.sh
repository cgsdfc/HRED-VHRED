#!/usr/bin/env bash

MODEL_PREFIX=
CONTEXT=/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.context.txt
OUTPUT=/home/cgsdfc/SavedModels/HRED-VHRED/OpenSubData/VHRED/output.txt
GPU=1

THEANO_FLAGS=device=cuda$GPU \
python bin/sample.py \
    $MODEL_PREFIX \
    $CONTEXT \
    $OUTPUT
