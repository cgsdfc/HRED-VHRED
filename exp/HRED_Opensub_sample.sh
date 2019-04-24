#!/usr/bin/env bash

MODEL_PREFIX=/home/cgsdfc/SavedModels/HRED-VHRED/OpenSubData/HRED/1555817830.719177

CONTEXT=/home/cgsdfc/SerbanOpenSubData/dialogue_length2_6/test.context.txt

OUTPUT=/home/cgsdfc/SavedModels/HRED-VHRED/OpenSubData/HRED/output.txt

GPU=1

THEANO_FLAGS=device=cuda$GPU \
python bin/sample.py \
    $MODEL_PREFIX \
    $CONTEXT \
    $OUTPUT
