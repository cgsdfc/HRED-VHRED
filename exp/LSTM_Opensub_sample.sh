#!/usr/bin/env bash

MODEL_PREFIX=/home/cgsdfc/SavedModels/HRED-VHRED/OpenSubData/LSTM/1555823246.5508652
CONTEXT=/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_contexts.txt
OUTPUT=/home/cgsdfc/SavedModels/HRED-VHRED/OpenSubData/LSTM/output.txt

THEANO_FLAGS=device=cuda1 \
python bin/sample.py \
    $MODEL_PREFIX \
    $CONTEXT \
    $OUTPUT
