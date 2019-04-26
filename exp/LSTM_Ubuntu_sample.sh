#!/usr/bin/env bash

MODEL_PREFIX=/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/LSTM/1554524954.6212888_UbuntuModel
CONTEXT=/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_contexts.txt
OUTPUT=/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/LSTM/output.txt

THEANO_FLAGS=device=cuda1 \
python bin/sample.py \
    $MODEL_PREFIX \
    $CONTEXT \
    $OUTPUT