#!/usr/bin/env bash

MODEL_PREFIX=/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/VHRED/1554192143.9237187_UbuntuModel

CONTEXT=/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_contexts.txt

OUTPUT=/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/VHRED/output.txt

THEANO_FLAGS=device=cuda0 \
python bin/sample.py \
    $MODEL_PREFIX \
    $CONTEXT \
    $OUTPUT \
    --verbose
