#!/usr/bin/env bash

MODEL_PREFIX=/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/HRED/1554192029.6059802_UbuntuModel

CONTEXT=/home/cgsdfc/UbuntuDialogueCorpus/ResponseContextPairs/raw_testing_contexts.txt

OUTPUT=/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/HRED/output.txt_tesing.txt

THEANO_FLAGS=device=cuda1 \
python bin/sample.py \
    $MODEL_PREFIX \
    $CONTEXT \
    $OUTPUT \
    --verbose
