#!/usr/bin/env bash
SERBAN_IMAGE=ufoym/deepo:theano-py36-cu90
SERBAN_ROOT=/home/cgsdfc/deployment/Models/HRED-VHRED
GPU_INDEX=0

MODEL_PREFIX=1556265276.0421634_UbuntuModel
TEST_PATH=/home/cgsdfc/UbuntuDialogueCorpus/Test.dialogues.pkl
SAVE_DIR=/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/LSTM

BS=80

docker run --rm -it --runtime nvidia \
    --name serban_ppl \
    -v $HOME:$HOME \
    -w $SERBAN_ROOT \
    -e PYTHONPATH=$SERBAN_ROOT \
    -e THEANO_FLAGS=device=cuda$GPU_INDEX \
    $SERBAN_IMAGE \
    python bin/evaluate.py $MODEL_PREFIX  --test-path $TEST_PATH --save-dir $SAVE_DIR --bs $BS
