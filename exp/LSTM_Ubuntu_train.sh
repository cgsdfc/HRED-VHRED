#!/usr/bin/env bash

SAVE_DIR=/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/LSTM
GPU=0

THEANO_FLAGS=device=cuda${GPU} \
python bin/train.py prototype_ubuntu_LSTM \
    --save-dir $SAVE_DIR \
    --auto_restart \
    --prefix 1554524954.6212888_UbuntuModel
