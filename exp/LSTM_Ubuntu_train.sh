#!/usr/bin/env bash

SAVE_DIR=/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/LSTM

THEANO_FLAGS=device=cuda1 \
python bin/train.py prototype_ubuntu_LSTM \
    --save-dir $SAVE_DIR \
    --auto_restart
