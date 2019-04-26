#!/usr/bin/env bash

SAVE_DIR=/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/HRED

THEANO_FLAGS=device=cuda0 \
python bin/train.py prototype_ubuntu_HRED \
    --auto_restart \
    --save-dir $SAVE_DIR
