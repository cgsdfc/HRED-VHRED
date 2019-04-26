#!/usr/bin/env bash

SAVE_DIR=/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/VHRED

THEANO_FLAGS=device=cuda0 \
python bin/train.py prototype_ubuntu_VHRED \
    --auto_restart \
    --save-dir $SAVE_DIR
