#!/usr/bin/env bash

SAVE_DIR=/home/cgsdfc/SavedModels/HRED-VHRED/Ubuntu/HRED

THEANO_FLAGS=device=cuda0 \
python bin/train.py prototype_ubuntu_HRED \
    --save-dir $SAVE_DIR \
    --prefix $SAVE_DIR/1554192029.6059802_UbuntuModel \
    --auto_restart
