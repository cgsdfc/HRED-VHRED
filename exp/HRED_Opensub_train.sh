#!/usr/bin/env bash

GPU_ID=1

THEANO_FLAGS=device=cuda${GPU_ID} \
python bin/train.py \
    prototype_opensubtitles_HRED \
    --auto_restart \
    --save-dir /home/cgsdfc/SavedModels/HRED-VHRED/OpenSubData/HRED
