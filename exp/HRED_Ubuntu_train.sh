#!/usr/bin/env bash

THEANO_FLAGS=device=cuda0 \
python bin/train.py prototype_ubuntu_HRED \
    --auto_restart
