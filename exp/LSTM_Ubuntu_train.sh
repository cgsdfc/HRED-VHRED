#!/usr/bin/env bash

THEANO_FLAGS=device=cuda1 \
python bin/train.py prototype_ubuntu_LSTM \
    --auto_restart
