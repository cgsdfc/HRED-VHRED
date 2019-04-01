#!/usr/bin/env bash

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=cuda \
    python sample.py <model_name> <contexts> <model_outputs> --beam_search --n-samples=<beams> --ignore-unk --verbose