#!/usr/bin/env bash

THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 python train.py prototype_test_variational
