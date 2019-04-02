#!/usr/bin/env bash

IMAGE=ufoym/deepo:theano-py36-cu90
PROJECT_PATH=/home/cgsdfc/deployment/HRED-VHRED
UBUNTU_DATA_PATH=/home/cgsdfc/UbuntuDialogueCorpus


docker run --runtime=nvidia --rm \
    -v $PROJECT_PATH:$PROJECT_PATH \
    -v $UBUNTU_DATA_PATH:$UBUNTU_DATA_PATH \
    -w $PROJECT_PATH \
    -e PYTHONPATH=$PROJECT_PATH \
    -e THEANO_FLAGS=device=cuda1 \
    ${IMAGE} \
    python bin/train.py prototype_ubuntu_LSTM \
    --auto_restart
