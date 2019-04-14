#!/usr/bin/env bash

IMAGE=ufoym/deepo:theano-py36-cu90

PROJECT_PATH=/home/cgsdfc/deployment/Models/Dialogue/HRED-VHRED
UBUNTU_DATA_PATH=/home/cgsdfc/UbuntuDialogueCorpus
SAVE_ROOT=/home/cgsdfc/SavedModels/HRED-VHRED
THEANO_CACHE_DIR=$HOME/.theano/
THEANO_DEST_DIR=/root/.theano

docker run --runtime nvidia --rm -it \
    -v $PROJECT_PATH:$PROJECT_PATH \
    -v $UBUNTU_DATA_PATH:$UBUNTU_DATA_PATH \
    -v $SAVE_ROOT:$SAVE_ROOT \
    -v $THEANO_CACHE_DIR:$THEANO_DEST_DIR \
    -w $PROJECT_PATH \
    -e PYTHONPATH=$PROJECT_PATH \
    $IMAGE \
    bash
