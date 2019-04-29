#!/usr/bin/env bash

IMAGE=ufoym/deepo:theano-py36-cu90

PROJECT_PATH=/home/cgsdfc/deployment/Models/HRED-VHRED
UBUNTU_DATA_PATH=/home/cgsdfc/UbuntuDialogueCorpus
OPENSUB_DATA_PATH=/home/cgsdfc/SerbanOpenSubData
LSDSCC_DATA_PATH=/home/cgsdfc/SerbanLSDSCC
RESULT_PATH=/home/cgsdfc/Result/HRED-VHRED
SAVE_ROOT=/home/cgsdfc/SavedModels

docker run --runtime nvidia --rm -it \
    -v $PROJECT_PATH:$PROJECT_PATH \
    -v $UBUNTU_DATA_PATH:$UBUNTU_DATA_PATH \
    -v $OPENSUB_DATA_PATH:$OPENSUB_DATA_PATH \
    -v $LSDSCC_DATA_PATH:$LSDSCC_DATA_PATH \
    -v $RESULT_PATH:$RESULT_PATH \
    -v $SAVE_ROOT:$SAVE_ROOT \
    -w $PROJECT_PATH \
    -e PYTHONPATH=$PROJECT_PATH \
    $IMAGE \
    bash
