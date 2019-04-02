#!/usr/bin/env bash

IMAGE=ufoym/deepo:theano-py36-cu90
LOCAL_PATH=/home/cgsdfc/deployment/HRED-VHRED
REMOTE_PATH=/root/HRED-VHRED

cd $LOCAL_PATH

docker run --runtime=nvidia \
    -v $LOCAL_PATH:$REMOTE_PATH \
    -w $REMOTE_PATH \
    -e PYTHONPATH=$REMOTE_PATH \
    ${IMAGE} \
    python bin/train.py prototype_test_VHRED
