#!/usr/bin/env bash

IMAGE=ufoym/deepo:theano-py36-cu90

PROJECT_PATH=/home/cgsdfc/deployment/Models/HRED-VHRED

docker run --runtime nvidia --rm -it \
    -v $HOME:$HOME \
    -w $PROJECT_PATH \
    -e PYTHONPATH=$PROJECT_PATH \
    $IMAGE  bash
