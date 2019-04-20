#!/usr/bin/env bash

FILE=$HOME/UbuntuDialogueCorpus/Dataset.dict.pkl

python -m pickle $FILE | less
