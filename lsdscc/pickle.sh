#!/usr/bin/env bash

INPUT=/home/cgsdfc/Serban_LSDSCC/dataset.txt.words
OUTPUT=/home/cgsdfc/Serban_LSDSCC/Dataset
CUTOFF=51200

python bin/convert_text2dict.py $INPUT $OUTPUT --cutoff $CUTOFF
