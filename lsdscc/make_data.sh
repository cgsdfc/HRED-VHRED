#!/usr/bin/env bash

RAW_INPUT=/home/cgsdfc/LSDSCC-Reddit-Movie/dataset.txt
INPUT=/home/cgsdfc/Serban_LSDSCC/dataset.txt.words
OUTPUT=/home/cgsdfc/Serban_LSDSCC/Dataset
CUTOFF=51200 # Value from the paper

# Normalize to required format.
python lsdscc/convert.py -r --input $RAW_INPUT --output $INPUT

# Generate dictionary and one big dialogues file.
python bin/convert_text2dict.py $INPUT $OUTPUT --cutoff $CUTOFF

INPUT=/home/cgsdfc/Serban_LSDSCC/Dataset.dialogues.pkl

# Split the big dialogues into train, valid and test files.
python lsdscc/convert.py -s --input $INPUT
