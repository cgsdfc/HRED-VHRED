#!/usr/bin/env bash

RAW_INPUT=/home/cgsdfc/LSDSCC-Reddit-Movie/dataset.txt
INPUT=/home/cgsdfc/SerbanLSDSCC/dataset.txt.words
OUTPUT=/home/cgsdfc/SerbanLSDSCC/Train
CUTOFF=30000

# Normalize to required format.
python lsdscc/convert.py -r --input $RAW_INPUT --output $INPUT

# Generate dictionary and one big dialogues file.
python bin/convert_text2dict.py $INPUT $OUTPUT --cutoff $CUTOFF
