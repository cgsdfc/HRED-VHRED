#!/usr/bin/env bash

# Generate training dataset.
CUTOFF=35000
RAW_INPUT=/home/cgsdfc/LSDSCC-Reddit-Movie/dataset.txt
INPUT=/home/cgsdfc/SerbanLSDSCC/vocab_${CUTOFF}/dataset.txt.words
OUTPUT=/home/cgsdfc/SerbanLSDSCC/vocab_${CUTOFF}/Train

# Normalize to required format.
python lsdscc/convert.py -r --input $RAW_INPUT --output $INPUT

# Generate dictionary and one big dialogues file.
python bin/convert_text2dict.py $INPUT $OUTPUT --cutoff $CUTOFF
