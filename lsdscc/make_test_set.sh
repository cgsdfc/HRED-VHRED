#!/usr/bin/env bash

# Generate the test set from the `test.group.json`.
CUTOFF=40000
GROUP_JSON=/home/cgsdfc/LSDSCC-Reddit-Movie/test.group.json
TEST_WORDS=/home/cgsdfc/SerbanLSDSCC/raw_test_dialogues.txt
PREFIX=/home/cgsdfc/SerbanLSDSCC/vocab_$CUTOFF/Test
DICT_FILE=/home/cgsdfc/SerbanLSDSCC/vocab_$CUTOFF/Train.dict.pkl

python lsdscc/convert.py -c -i $GROUP_JSON -o $TEST_WORDS
python bin/convert_text2dict.py $TEST_WORDS $PREFIX --dict $DICT_FILE
