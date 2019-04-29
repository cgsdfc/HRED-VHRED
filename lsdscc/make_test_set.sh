#!/usr/bin/env bash

GROUP_JSON=/home/cgsdfc/LSDSCC-Reddit-Movie/test.group.json
TEST_WORDS=/home/cgsdfc/SerbanLSDSCC/raw_test_dialogues.txt
TEST_PKL=/home/cgsdfc/SerbanLSDSCC/Test
DICT_FILE=/home/cgsdfc/SerbanLSDSCC/Dataset.dict.pkl

python lsdscc/convert.py -c -i $GROUP_JSON -o $TEST_WORDS
python bin/convert_text2dict.py $TEST_WORDS $TEST_PKL --dict $DICT_FILE
