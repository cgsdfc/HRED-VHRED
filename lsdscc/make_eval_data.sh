#!/usr/bin/env bash

TEST_FILE=/home/cgsdfc/SerbanLSDSCC/raw_test_dialogues.txt

python lsdscc/convert.py -x -i $TEST_FILE
