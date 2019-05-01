#!/usr/bin/env bash

# Generate evaluation data: one file for query, another for ground truth.
TEST_FILE=/home/cgsdfc/SerbanLSDSCC/raw_test_dialogues.txt

python lsdscc/convert.py -x -i $TEST_FILE
