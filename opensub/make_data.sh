#!/usr/bin/env bash


INPUT=/home/cgsdfc/OpenSubData/dialogue_length2_3/train.txt
PREFIX=/home/cgsdfc/SerbanOpenSubData/dialogue_length2_3

python
python bin/convert_text2dict.py $INPUT ${PREFIX}/Training
