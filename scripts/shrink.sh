#!/usr/bin/env bash

python scripts/shrink_opensub_test_data.py \
    -c /home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.context.txt \
    -r /home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.response.txt \
    -d /home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/test.words.txt \
    -p /home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/eval
