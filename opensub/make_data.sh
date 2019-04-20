#!/usr/bin/env bash

CUTOFF=25000
OUTPUTS=(Training Validation Test)
PREFIX=/home/cgsdfc/OpenSubData/serban



python bin/convert_text2dict.py --dict $dict $out
