import argparse
import pickle
import re
import logging
from pathlib import Path

import numpy as np

LSDSCC_SEP = '<EOS>#TAB#'
LSDSCC_SEP_RE = re.compile(LSDSCC_SEP)
EOS = '</s>'
EOS = EOS.join('  ')

UNKNOWN = 'unknown'
UNKNOWN_RE = re.compile(UNKNOWN)
UNK = '<unk>'

CONVERTER = 'convert_text2dict.py'
SUFFIX = '.dialogues.pkl'
DEF_INPUT = '/home/cgsdfc/LSDSCC-Reddit-Movie/dataset.txt'

RESPONSE_MAX_LEN = 60
QUERY_MAX_LEN = 100


def replace_eos_on_the_whole_file(input, output, verbose):
    def check_max_len(query, response):
        if len(query.split()) > QUERY_MAX_LEN:
            logging.info('bad query: %s', query)
        if len(response.split() > RESPONSE_MAX_LEN):
            logging.info('bad response: %s', response)

    def replace_each_line():
        with open(input) as f:
            for line in f:
                query, response = line.split(LSDSCC_SEP)
                if verbose:
                    check_max_len(query, response)

                line = EOS.join((query, response))
                line = UNKNOWN_RE.sub(UNK, line)
                yield line

    with open(output, 'w') as out:
        for line in replace_each_line():
            out.write(line)


def split_train_test(filename):
    filename = Path(filename)
    output_dir: Path = filename.parent
    logging.info('input_file: %s', filename)
    logging.info('output_dir: %s', output_dir)

    with open(filename, 'rb') as f:
        data = pickle.load(f)
    np.random.shuffle(data)
    logging.info('#dialogues: %d', len(data))

    files = {
        'Training': 80,
        'Validation': 10,
        'Test': 10,
    }

    offset = 0
    for prefix, percent in files.items():
        filename = output_dir.joinpath(prefix + SUFFIX)
        size = round(percent * len(data) / 100)
        chunk = data[offset: offset + size]
        logging.info('%d dialogues write to %s', len(chunk), filename)
        with open(filename, 'wb') as f:
            pickle.dump(chunk, f)
        offset += size


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=DEF_INPUT)
    parser.add_argument('--output')
    parser.add_argument('-r', '--replace', action='store_true', help='replace <eos> and <unk> token in input')
    parser.add_argument('-s', '--split', action='store_true', help='split a big dialogues pickle file into 3 files')
    parser.add_argument('-v', '--verbose')

    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()

    if args.replace:
        replace_eos_on_the_whole_file(args.input, args.output, args.verbose)
    elif args.split:
        split_train_test(args.input)
