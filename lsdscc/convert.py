import argparse
import pickle
import re
import logging
from pathlib import Path

import numpy as np

LSDSCC_SEP = '<EOS>#TAB#'
LSDSCC_SEP_RE = re.compile(LSDSCC_SEP)

# The desired EOS token.
EOS = '</s>'.join(' ' * 2)

LSDSCC_UNK = 'unknown'
LSDSCC_UNK_RE = re.compile(LSDSCC_UNK)
UNK = '<unk>'

SUFFIX = '.dialogues.pkl'

# Consts from the paper. Used to verify the actual data.
RESPONSE_MAX_LEN = 60
QUERY_MAX_LEN = 100


def normalize(input, output, verbose):
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
                line = LSDSCC_UNK_RE.sub(UNK, line)
                yield line

    with open(output, 'w') as out:
        for line in replace_each_line():
            out.write(line)


def split_train_test(filename):
    filename = Path(filename)
    output_dir = filename.parent
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
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('-r', '--replace', action='store_true', help='replace <eos> and <unk> token in input')
    parser.add_argument('-s', '--split', action='store_true', help='split a big dialogues pickle file into 3 files')
    parser.add_argument('-v', '--verbose')

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.replace:
        normalize(args.input, args.output, args.verbose)
    elif args.split:
        split_train_test(args.input)
