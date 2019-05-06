"""
The opensub test set has 1M example, which is way to large for testing purpose.
It runs on  a GPU for nearly 2 days.
"""

import argparse
from pathlib import Path
import logging
import pickle

import pandas as pd

DEF_FRAC = 0.01

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help='output dir')
    parser.add_argument('-c', dest='context_file')
    parser.add_argument('-r', dest='reference_file')
    parser.add_argument('-d', dest='dialogues_file')
    parser.add_argument('-f', dest='frac', type=float,
                        help='fraction of samples', default=DEF_FRAC)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.p)


    def load_file(path):
        logging.info('loading %s', path)
        return Path(path).read_text().splitlines()


    def save_file(input_file, series, binary=False):
        output = output_dir.joinpath(Path(input_file).name)
        logging.info('saving to %s', output)
        if binary:
            output.write_bytes(pickle.dumps(series))
        else:
            output.write_text('\n'.join(series))


    payload = {
        'contexts': load_file(args.context_file),
        'references': load_file(args.reference_file),
        'dialogues': pickle.load(Path(args.dialogues_file).read_bytes()),
    }

    logging.info('loading data frame')
    df = pd.DataFrame(data=payload)
    print(df.head())

    df = df.sample(frac=args.frac)
    logging.info('samples: %d', len(df))

    save_file(args.context_file, df['contexts'])
    save_file(args.reference_file, df['references'])
    save_file(args.dialouges_file, df['dialogues'], True)
