import argparse
import logging
from pathlib import Path
import os

FIRST_SPEAKER = '<first_speaker>'
SECOND_SPEAKER = '<second_speaker>'
EOS = '</s>'
EOD = '</d>'
SEP = '|'
UNK = '<unk>'
UNK_ID = 1


def load_dict(file):
    """
    Create a vocab dict from a file.
    Index starts from 0, which is for the unknown token.

    :param file:
    :return:
    """
    with open(file) as f:
        return dict(enumerate(f.read().splitlines(), start=1))


def get_examples(filename, dict):
    def ids_to_words(ids_string):
        return [dict[id] if id != UNK_ID else UNK for id in map(int, ids_string.split())]

    def make_text_lines(utterances):
        utterances = list(map(lambda words: ' '.join(words), utterances))
        return EOS.join(' ' * 2).join(utterances)

    with open(filename) as f:
        for line in f:
            utterances = line.split(SEP)
            utterances = list(map(ids_to_words, utterances))
            utterances = make_text_lines(utterances)
            yield utterances


def make_one_file(dialog_file, output, dict):
    with open(output, 'w') as f:
        for example in get_examples(dialog_file, dict):
            print(example, file=f)


def make_one_dir(input_dir, output_dir, dict):
    def output_basename(name):
        stem, ext = os.path.splitext(name)
        return '.words'.join((stem, ext))

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for subdir in input_dir.iterdir():
        for id_file in subdir.glob('*.txt'):
            logging.info('subdir: %s', subdir.name)
            prefix = output_dir.joinpath(subdir.name)
            if not prefix.is_dir():
                prefix.mkdir()

            output = prefix.joinpath(output_basename(id_file.name))
            if output.is_file():
                logging.info('skipping existing file: %s', output)
            else:
                logging.info('convert %s to %s', id_file, output)
                make_one_file(id_file, output, dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialog-file')
    parser.add_argument('--dict-file')
    parser.add_argument('--output')

    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dict = load_dict(args.dict_file)

    if args.dialog_file:
        logging.info('dialog_file: %s', args.dialog_file)
        make_one_file(args.dialog_file, args.output, dict)
    else:
        logging.info('input_dir: %s', args.input_dir)
        logging.info('output_dir: %s', args.output_dir)
        make_one_dir(args.input_dir, args.output_dir, dict)
