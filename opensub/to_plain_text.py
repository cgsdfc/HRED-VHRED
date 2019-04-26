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


# digits|digits => utterance </s> </d> utterance </s>

def load_dict(file):
    """
    Create a vocab dict from a file.

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
        return utterances

    with open(filename) as f:
        for line in f:
            utterances = line.split(SEP)
            utterances = list(map(ids_to_words, utterances))
            utterances = make_text_lines(utterances)

            context = ' '.join((utterances[:-1], EOS, EOD))
            response = ' '.join((utterances[-1], EOS))
            words = ' '.join((context, response))
            yield context, response, words


def output_basename(name):
    stem, ext = os.path.splitext(name)
    return '.words'.join((stem, ext))


def make_output_names(input_path, output_dir):
    stem, ext = os.path.splitext(input_path.name)
    return [output_dir.joinpath(s.join((stem, ext))) for s in ('.context', '.response', '.words')]


def make_one_file(dialog_file, prefix, dict):
    dialog_file = Path(dialog_file)
    prefix = Path(prefix)

    context_file, response_file, words_file = make_output_names(dialog_file, prefix)
    logging.info('context_file: %s', context_file)
    logging.info('response_file: %s', response_file)
    logging.info('words_file: %s', words_file)

    with open(context_file, 'w') as context_out, \
            open(response_file, 'w') as response_out, open(words_file, 'w') as out:
        for context, response, whole in get_examples(dialog_file, dict):
            print(context, file=context_out)
            print(response, file=response_out)
            print(whole, file=out)


def make_one_dir(input_dir, output_dir, dict):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for subdir in input_dir.iterdir():
        for dialog_file in subdir.glob('*.txt'):
            if 'shuffle' in dialog_file.name:
                continue
            logging.info('subdir: %s', subdir.name)
            prefix = output_dir.joinpath(subdir.name)
            if not prefix.is_dir():
                prefix.mkdir()
            make_one_file(dialog_file, prefix, dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialog-file')
    parser.add_argument('--dict-file')

    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    dict = load_dict(args.dict_file)
    if args.dialog_file:
        logging.info('dialog_file: %s', args.dialog_file)
        make_one_file(args.dialog_file, args.output_dir, dict)
    else:
        logging.info('input_dir: %s', args.input_dir)
        logging.info('output_dir: %s', args.output_dir)
        make_one_dir(args.input_dir, args.output_dir, dict)
