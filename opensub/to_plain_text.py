import argparse

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


def make_all(dialog_file, dict_file, output):
    dict = load_dict(dict_file)
    with open(output, 'w') as f:
        for example in get_examples(dialog_file, dict):
            print(example, file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialog-file')
    parser.add_argument('--dict-file')
    parser.add_argument('--output')
    args = parser.parse_args()

    make_all(args.dialog_file, args.dict_file, args.output)
