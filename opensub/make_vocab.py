import argparse

MOVIE_VOCAB = '/home/cgsdfc/OpenSubData/movie_25000'

SPECIAL = {
    '<unk>': 0,
    '</s>': 1,
    '</d>': 2,
    '<first_speaker>': 3,
    '<second_speaker>': 4,
    '<third_speaker>': 5,
    '<minor_speaker>': 6,
    '<voice_over>': 7,
    '<off_screen>': 8,
    '<pause>': 9,
}


def convert_vocab(input_file, output_file):
    vocab = SPECIAL.copy()
    with open(input_file) as f:



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
