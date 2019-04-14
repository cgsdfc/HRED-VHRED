import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dialogues')
    parser.add_argument('test_dialogues')
    parser.add_argument('valid_dialogues')

    parser.add_argument('-seed', default=1234)
    parser.add_argument('-level', default='DEBUG')
    parser.add_argument('-oov', default='<unk>', help='Out-of-vocabulary token string')
    parser.add_argument('-end_sym_utterance', default='</s>', help='end-of-sequence marks')

    parser.add_argument('unk_sym', default=0)
    parser.add_argument('eos_sym', default=1)
    parser.add_argument('eod_sym', default=2)

    parser.add_argument('first_speaker_sym', default=3)
    parser.add_argument('second_speaker_sym')
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    return parser.parse_args()