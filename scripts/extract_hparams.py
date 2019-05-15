import pprint
import argparse
import itertools
from operator import itemgetter

import serban.state

models = ['VHRED', 'LSTM', 'HRED']
datasets = ['ubuntu', 'opensubtitles', 'lsdscc']


def iter_prototypes():
    for m, d in itertools.product(models, datasets):
        name = 'prototype_{}_{}'.format(d, m)
        value = eval(name + '()', serban.state.__dict__)
        yield m, d, value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--key', help='key to inspect')
    args = parser.parse_args()
    if args.key is None:
        for item in iter_prototypes():
            pprint.pprint(item)
    else:
        payload = sorted(iter_prototypes(), key=itemgetter(1))
        for model, dataset, prototype in payload:
            print(model, dataset, prototype[args.key])
