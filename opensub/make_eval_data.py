import argparse
from pathlib import Path
import logging
import tqdm

EOS = '</s>'

CONTEXT_SUFFIX = '.context.txt'
RESPONSE_SUFFIX = '.response.txt'


def get_output_files(input: Path):
    parent = input.parent
    prefix = input.name.split('.')[0]
    return [parent.joinpath(prefix + suffix) for suffix in (CONTEXT_SUFFIX, RESPONSE_SUFFIX)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    input = Path(args.input)
    logging.info('input: %s', input)

    context, response = get_output_files(input)
    logging.info('output context: %s', context)
    logging.info('output response: %s', response)

    with open(input) as f:
        with open(context, 'w') as c, open(response, 'w') as r:
            for line in tqdm.tqdm(f):
                utters = line.strip().split(EOS)
                assert len(utters) >= 2, 'Invalid data format'

                ctx = EOS.join(utters[:-1]) + ' ' + EOS
                resp = utters[-1] + ' ' + EOS
                print(ctx, file=c)
                print(resp, file=r)
