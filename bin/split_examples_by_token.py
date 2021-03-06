"""
Takes as input a binarized dialogue corpus, splits the examples by a certain token and shuffles it

Example run:

   python split_examples_by_token.py Training.dialogues.pkl 2 Training_SplitByDialogues.dialogues --join_last_two_examples

@author Iulian Vlad Serban
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import math
import os
import pickle
import numpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('text2dict')


def safe_pickle(obj, filename):
    if os.path.isfile(filename):
        logger.info("Overwriting %s." % filename)
    else:
        logger.info("Saving to %s." % filename)

    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# Thanks to Emile on Stackoverflow:
# http://stackoverflow.com/questions/4322705/split-a-list-into-nested-lists-on-a-value

def _iter_split(l, splitters):
    current = []
    for item in l:
        if item in splitters:
            yield current
            current = []
        else:
            current.append(item)
    yield current


def magic_split(l, *splitters):
    return [subl for subl in _iter_split(l, splitters) if subl]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Binarized dialogue corpus (pkl file)")

    parser.add_argument("token_id", type=int,
                        help="Token index to split examples by (e.g. to split by end-of-dialogue set this to 2)")

    parser.add_argument("consecutive_examples_to_merge", type=int, default=1,
                        help="After splitting these number of examples will be merged.")

    parser.add_argument("--join_last_two_examples",
                        action="store_true",
                        help="If on, will join the last two splits generated from each example. "
                             "This is useful to handle empty or very short last samples")

    parser.add_argument("output", type=str, help="Filename of processed binarized dialogue corpus (pkl file)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise Exception("Input file not found!")

    logger.info("Loading dialogue corpus")

    with open(args.input, 'rb') as f:
        data = pickle.load(f)
    data_len = len(data)

    logger.info('Corpus loaded... Data len is %d' % data_len)

    # Count number of tokens
    tokens_count = 0
    for i in range(data_len):
        tokens_count += len(data[i])
    logger.info('Tokens count %d' % tokens_count)

    logger.info("Splitting corpus examples by token id... ")
    processed_binarized_corpus = []
    for i in range(data_len):
        logger.info('    Example %d ' % i)
        new_examples = magic_split(data[i], int(args.token_id))

        # If option is specified, we append the last new example to the second last one
        if args.join_last_two_examples and len(new_examples) > 1:
            new_examples[len(new_examples) - 2] += new_examples[len(new_examples) - 1]
            del new_examples[len(new_examples) - 1]

        # Simpler version of the two for loops, which does not allow merging together samples
        # for new_example in new_examples:
        #    processed_binarized_corpus.append(new_example + [int(args.token_id)])

        s = int(math.floor(len(new_examples) / args.consecutive_examples_to_merge))
        for j in range(1, s):
            start_index = j * args.consecutive_examples_to_merge
            merged_example = []
            for k in reversed(range(args.consecutive_examples_to_merge)):
                merged_example += new_examples[start_index - k - 1] + [int(args.token_id)]
            processed_binarized_corpus.append(merged_example)

        if s > 0:
            merged_example = []
            for k in range((s - 1) * args.consecutive_examples_to_merge, len(new_examples)):
                merged_example += new_examples[k] + [int(args.token_id)]
            processed_binarized_corpus.append(merged_example)
        else:
            merged_example = []
            for k in range(len(new_examples)):
                merged_example += new_examples[k] + [int(args.token_id)]
            processed_binarized_corpus.append(merged_example)

    logger.info('New data len is %d' % len(processed_binarized_corpus))

    # Count number of tokens
    processed_tokens_count = 0
    for i in range(len(processed_binarized_corpus)):
        processed_tokens_count += len(processed_binarized_corpus[i])
    logger.info('New tokens count %d' % processed_tokens_count)

    # When splitting by end-of-utterance token </s>,
    # there are some instances with multiple </s> at the end of each example.
    # Our splitting method will effectively remove these, but it is not of any concern to us.
    # assert(processed_tokens_count == tokens_count)

    logger.info("Reshuffling corpus.")
    rng = numpy.random.RandomState(13248)
    rng.shuffle(processed_binarized_corpus)

    logger.info("Saving corpus.")
    safe_pickle(processed_binarized_corpus, args.output + ".pkl")

    logger.info("Corpus saved. All done!")
