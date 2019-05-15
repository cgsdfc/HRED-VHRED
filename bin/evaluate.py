#!/usr/bin/env python
"""
Evaluation script.

For paper submissions, this script should normally be run with flags and both with and without the flag --exclude-stop-words.

Run example:

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,allow_gc=True,scan.allow_gc=False,nvcc.flags=-use_fast_math python evaluate.py Output/1432724394.9_MovieScriptModel &> Test_Eval_Output.txt

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
from pathlib import Path

import numpy

from serban.data_iterator import get_test_iterator
from serban.dialog_encoder_decoder import DialogEncoderDecoder
from serban.state import prototype_state

logger = logging.getLogger(__name__)

# List of all 77 English pronouns, all punctuation signs included in Movie-Scriptolog and other special tokens.
stopwords = "all another any anybody anyone anything both each each other either everybody everyone everything" \
            " few he her hers herself him himself his I it its itself many me mine more most much myself neither" \
            " no one nobody none nothing one one another other others ours ourselves several she some somebody" \
            " someone something that their theirs them themselves these they this those us we what whatever" \
            " which whichever who whoever whom whomever whose you your yours yourself yourselves" \
            " . , ? ' - -- ! <unk> </s> <s>"


def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")

    parser.add_argument("model_prefix",
                        help="Path to the model prefix (without _model.npz or _state.pkl)")
    parser.add_argument('--save-dir')

    parser.add_argument("--test-path",
                        type=str, help="File of test data")

    parser.add_argument('-e', "--exclude-stop-words", action="store_true",
                        help="Exclude stop words "
                             "(English pronouns, punctuation signs and special tokens) from all metrics."
                             " These words make up approximate 48.37% of the training set,"
                             " so removing them should focus the metrics on the topical content "
                             "and ignore syntactic errors.")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    state = prototype_state()

    state_path = save_dir.joinpath(args.model_prefix + "_state.pkl")
    model_path = save_dir.joinpath(args.model_prefix + "_model.npz")

    with open(state_path, 'rb') as src:
        state.update(pickle.load(src))
    state['bs'] = 1  # utterance level
    state['sort_k_batches'] = 1

    logging.basicConfig(level=getattr(logging, state['level']),
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    model = DialogEncoderDecoder(state)
    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")

    eval_batch = model.build_eval_function()

    if args.test_path:
        state['test_dialogues'] = args.test_path
    else:
        raise ValueError('test_path not given')

    # Initialize list of stopwords to remove
    stopwords_indices = []
    if args.exclude_stop_words:
        logger.debug("Initializing stop-word list")
        stopwords_lowercase = stopwords.lower().split(' ')
        for word in stopwords_lowercase:
            if word in model.str_to_idx:
                stopwords_indices.append(model.str_to_idx[word])

    test_data = get_test_iterator(state)
    test_data.start()

    # Variables to store test statistics
    test_cost = 0  # negative log-likelihood
    total_word_preds = 0  # number of words in total

    logger.debug("[TEST START]")
    while True:
        batch = test_data.next()
        # Train finished
        if not batch:
            break

        logger.debug("[TEST] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))

        x_data = batch['x']
        x_data_reversed = batch['x_reversed']
        max_length = batch['max_length']
        x_cost_mask = batch['x_mask']
        reset_mask = batch['x_reset']
        ran_cost_utterance = batch['ran_var_constutterance']
        ran_decoder_drop_mask = batch['ran_decoder_drop_mask']

        if args.exclude_stop_words:
            for word_index in stopwords_indices:
                x_cost_mask[x_data == word_index] = 0

        num_preds = numpy.sum(x_cost_mask)

        c, _, c_list, _, _ = eval_batch(x_data, x_data_reversed, max_length, x_cost_mask, reset_mask,
                                        ran_cost_utterance, ran_decoder_drop_mask)

        if numpy.isinf(c) or numpy.isnan(c):
            continue

        utterance_word_ppl = math.exp(c / num_preds)
        print('utterance word-perplexity = {}'.format(utterance_word_ppl))

        test_cost += c
        total_word_preds += num_preds

    logger.debug("[TEST END]")
    test_cost /= total_word_preds
    print('system word-perplexity = {}'.format(math.exp(test_cost)))
    logger.debug("All done, exiting...")


if __name__ == "__main__":
    main()
