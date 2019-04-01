#!/usr/bin/env python
# !/usr/bin/env python

import argparse
import logging
import os
import pickle

import serban.search as search
from serban.dialog_encoder_decoder import DialogEncoderDecoder
from serban.state import prototype_state

logger = logging.getLogger(__file__)


def parse_args():
    parser = argparse.ArgumentParser("Sample (with beam-search) from the session model")

    parser.add_argument("--ignore-unk",
                        action="store_false",
                        help="Allows generation procedure to output unknown words (<unk> tokens)")

    parser.add_argument("model_prefix",
                        help="Path to the model prefix (without _model.npz or _state.pkl)")

    parser.add_argument("context", help="File of input contexts")

    parser.add_argument("output", help="Output file")

    parser.add_argument("--beam_search",
                        action="store_true",
                        help="Use beam search instead of random search")

    parser.add_argument("--n-samples", default=1, type=int, help="Number of samples")

    parser.add_argument("--n-turns", default=1, type=int, help="Number of dialog turns to generate")

    parser.add_argument("--verbose", action="store_true", help="Be verbose")

    return parser.parse_args()


def main():
    args = parse_args()
    state = prototype_state()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s: %(name)s:%(lineno)d: %(levelname)s: %(message)s")

    state_path = args.model_prefix + "_state.pkl"
    model_path = args.model_prefix + "_model.npz"

    logger.info('loading state: %s', state_path)
    with open(state_path, 'rb') as src:
        state.update(pickle.load(src))

    logger.info('creating model...')
    model = DialogEncoderDecoder(state)

    logger.info('creating sampler...')
    if args.beam_search:
        logging.info('Using Beam Search')
        sampler = search.BeamSampler(model)
    else:
        logging.info('Using Random Search')
        sampler = search.RandomSampler(model)

    if os.path.isfile(model_path):
        logger.debug("Loading previous model from %s", model_path)
        model.load(model_path)
    else:
        raise ValueError("Must specify a valid model path")

    logging.info('loading context %s', args.context)
    with open(args.context) as f:
        lines = f.readlines()
    logging.info('loaded context, #lines %d', len(lines))

    if len(lines):
        contexts = [x.strip() for x in lines]
    else:
        contexts = [[]]

    logger.info('Sampling started...')
    context_samples, context_costs = sampler.sample(
        contexts,
        n_samples=args.n_samples,
        n_turns=args.n_turns,
        ignore_unk=args.ignore_unk,
        verbose=args.verbose,
    )

    logger.info('Sampling finished.')
    logger.info('Saving to file %s...', args.output)

    # Write to output file
    parent = os.path.dirname(args.output)
    if not os.path.isdir(parent):
        os.makedirs(parent)
        logger.info('Created directory: %s', parent)

    with open(args.output, "w") as output_handle:
        for context_sample in context_samples:
            print('\t'.join(context_sample), file=output_handle)

    logger.info('Saving to file finished.')
    logger.info('All done!')


if __name__ == "__main__":
    main()
