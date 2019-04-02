#!/usr/bin/env python
"""
This script computes dialogue embeddings for dialogues found in a text file.
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

from serban.dialog_encoder_decoder import DialogEncoderDecoder
from serban.state import prototype_state

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Compute dialogue embeddings from model")

    parser.add_argument("model_prefix", help="Path to the model prefix (without _model.npz or _state.pkl)")

    parser.add_argument("dialogues", help="File of input dialogues (tab separated)")

    parser.add_argument("output", help="Output file")

    parser.add_argument("--verbose", action="store_true", default=False, help="Be verbose")

    parser.add_argument(
        "--use-second-last-state",
        action="store_true",
        default=False,
        help="Outputs the second last dialogue encoder state instead of the last one"
    )

    return parser.parse_args()


def compute_encodings(joined_contexts, model, model_compute_encoding, output_second_last_state=False):
    # TODO Fix seq_len below
    seq_len = 600
    context = numpy.zeros((seq_len, len(joined_contexts)), dtype='int32')
    context_lengths = numpy.zeros(len(joined_contexts), dtype='int32')
    second_last_utterance_position = numpy.zeros(len(joined_contexts), dtype='int32')

    for idx in range(len(joined_contexts)):
        context_lengths[idx] = len(joined_contexts[idx])
        if context_lengths[idx] < seq_len:
            context[:context_lengths[idx], idx] = joined_contexts[idx]
        else:
            # If context is longer than max context, truncate it and force the end-of-utterance token at the end
            context[:seq_len, idx] = joined_contexts[idx][0:seq_len]
            context[seq_len - 1, idx] = model.eos_sym
            context_lengths[idx] = seq_len

        eos_indices = list(numpy.where(context[:context_lengths[idx], idx] == model.eos_sym)[0])

        if len(eos_indices) > 1:
            second_last_utterance_position[idx] = eos_indices[-2]
        else:
            second_last_utterance_position[idx] = context_lengths[idx]

    # Generate the reversed context
    reversed_context = model.reverse_utterances(context)

    encoder_states = model_compute_encoding(context, reversed_context, seq_len + 1)
    hidden_states = encoder_states[-2]  # hidden state for the "context" encoder, h_s,
    # and last hidden state of the utterance "encoder", h
    # hidden_states = encoder_states[-1] # mean for the stochastic latent variable, z

    if output_second_last_state:
        second_last_hidden_state = numpy.zeros(
            (hidden_states.shape[1], hidden_states.shape[2]),
            dtype='float64'
        )

        for i in range(hidden_states.shape[1]):
            second_last_hidden_state[i, :] = hidden_states[second_last_utterance_position[i], i, :]
        return second_last_hidden_state
    else:
        return hidden_states[-1, :, :]


def main():
    args = parse_args()
    state = prototype_state()

    state_path = args.model_prefix + "_state.pkl"
    model_path = args.model_prefix + "_model.npz"

    with open(state_path, 'rb') as src:
        state.update(pickle.load(src))

    logging.basicConfig(level=getattr(logging, state['level']),
                        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

    state['bs'] = 10

    model = DialogEncoderDecoder(state)

    if os.path.isfile(model_path):
        logger.debug("Loading previous model")
        model.load(model_path)
    else:
        raise Exception("Must specify a valid model path")

    contexts = [[]]
    with  open(args.dialogues, "rb") as f:
        lines = f.readlines()
    if len(lines):
        contexts = [x.strip() for x in lines]

    model_compute_encoding = model.build_encoder_function()
    dialogue_encodings = []

    # Start loop
    joined_contexts = []
    batch_index = 0
    batch_total = int(math.ceil(float(len(contexts)) / float(model.bs)))
    for context_id, context_sentences in enumerate(contexts):
        # Convert contexts into list of ids

        if len(context_sentences) == 0:
            joined_context = [model.eos_sym]
        else:
            joined_context = model.words_to_indices(context_sentences.split())

            if joined_context[0] != model.eos_sym:
                joined_context = [model.eos_sym] + joined_context

            if joined_context[-1] != model.eos_sym:
                joined_context += [model.eos_sym]

        # print('joined_context', joined_context)

        joined_contexts.append(joined_context)

        if len(joined_contexts) == model.bs:
            batch_index = batch_index + 1
            logger.debug("[COMPUTE] - Got batch %d / %d" % (batch_index, batch_total))
            encodings = compute_encodings(joined_contexts, model, model_compute_encoding, args.use_second_last_state)
            for i in range(len(encodings)):
                dialogue_encodings.append(encodings[i])

            joined_contexts = []

    if len(joined_contexts) > 0:
        logger.debug("[COMPUTE] - Got batch %d / %d" % (batch_total, batch_total))
        encodings = compute_encodings(joined_contexts, model, model_compute_encoding, args.use_second_last_state)
        for i in range(len(encodings)):
            dialogue_encodings.append(encodings[i])

    # Save encodings to disc
    with open(args.output + '.pkl', 'wb') as f:
        pickle.dump(dialogue_encodings, f)


if __name__ == "__main__":
    main()

    #  THEANO_FLAGS=mode=FAST_COMPILE,floatX=float32 python compute_dialogue_embeddings.py tests/models/1462302387.69_testmodel tests/data/tvalid_contexts.txt Latent_Variable_Means --verbose --use-second-last-state
