#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import argparse
import math
import os
import os.path
import pathlib
import pprint
import signal
import sys
import time
from os import listdir
from os.path import isfile, join
import logging
import gc
import pickle
import numpy
import theano

import serban.state as prototype_states
import serban.search as search
from serban.data_iterator import get_train_iterator, add_random_variables_to_batch
from serban.dialog_encoder_decoder import DialogEncoderDecoder
from serban.utils import ConvertTimedelta


class Unbuffered:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = Unbuffered(sys.stdout)
_logger = logging.getLogger(__file__)

# Unique RUN_ID for this execution
RUN_ID = str(time.time())

# Additional measures can be set here
measures = [
    "train_cost",
    "train_misclass",
    "train_kl_divergence_cost",
    "train_posterior_mean_variance",
    "valid_cost",
    "valid_misclass",
    "valid_kl_divergence_cost",
    "valid_posterior_mean_variance",
    "valid_emi"
]


def make_metrics():
    return {key: [] for key in measures}


def save(model, timings, post_fix=''):
    _logger.info("Saving the model...")

    save_dir = model.state['save_dir']
    if not save_dir.is_dir():
        save_dir.mkdir()
    _logger.info('saving to directory: %s', save_dir)

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)

    # This is the model weights.
    filename = model.state['save_dir'].joinpath(
        model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'model.npz')
    model.save(filename)
    _logger.info('saved model weights: %s', filename)

    # This is the model hparams.
    filename = model.state['save_dir'].joinpath(
        model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'state.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(model.state, f)
    _logger.info('saved model hyperparameters: %s', filename)

    # This is the model metrics.
    filename = model.state['save_dir'].joinpath(
        model.state['run_id'] + "_" + model.state['prefix'] + post_fix + 'timing.npz')
    numpy.savez(filename, **timings)
    _logger.info('saved model metrics: %s', filename)

    signal.signal(signal.SIGINT, s)
    _logger.info("Model saved, took {:.4f}".format(time.time() - start))


def load(model, filename, parameter_strings_to_ignore):
    _logger.info("Loading the model...")

    # ignore keyboard interrupt while saving
    start = time.time()
    s = signal.signal(signal.SIGINT, signal.SIG_IGN)
    model.load(filename, parameter_strings_to_ignore)
    signal.signal(signal.SIGINT, s)

    _logger.info("Model loaded, took {:.4f}".format(time.time() - start))


def main(args):
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s: %(name)s:%(lineno)d: %(levelname)s: %(message)s")

    _logger.info('loading prototype %s', args.prototype)
    state = eval(args.prototype, prototype_states.__dict__)()

    metrics_dict = make_metrics()
    valid_rounds = 0
    model, save_model_on_first_valid = auto_resume(args, metrics_dict, state)
    rng = model.rng

    _logger.debug("Compile trainer")
    if not state["use_nce"]:
        if 'add_latent_gaussian_per_utterance' in state and state["add_latent_gaussian_per_utterance"]:
            _logger.debug("Training using variational lower bound on log-likelihood")
        else:
            _logger.debug("Training using exact log-likelihood")
        train_batch = model.build_train_function()
    else:
        _logger.debug("Training with noise contrastive estimation")
        train_batch = model.build_nce_function()

    eval_batch = model.build_eval_function()

    if model.add_latent_gaussian_per_utterance:
        eval_grads = model.build_eval_grads()

    random_sampler = search.RandomSampler(model)

    _logger.debug("Loading data")
    train_data, valid_data = get_train_iterator(state)
    train_data.start()

    # Start looping through the dataset
    step = 0
    patience = state['patience']
    start_time = time.time()

    train_cost = 0
    train_kl_divergence_cost = 0
    train_posterior_mean_variance = 0
    train_misclass = 0
    train_done = 0
    train_dialogues_done = 0.0

    prev_train_cost = 0
    prev_train_done = 0
    start_validation = False

    while (step < state['loop_iters'] and
           (time.time() - start_time) / 60. < state['time_stop'] and
           patience >= 0):

        # Sampling phase
        if step % 200 == 0:
            # First generate stochastic samples
            for param in model.params:
                _logger.info("%s = %.4f" % (param.name, numpy.sum(param.get_value() ** 2) ** 0.5))
            samples, costs = random_sampler.sample([[]], n_samples=1, n_turns=3)
            _logger.info("Sampled : {}".format(samples[0]))

        # Training phase
        batch = train_data.next()

        # Train finished
        if not batch:
            # Restart training
            _logger.debug("Got None...")
            break

        _logger.debug("[TRAIN] - batch_size: %d, max_length: %d" % (batch['x'].shape[1], batch['max_length']))

        x_data = batch['x']
        x_data_reversed = batch['x_reversed']
        max_length = batch['max_length']
        x_cost_mask = batch['x_mask']
        x_reset = batch['x_reset']
        ran_cost_utterance = batch['ran_var_constutterance']
        ran_decoder_drop_mask = batch['ran_decoder_drop_mask']

        is_end_of_batch = False
        if numpy.sum(numpy.abs(x_reset)) < 1:
            # Print when we reach the end of an example (e.g. the end of a dialogue or a document)
            # Knowing when the training procedure reaches the end is useful for diagnosing training problems
            # print 'END-OF-BATCH EXAMPLE!'
            is_end_of_batch = True

        if state['use_nce']:
            y_neg = rng.choice(
                size=(10, max_length, x_data.shape[1]),
                a=model.idim,
                p=model.noise_probs
            ).astype('int32')

            c, kl_divergence_cost, posterior_mean_variance = train_batch(x_data, x_data_reversed, y_neg, max_length,
                                                                         x_cost_mask, x_reset, ran_cost_utterance,
                                                                         ran_decoder_drop_mask)
        else:
            c, kl_divergence_cost, posterior_mean_variance = train_batch(x_data, x_data_reversed, max_length,
                                                                         x_cost_mask, x_reset, ran_cost_utterance,
                                                                         ran_decoder_drop_mask)

        print_batch_statistics(c, kl_divergence_cost, model, posterior_mean_variance, x_cost_mask, x_data)

        if numpy.isinf(c) or numpy.isnan(c):
            _logger.warn("Got NaN cost .. skipping")
            gc.collect()
            continue

        train_cost += c
        train_kl_divergence_cost += kl_divergence_cost
        train_posterior_mean_variance += posterior_mean_variance

        train_done += batch['num_preds']
        train_dialogues_done += batch['num_dialogues']

        this_time = time.time()
        if step % state['train_freq'] == 0:
            # Keep track of training cost for the last 'train_freq' batches.
            current_train_cost = train_cost / train_done
            if prev_train_done >= 1 and abs(train_done - prev_train_done) > 0:
                current_train_cost = (train_cost - prev_train_cost) / (train_done - prev_train_done)

            if numpy.isinf(c) or numpy.isnan(c):
                current_train_cost = 0

            prev_train_cost = train_cost
            prev_train_done = train_done

            print_batch_summary(
                batch, current_train_cost, start_time, state, step,
                this_time, train_cost, train_dialogues_done,
                train_done, train_kl_divergence_cost, train_misclass, train_posterior_mean_variance)

        # Inspection phase
        # Evaluate gradient variance every 200 steps for GRU decoder
        if state['utterance_decoder_gating'].upper() == "GRU":
            if (step % 200 == 0) and model.add_latent_gaussian_per_utterance:
                k_eval = 10

                softmax_costs = numpy.zeros((k_eval), dtype='float32')
                var_costs = numpy.zeros((k_eval), dtype='float32')
                gradients_wrt_softmax = numpy.zeros((k_eval, model.qdim_decoder, model.qdim_decoder), dtype='float32')
                for k in range(0, k_eval):
                    batch = add_random_variables_to_batch(model.state, model.rng, batch, None, False)
                    ran_cost_utterance = batch['ran_var_constutterance']
                    ran_decoder_drop_mask = batch['ran_decoder_drop_mask']
                    softmax_cost, var_cost, grads_wrt_softmax, grads_wrt_kl_divergence_cost = eval_grads(x_data,
                                                                                                         x_data_reversed,
                                                                                                         max_length,
                                                                                                         x_cost_mask,
                                                                                                         x_reset,
                                                                                                         ran_cost_utterance,
                                                                                                         ran_decoder_drop_mask)
                    softmax_costs[k] = softmax_cost
                    var_costs[k] = var_cost
                    gradients_wrt_softmax[k, :, :] = grads_wrt_softmax

                print_inference(gradients_wrt_softmax, model, softmax_costs, state, var_costs)

        # Evaluation phase
        if valid_data is not None and step % state['valid_freq'] == 0 and step > 1:
            start_validation = True

        # Only start validation loop once it's time to validate and once all previous batches have been reset
        if start_validation and is_end_of_batch:
            start_validation = False
            valid_data.start()
            valid_cost = 0
            valid_kl_divergence_cost = 0
            valid_posterior_mean_variance = 0

            valid_word_preds_done = 0
            valid_dialogues_done = 0

            _logger.debug("[VALIDATION START]")

            while True:
                batch = valid_data.next()

                # Validation finished
                if not batch:
                    break

                _logger.debug("[VALID] - Got batch %d,%d" % (batch['x'].shape[1], batch['max_length']))

                x_data = batch['x']
                x_data_reversed = batch['x_reversed']
                max_length = batch['max_length']
                x_cost_mask = batch['x_mask']

                x_reset = batch['x_reset']
                ran_cost_utterance = batch['ran_var_constutterance']
                ran_decoder_drop_mask = batch['ran_decoder_drop_mask']

                c, kl_term, c_list, kl_term_list, posterior_mean_variance = eval_batch(x_data, x_data_reversed,
                                                                                       max_length, x_cost_mask, x_reset,
                                                                                       ran_cost_utterance,
                                                                                       ran_decoder_drop_mask)

                # Reshape into matrix, where rows are validation samples and columns are tokens
                # Note that we use max_length-1 because we don't get a cost for the first token
                # (the first token is always assumed to be eos)
                c_list = c_list.reshape((batch['x'].shape[1], max_length - 1), order=(1, 0))
                c_list = numpy.sum(c_list, axis=1)

                words_in_dialogues = numpy.sum(x_cost_mask, axis=0)
                c_list = c_list / words_in_dialogues

                if numpy.isinf(c) or numpy.isnan(c):
                    continue

                valid_cost += c
                valid_kl_divergence_cost += kl_divergence_cost
                valid_posterior_mean_variance += posterior_mean_variance

                # Print batch statistics
                print('valid_cost', valid_cost)
                print('valid_kl_divergence_cost sample', kl_divergence_cost)
                print('posterior_mean_variance', posterior_mean_variance)

                valid_word_preds_done += batch['num_preds']
                valid_dialogues_done += batch['num_dialogues']

            _logger.debug("[VALIDATION END]")

            valid_cost /= valid_word_preds_done
            valid_kl_divergence_cost /= valid_word_preds_done
            valid_posterior_mean_variance /= valid_dialogues_done

            if (len(metrics_dict["valid_cost"]) == 0) \
                    or (valid_cost < numpy.min(metrics_dict["valid_cost"])) \
                    or (save_model_on_first_valid and valid_rounds == 0):
                patience = state['patience']

                # Save model if there is  decrease in validation cost
                save(model, metrics_dict)
                print('best valid_cost', valid_cost)
            elif valid_cost >= metrics_dict["valid_cost"][-1] * state['cost_threshold']:
                patience -= 1

            if args.save_every_valid_iteration:
                save(model, metrics_dict, '_' + str(step) + '_')
            if args.auto_restart:
                save(model, metrics_dict, '_auto_')

            print_final_summary(patience, valid_cost, valid_kl_divergence_cost, valid_posterior_mean_variance)

            metrics_dict["train_cost"].append(train_cost / train_done)
            metrics_dict["train_kl_divergence_cost"].append(train_kl_divergence_cost / train_done)
            metrics_dict["train_posterior_mean_variance"].append(train_posterior_mean_variance / train_dialogues_done)
            metrics_dict["valid_cost"].append(valid_cost)
            metrics_dict["valid_kl_divergence_cost"].append(valid_kl_divergence_cost)
            metrics_dict["valid_posterior_mean_variance"].append(valid_posterior_mean_variance)

            # Reset train cost, train misclass and train done metrics
            train_cost = 0
            train_done = 0
            prev_train_cost = 0
            prev_train_done = 0

            # Count number of validation rounds done so far
            valid_rounds += 1

        step += 1

    _logger.debug("All done, exiting...")


def print_inference(gradients_wrt_softmax, model, softmax_costs, state, var_costs):
    _logger.info('mean softmax_costs: %f', numpy.mean(softmax_costs))
    _logger.info('std softmax_costs: %f', numpy.std(softmax_costs))
    _logger.info('mean var_costs: %f', numpy.mean(var_costs))
    _logger.info('std var_costs: %f', numpy.std(var_costs))

    _logger.info(
        'mean gradients_wrt_softmax: %f',
        numpy.mean(numpy.abs(numpy.mean(gradients_wrt_softmax, axis=0))),
        numpy.mean(gradients_wrt_softmax, axis=0)
    )

    _logger.info(
        'std gradients_wrt_softmax: %f',
        numpy.mean(numpy.std(gradients_wrt_softmax, axis=0)),
        numpy.std(gradients_wrt_softmax, axis=0)
    )

    _logger.info(
        'std greater than mean: %f',
        numpy.where(
            numpy.std(
                gradients_wrt_softmax, axis=0) > numpy.abs(
                numpy.mean(gradients_wrt_softmax, axis=0)
            )
        )[0].shape[0]
    )

    Wd_s_q = model.utterance_decoder.Wd_s_q.get_value()

    _logger.info('Wd_s_q all: %f', numpy.sum(numpy.abs(Wd_s_q)), numpy.mean(numpy.abs(Wd_s_q)))

    _logger.info(
        'Wd_s_q latent: %f, %f',
        numpy.sum(
            numpy.abs(
                Wd_s_q[(Wd_s_q.shape[0] - state['latent_gaussian_per_utterance_dim']):Wd_s_q.shape[0], :]
            )
        ),
        numpy.mean(
            numpy.abs(
                Wd_s_q[(Wd_s_q.shape[0] - state['latent_gaussian_per_utterance_dim']):Wd_s_q.shape[0], :]
            )
        )
    )

    _logger.info(
        'Wd_s_q ratio: %f',
        numpy.sum(
            numpy.abs(
                Wd_s_q[(Wd_s_q.shape[0] - state['latent_gaussian_per_utterance_dim']):Wd_s_q.shape[0], :]
            )
        ) / numpy.sum(numpy.abs(Wd_s_q))
    )

    if 'latent_gaussian_linear_dynamics' in state:
        if state['latent_gaussian_linear_dynamics']:
            prior_Wl_linear_dynamics = \
                model.latent_utterance_variable_prior_encoder.Wl_linear_dynamics.get_value()

            _logger.info(
                'prior_Wl_linear_dynamics: %f, %f, %f',
                numpy.sum(numpy.abs(prior_Wl_linear_dynamics)),
                numpy.mean(numpy.abs(prior_Wl_linear_dynamics)),
                numpy.std(numpy.abs(prior_Wl_linear_dynamics))
            )

            approx_posterior_Wl_linear_dynamics = \
                model.latent_utterance_variable_approx_posterior_encoder.Wl_linear_dynamics.get_value()

            _logger.info(
                'approx_posterior_Wl_linear_dynamics: %f, %f, %f',
                numpy.sum(numpy.abs(approx_posterior_Wl_linear_dynamics)),
                numpy.mean(numpy.abs(approx_posterior_Wl_linear_dynamics)),
                numpy.std(numpy.abs(approx_posterior_Wl_linear_dynamics))
            )


def print_batch_statistics(c, kl_divergence_cost,
                           model, posterior_mean_variance,
                           x_cost_mask, x_data):
    # Print batch statistics
    _logger.info('cost_sum = %f', c)
    _logger.info('cost_mean = %f', c / (numpy.sum(x_cost_mask)))
    _logger.info('kl_divergence_cost_sum = %f', kl_divergence_cost)
    _logger.info('kl_divergence_cost_mean = %f',
                 kl_divergence_cost / len(numpy.where(x_data == model.eos_sym)[0]))
    _logger.info('posterior_mean_variance = %f', posterior_mean_variance)


def print_batch_summary(batch, current_train_cost,
                        start_time, state, step, this_time,
                        train_cost, train_dialogues_done,
                        train_done, train_kl_divergence_cost,
                        train_misclass, train_posterior_mean_variance):
    h, m, s = ConvertTimedelta(this_time - start_time)
    # We need to catch exceptions due to high numbers in exp
    try:
        _logger.info('Epoch: %d', step)
        _logger.info("SpentTime: %.2d:%.2d:%.2d", h, m, s)
        _logger.info('Hours Remained: %4d', state['time_stop'] - (time.time() - start_time) / 60.0)
        _logger.info("batch_size: %d", batch['x'].shape[1])
        _logger.info('max_length %d', batch['max_length'])

        _logger.info('acc_cost = %.4f', (train_cost / train_done))
        _logger.info("acc_word_perplexity = %.4f ", math.exp(train_cost / train_done))

        _logger.info('cur_cost = %.4f', current_train_cost)
        _logger.info('cur_word_perplexity = %.4f', math.exp(current_train_cost))

        _logger.info("acc_mean_word_error = %.4f ", train_misclass / train_done)
        _logger.info("acc_mean_kl_divergence_cost = %.8f", train_kl_divergence_cost / train_done)
        _logger.info("acc_mean_posterior_variance = %.8f", train_posterior_mean_variance / train_dialogues_done)
    except Exception:
        pass


def print_final_summary(patience, valid_cost, valid_kl_divergence_cost, valid_posterior_mean_variance):
    # We need to catch exceptions due to high numbers in exp
    try:
        _logger.info("valid cost (NLL) = %.4f", valid_cost)
        _logger.info("valid word-perplexity = %.4f", math.exp(valid_cost))
        _logger.info("valid kldiv cost (per word) = %.8f", valid_kl_divergence_cost)
        _logger.info("valid mean posterior variance (per word) = %.8f", valid_posterior_mean_variance)
        _logger.info("patience = %d", patience)
    except Exception:
        try:
            _logger.info("valid cost (NLL) = %.4f, patience = %d", valid_cost, patience)
        except Exception:
            pass


def auto_resume(args, metrics_dict, state):
    if args.auto_restart:
        assert not args.save_every_valid_iteration
        assert len(args.resume) == 0

        directory = state['save_dir']
        if not directory[-1] == '/':
            directory = directory + '/'

        auto_resume_postfix = state['prefix'] + '_auto_model.npz'

        if os.path.exists(directory):
            directory_files = [f for f in listdir(directory) if isfile(join(directory, f))]
            resume_filename = ''
            for f in directory_files:
                if len(f) > len(auto_resume_postfix):
                    if f[len(f) - len(auto_resume_postfix):len(f)] == auto_resume_postfix:
                        if len(resume_filename) > 0:
                            raise ValueError('ERROR: FOUND MULTIPLE MODELS IN DIRECTORY:', directory)
                        else:
                            resume_filename = directory + f[0:len(f) - len('__auto_model.npz')]
            if len(resume_filename) > 0:
                _logger.debug("Found model to automatically resume: %s" % resume_filename)
                # Setup training to automatically resume training with the model found
                args.resume = resume_filename + '__auto'
                # Disable training from reinitialization any parameters
                args.reinitialize_decoder_parameters = False
                args.reinitialize_latent_variable_parameters = False
            else:
                _logger.debug("Could not find any model to automatically resume...")

    if args.resume != "":
        _logger.debug("Resuming %s" % args.resume)

        state_file = args.resume + '_state.pkl'
        metrics_file = args.resume + '_timing.npz'

        if os.path.isfile(state_file) and os.path.isfile(metrics_file):
            _logger.debug("Loading previous state")

            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            with open(metrics_file, 'rb') as f:
                metrics_dict = dict(numpy.load(f))
            for x, y in metrics_dict.items():
                metrics_dict[x] = list(y)
            # Increment seed to make sure we get newly shuffled batches when training on large datasets
            state['seed'] = state['seed'] + 10
        else:
            raise ValueError("Cannot resume, cannot find files!")

    _logger.debug("State:\n{}".format(pprint.pformat(state)))
    _logger.debug("Metrics:\n{}".format(pprint.pformat(metrics_dict)))

    if args.force_train_all_wordemb:
        state['fix_pretrained_word_embeddings'] = False
    _logger.info('Creating model...')
    model = DialogEncoderDecoder(state)
    save_model_on_first_valid = False

    if args.resume != "":
        filename = args.resume + '_model.npz'
        if os.path.isfile(filename):
            _logger.debug("Loading previous model")

            parameter_strings_to_ignore = []
            if args.reinitialize_decoder_parameters:
                parameter_strings_to_ignore.append('Wd_')
                parameter_strings_to_ignore += ['bd_']
                save_model_on_first_valid = True
            if args.reinitialize_latent_variable_parameters:
                parameter_strings_to_ignore += ['latent_utterance_prior']
                parameter_strings_to_ignore += ['latent_utterance_approx_posterior']
                parameter_strings_to_ignore += ['kl_divergence_cost_weight']
                parameter_strings_to_ignore += ['latent_dcgm_encoder']
                save_model_on_first_valid = True
            load(model, filename, parameter_strings_to_ignore)
        else:
            raise ValueError("Cannot resume, cannot find model file!")

        if 'run_id' not in model.state:
            raise ValueError('Backward compatibility not ensured! (need run_id in state)')
    else:
        # assign new run_id key
        model.state['run_id'] = RUN_ID
    return model, save_model_on_first_valid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="", help="Resume training from that state")

    parser.add_argument("--force_train_all_wordemb", action='store_true',
                        help="If true, will force the model to train all word embeddings in the encoder."
                             " This switch can be used to fine-tune a model which was "
                             "trained with fixed (pretrained) encoder word embeddings.")

    parser.add_argument("--save_every_valid_iteration", action='store_true',
                        help="If true, will save a unique copy of the model at every validation round.")

    parser.add_argument("--auto_restart", action='store_true',
                        help="If true, will maintain a copy of the current model parameters updated at "
                             "every validation round.Upon initialization, the script will automatically scan the "
                             " output directory and and resume training of a previous model(if such exists). This "
                             " option is meant to be used for training models on clusters with hard wall-times. This "
                             "option is incompatible with the \"resume\" and \"save_every_valid_iteration\" options.")

    # the old default, prototype_state, does not provide a dictionary and this trigger mysterious errors when
    # a prototype is not given. Thus I make it a positional argument -- required.
    # In fact, they said in the help -- (must be specified) ;-).
    # I just express their meaning better.
    parser.add_argument("prototype", type=str, help="Prototype to use (must be specified)")

    parser.add_argument("--reinitialize-latent-variable-parameters", action='store_true',
                        help="Can be used when resuming a model."
                             " If true, will initialize all latent variable parameters randomly instead of "
                             "loading them from previous model.")

    parser.add_argument("--reinitialize-decoder-parameters", action='store_true',
                        help="Can be used when resuming a model. "
                             "If true, will initialize all parameters of the utterance decoder "
                             "randomly instead of loading them from previous model.")

    return parser.parse_args()


if __name__ == "__main__":
    # Models only run with float32
    assert theano.config.floatX == 'float32'
    args = parse_args()
    main(args)
