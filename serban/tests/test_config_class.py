import unittest
import logging
from pprint import pprint

from serban.config import config_to_state
import serban.state as prototype_system
import serban.config.run as config_system


# prototype_to_config = {
#     'prototype_test_HRED': 'HRED_Test_Config',
#     'prototype_test_VHRED': 'VHRED_Test_Config',
#     'prototype_twitter_lstm': 'LSTM_Twitter_Config',
#     '': '',
#     '': '',
#     '': '',
#     '': '',
#     '': '',
# }

# def make_proto_state(prototype):
#     state = pr


def prototype_to_config(name: str):
    _, dataset, model = name.split('_', maxsplit=2)
    dataset = dataset.capitalize()
    model = model.upper()
    return '_'.join([model, dataset, 'Config'])


def find_all_tests():
    def find_all_pairs():
        for name in dir(prototype_system):
            if name.startswith('prototype') and name.count('_') >= 2:
                config_name = prototype_to_config(name)
                try:
                    yield (getattr(prototype_system, name),
                           getattr(config_system, config_name))
                except AttributeError:
                    logging.info('%s has no config counterpart', name)

    return list(find_all_pairs())


def filter_prototype_state(state):
    # Remove hard-code paths.
    state.pop('prefix', None)
    state.pop('save_dir', None)
    state.pop('train_dialogues', None)
    state.pop('test_dialogues', None)
    state.pop('valid_dialogues', None)
    state.pop('dictionary', None)
    return state


class TestConfigClass(unittest.TestCase):
    maxDiff = None

    def test(self):
        all_pairs = find_all_tests()
        for prototype, config_cls in all_pairs:
            config_state = filter_prototype_state(config_to_state(config_cls))
            proto_state = filter_prototype_state(prototype())

            self.assertEqual(config_state, proto_state, msg="""
            prototype: %s
            config: %s
            """ % (prototype, config_cls))
