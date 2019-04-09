import os
from serban.config import basic

# Fill in this path!!
TWITTER_DATA_PATH = ''


class DatasetConfig(basic.DatasetConfig):
    train_dialogues = "../TwitterData/Training.dialogues.pkl"
    test_dialogues = "../TwitterData/Test.dialogues.pkl"
    valid_dialogues = "../TwitterData/Validation.dialogues.pkl"
    dictionary = "../TwitterData/Dataset.dict.pkl"


class ModelArchConfig(basic.ModelArchConfig):
    bidirectional_utterance_encoder = True
    decoder_bias_type = 'selective'
    utterance_decoder_gating = 'GRU'


class TrainingConfig(basic.TrainingConfig):
    lr = 0.0001


class HiddenLayerConfig(basic.HiddenLayerConfig):
    qdim_encoder = 1000
    qdim_decoder = 1000
    rankdim = 400
