from serban.config.dataset import twitter
from serban.config.model import hred
from serban.config import basic


class ModelArchConfig(twitter.ModelArchConfig, hred.ModelArchConfig):
    reset_utterance_encoder_at_end_of_utterance = False


class TrainingConfig(twitter.TrainingConfig):
    pass


class HiddenLayerConfig(twitter.HiddenLayerConfig, hred.HiddenLayerConfig):
    pass


class Config(ModelArchConfig,
             HiddenLayerConfig,
             TrainingConfig,
             twitter.DatasetConfig,
             basic.BasicConfig):
    pass


if __name__ == '__main__':
    c = Config()
    assert c.max_grad_steps == 80
    assert c.valid_freq == 5000
    assert c.updater == 'adam'
    assert c.bidirectional_utterance_encoder

    assert c.deep_dialogue_input
    assert c.deep_out
    assert c.bs == 20
    assert c.decoder_bias_type == 'selective'

    assert not c.direct_connection_between_encoders_and_decoder
    assert not c.deep_direct_connection

    assert c.reset_utterance_decoder_at_end_of_utterance
    assert not c.reset_utterance_encoder_at_end_of_utterance

    assert c.lr == 0.0001

    assert c.qdim_encoder == 1000
    assert c.qdim_decoder == 1000
    assert c.sdim == 1000
    assert c.rankdim == 400
    assert c.utterance_decoder_gating == 'GRU'
    assert c.pretrained_word_embeddings_file is None
