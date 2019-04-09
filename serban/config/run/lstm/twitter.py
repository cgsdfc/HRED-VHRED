from serban.config.model import lstm
from serban.config.dataset import twitter
from serban.config import basic


class ModelArchConfig(lstm.ModelArchConfig, twitter.ModelArchConfig):
    pass


class TrainingConfig(twitter.TrainingConfig):
    pass


class HiddenLayerConfig(lstm.HiddenLayerConfig):
    rankdim = 400


class Config(ModelArchConfig,
             TrainingConfig,
             HiddenLayerConfig,
             basic.BasicConfig):
    pass


if __name__ == '__main__':
    c = Config()
    assert c.max_grad_steps == 80

    assert c.valid_freq == 5000

    assert c.updater == 'adam'

    assert c.deep_dialogue_input == True
    assert c.deep_out == True

    assert c.collapse_to_standard_rnn == True

    assert c.bs == 20
    assert c.decoder_bias_type == 'all'
    assert c.direct_connection_between_encoders_and_decoder == False
    assert c.deep_direct_connection == False

    assert c.reset_utterance_decoder_at_end_of_utterance == False
    assert c.reset_utterance_encoder_at_end_of_utterance == False
    assert c.lr == 0.0001

    assert c.qdim_encoder == 10
    assert c.qdim_decoder == 2000
    assert c.sdim == 10
    assert c.rankdim == 400
