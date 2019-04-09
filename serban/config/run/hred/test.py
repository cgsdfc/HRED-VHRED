from serban.config import basic
from serban.config.dataset import test
from serban.config.model import hred


class TrainingConfig(basic.TrainingConfig):
    loop_iters = 10000
    max_grad_steps = 20
    valid_freq = 50
    bs = 5
    sort_k_batches = 1
    use_nce = False


class LatentVariableConfig(basic.LatentVariableConfig):
    initialize_from_pretrained_word_embeddings = False
    fix_pretrained_word_embeddings = False


class ModelArchConfig(hred.ModelArchConfig):
    utterance_decoder_gating = 'GRU'
    bidirectional_utterance_encoder = True
    direct_connection_between_encoders_and_decoder = True


class HiddenLayerConfig(hred.HiddenLayerConfig):
    qdim_encoder = 15
    qdim_decoder = 5
    sdim = 10
    rankdim = 10


class Config(TrainingConfig,
             LatentVariableConfig,
             ModelArchConfig,
             HiddenLayerConfig,
             test.DatasetConfig,
             basic.BasicConfig):
    pass


if __name__ == '__main__':
    c = Config()

    assert c.loop_iters == 10000
    assert c.max_grad_steps == 20
    assert c.valid_freq == 50
    assert c.bs == 5
    assert c.sort_k_batches == 1
    assert c.use_nce == False
    assert c.updater == 'adam'

    assert c.initialize_from_pretrained_word_embeddings == False
    assert c.fix_pretrained_word_embeddings == False

    assert c.collapse_to_standard_rnn == False
    assert c.maxout_out == False
    assert c.deep_out == True
    assert c.deep_dialogue_input == True
    assert c.utterance_encoder_gating == 'GRU'
    assert c.dialogue_encoder_gating == 'GRU'
    assert c.utterance_decoder_gating == 'GRU'
    assert c.bidirectional_utterance_encoder == True
    assert c.direct_connection_between_encoders_and_decoder == True
    assert c.decoder_bias_type == 'all'

    assert c.qdim_encoder == 15
    assert c.qdim_decoder == 5
    assert c.sdim == 10
    assert c.rankdim == 10
