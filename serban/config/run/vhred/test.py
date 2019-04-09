from serban.config.dataset import test
from serban.config import basic
from serban.config.model import vhred


class ModelArchConfig(vhred.ModelArchConfig):
    direct_connection_between_encoders_and_decoder = True
    deep_direct_connection = True


class LatentVariableConfig(vhred.LatentVariableConfig):
    latent_gaussian_per_utterance_dim = 5
    latent_gaussian_linear_dynamics = True


class HiddenLayerConfig(vhred.HiddenLayerConfig):
    qdim_encoder = 15
    qdim_decoder = 5
    sdim = 10
    rankdim = 10


class Config(ModelArchConfig,
             HiddenLayerConfig,
             LatentVariableConfig,
             test.TrainingConfig,
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

    # Handle pretrained word embeddings. Using this requires rankdim=10
    assert c.initialize_from_pretrained_word_embeddings == True
    assert c.fix_pretrained_word_embeddings == True

    assert c.collapse_to_standard_rnn == False

    assert c.maxout_out == False
    assert c.deep_out == True
    assert c.deep_dialogue_input == True
    assert c.direct_connection_between_encoders_and_decoder == True
    assert c.deep_direct_connection == True

    assert c.utterance_encoder_gating == 'GRU'
    assert c.dialogue_encoder_gating == 'GRU'
    assert c.utterance_decoder_gating == 'GRU'

    assert c.bidirectional_utterance_encoder == True
    assert c.add_latent_gaussian_per_utterance == True
    assert c.latent_gaussian_per_utterance_dim == 5
    assert c.condition_latent_variable_on_dialogue_encoder == True
    assert c.condition_latent_variable_on_dcgm_encoder == False
    assert c.train_latent_gaussians_with_kl_divergence_annealing == True
    assert c.kl_divergence_annealing_rate == 1.0 / 60000.0
    assert c.latent_gaussian_linear_dynamics == True

    assert c.decoder_drop_previous_input_tokens == True
    assert c.decoder_drop_previous_input_tokens_rate == 0.75
    assert c.decoder_bias_type == 'all'

    assert c.qdim_encoder == 15
    assert c.qdim_decoder == 5
    assert c.sdim == 10
    assert c.rankdim == 10

    pass
