from serban.config.model import vhred
from serban.config.dataset import twitter
from serban.config import basic


class ModelArchConfig(vhred.ModelArchConfig, twitter.ModelArchConfig):
    reset_utterance_encoder_at_end_of_utterance = False
    initialize_from_pretrained_word_embeddings = False
    fix_pretrained_word_embeddings = False


class Config(ModelArchConfig,
             twitter.TrainingConfig,
             twitter.HiddenLayerConfig,
             vhred.LatentVariableConfig,
             twitter.DatasetConfig,
             basic.BasicConfig):
    pass


if __name__ == '__main__':
    c = Config()
    assert c.max_grad_steps == 80

    assert c.valid_freq == 5000

    assert c.bidirectional_utterance_encoder == True

    assert c.deep_dialogue_input == True
    assert c.deep_out == True

    assert c.bs == 20
    assert c.decoder_bias_type == 'selective'  # Choose between 'first', 'all' and 'selective'
    assert c.direct_connection_between_encoders_and_decoder == False
    assert c.deep_direct_connection == False

    assert c.reset_utterance_decoder_at_end_of_utterance == True
    assert c.reset_utterance_encoder_at_end_of_utterance == False
    assert c.lr == 0.0001

    assert c.qdim_encoder == 1000
    assert c.qdim_decoder == 1000
    assert c.sdim == 1000
    assert c.rankdim == 400

    assert c.utterance_decoder_gating == 'GRU'

    assert c.add_latent_gaussian_per_utterance == True
    assert c.latent_gaussian_per_utterance_dim == 100

    assert c.scale_latent_variable_variances == 0.1
    assert c.condition_latent_variable_on_dialogue_encoder == True
    assert c.train_latent_gaussians_with_kl_divergence_annealing == True
    assert c.kl_divergence_annealing_rate == 1.0 / 60000.0
    assert c.decoder_drop_previous_input_tokens == True
    assert c.decoder_drop_previous_input_tokens_rate == 0.75

    assert c.patience == 20
