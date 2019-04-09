from serban.config import basic
from serban.config.model import hred


class ModelArchConfig(hred.ModelArchConfig):
    bidirectional_utterance_encoder = True
    reset_utterance_decoder_at_end_of_utterance = True
    reset_utterance_encoder_at_end_of_utterance = True
    utterance_decoder_gating = 'GRU'
    deep_dialogue_input = True
    deep_out = True


class HiddenLayerConfig(basic.HiddenLayerConfig):
    qdim_encoder = 500
    qdim_decoder = 500
    sdim = 1000
    rankdim = 300


class LatentVariableConfig(basic.LatentVariableConfig):
    # Latent variable configuration
    kl_divergence_annealing_rate = 1.0 / 60000.0
    initialize_from_pretrained_word_embeddings = True
    fix_pretrained_word_embeddings = True
    add_latent_gaussian_per_utterance = True
    latent_gaussian_per_utterance_dim = 100
    scale_latent_variable_variances = 0.1
    condition_latent_variable_on_dialogue_encoder = True
    train_latent_gaussians_with_kl_divergence_annealing = True
    decoder_drop_previous_input_tokens = True
    decoder_drop_previous_input_tokens_rate = 0.75
