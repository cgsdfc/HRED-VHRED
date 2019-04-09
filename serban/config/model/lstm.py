from serban.config import basic


class ModelArchConfig(basic.ModelArchConfig):
    utterance_decoder_gating = 'LSTM'  # Supports 'None', 'GRU' and 'LSTM'
    bidirectional_utterance_encoder = False
    deep_dialogue_input = True
    deep_out = True
    collapse_to_standard_rnn = True
    decoder_bias_type = 'all'
    direct_connection_between_encoders_and_decoder = False
    deep_direct_connection = False
    reset_utterance_decoder_at_end_of_utterance = False
    reset_utterance_encoder_at_end_of_utterance = False


class HiddenLayerConfig(basic.HiddenLayerConfig):
    qdim_encoder = 10
    qdim_decoder = 2000
    sdim = 10
    rankdim = 300
