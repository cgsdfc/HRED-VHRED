from serban.config import basic


class ModelArchConfig(basic.ModelArchConfig):
    reset_utterance_decoder_at_end_of_utterance = True
    reset_utterance_encoder_at_end_of_utterance = True
    utterance_decoder_gating = 'LSTM'
    bidirectional_utterance_encoder = False
    deep_dialogue_input = True
    deep_out = True


class HiddenLayerConfig(basic.HiddenLayerConfig):
    qdim_encoder = 500
    qdim_decoder = 500
    sdim = 1000
    rankdim = 300
