from serban.config import basic
from serban.config.dataset import ubuntu
from serban.config.model import lstm


class ModelArchConfig(lstm.ModelArchConfig):
    pass


class HiddenLayerConfig(lstm.HiddenLayerConfig):
    pass


class TrainingConfig(basic.TrainingConfig):
    pass


class Config(ubuntu.ConstantConfig,
             ModelArchConfig,
             HiddenLayerConfig,
             TrainingConfig,
             ubuntu.DatasetConfig,
             basic.BasicConfig):
    pass


if __name__ == '__main__':
    c = Config()
    assert c.end_sym_utterance == '__eot__'

    assert c.unk_sym == 0  # Unknown word token <unk>
    assert c.eos_sym == 1  # end-of-utterance symbol </s>
    assert c.eod_sym == -1  # end-of-dialogue symbol </d>
    assert c.first_speaker_sym == -1  # first speaker symbol <first_speaker>
    assert c.second_speaker_sym == -1  # second speaker symbol <second_speaker>
    assert c.third_speaker_sym == -1  # third speaker symbol <third_speaker>
    assert c.minor_speaker_sym == -1  # minor speaker symbol <minor_speaker>
    assert c.voice_over_sym == -1  # voice over symbol <voice_over>
    assert c.off_screen_sym == -1  # off screen symbol <off_screen>
    assert c.pause_sym == -1  # pause symbol <pause>

    # train
    assert c.max_grad_steps == 80
    assert c.valid_freq == 5000
    assert c.updater == 'adam'
    assert c.bs == 20
    assert c.lr == 0.0002

    # model arch
    assert c.utterance_decoder_gating == 'LSTM'  # Supports 'None', 'GRU' and 'LSTM'
    assert c.bidirectional_utterance_encoder == False
    assert c.deep_dialogue_input == True
    assert c.deep_out == True
    assert c.collapse_to_standard_rnn == True
    assert c.decoder_bias_type == 'all'
    assert c.direct_connection_between_encoders_and_decoder == False
    assert c.deep_direct_connection == False
    assert c.reset_utterance_decoder_at_end_of_utterance == False
    assert c.reset_utterance_encoder_at_end_of_utterance == False

    assert c.qdim_encoder == 10
    assert c.qdim_decoder == 2000
    assert c.sdim == 10
    assert c.rankdim == 300