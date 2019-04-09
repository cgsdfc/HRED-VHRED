from serban.config import basic
from serban.config.dataset import ubuntu
from serban.config.model import hred


class ModelArchConfig(hred.ModelArchConfig):
    pass


class HiddenLayerConfig(hred.HiddenLayerConfig):
    pass


class TrainingConfig(basic.TrainingConfig):
    max_grad_steps = 80
    valid_freq = 5000
    bs = 20
    lr = 0.0002


class Config(ubuntu.ConstantConfig,
             ModelArchConfig,
             HiddenLayerConfig,
             TrainingConfig,
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

    assert c.max_grad_steps == 80

    assert c.valid_freq == 5000

    assert c.updater == 'adam'

    assert c.bidirectional_utterance_encoder == False
    assert c.deep_dialogue_input == True
    assert c.deep_out == True

    assert c.bs == 20

    assert c.reset_utterance_decoder_at_end_of_utterance == True
    assert c.reset_utterance_encoder_at_end_of_utterance == True
    assert c.utterance_decoder_gating == 'LSTM'

    assert c.lr == 0.0002

    assert c.qdim_encoder == 500
    assert c.qdim_decoder == 500
    assert c.sdim == 1000
    assert c.rankdim == 300
