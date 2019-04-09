import os
from serban.config import basic

# Fill in this path!!
UBUNTU_DATA_PATH = ''


class ConstantConfig(basic.ConstantConfig):
    end_sym_utterance = '__eot__'

    unk_sym = 0  # Unknown word token <unk>
    eos_sym = 1  # end-of-utterance symbol </s>
    eod_sym = -1  # end-of-dialogue symbol </d>
    first_speaker_sym = -1  # first speaker symbol <first_speaker>
    second_speaker_sym = -1  # second speaker symbol <second_speaker>
    third_speaker_sym = -1  # third speaker symbol <third_speaker>
    minor_speaker_sym = -1  # minor speaker symbol <minor_speaker>
    voice_over_sym = -1  # voice over symbol <voice_over>
    off_screen_sym = -1  # off screen symbol <off_screen>
    pause_sym = -1  # pause symbol <pause>


class DatasetConfig(basic.DatasetConfig):
    train_dialogues = os.path.join(UBUNTU_DATA_PATH, 'Training.dialogues.pkl')
    test_dialogues = os.path.join(UBUNTU_DATA_PATH, 'Test.dialogues.pkl')
    valid_dialogues = os.path.join(UBUNTU_DATA_PATH, 'Validation.dialogues.pkl')
    dictionary = os.path.join(UBUNTU_DATA_PATH, 'Dataset.dict.pkl')
