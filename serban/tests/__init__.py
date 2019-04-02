import os

TEST_ROOT = os.path.dirname(__file__)

DATA_ROOT = os.path.join(TEST_ROOT, 'data')
MODEL_ROOT = os.path.join(TEST_ROOT, 'model')
OUTPUT_ROOT = os.path.join(TEST_ROOT, 'output')

EMBEDDINGS_FILE = os.path.join(DATA_ROOT, 'MT_WordEmb.pkl')
DICTIONARY_FILE = os.path.join(DATA_ROOT, 'ttrain.dict.pkl')
TEST_DIALOGS_FILE = os.path.join(DATA_ROOT, 'ttest.dialogues.pkl')
VALID_DIALOGS_FILE = os.path.join(DATA_ROOT, 'tvalid.dialogues.pkl')
TRAIN_DIALOGS_FILE = os.path.join(DATA_ROOT, 'ttrain.dialogues.pkl')
