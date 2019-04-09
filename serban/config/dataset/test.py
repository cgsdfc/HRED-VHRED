import serban.tests as T
from serban.config import basic


class DatasetConfig(basic.DatasetConfig):
    train_dialogues = T.TRAIN_DIALOGS_FILE
    test_dialogues = T.TEST_DIALOGS_FILE
    valid_dialogues = T.VALID_DIALOGS_FILE
    dictionary = T.DICTIONARY_FILE
    pretrained_word_embeddings_file = T.EMBEDDINGS_FILE


class TrainingConfig(basic.TrainingConfig):
    loop_iters = 10000
    max_grad_steps = 20
    valid_freq = 50
    bs = 5
    sort_k_batches = 1
    use_nce = False
