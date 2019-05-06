"""
Merge 2 words files -- contexts and response, into one dialogues pkl file
"""

import pickle
from pathlib import Path
import logging

MERGE_CONTEXTS = Path('/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/eval/test.context.txt')
MERGE_REFERENCES = Path('/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/eval/test.response.txt')
MERGE_DICT = Path('/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/train.dict.pkl')
MERGE_OUTPUT = Path('/home/cgsdfc/SerbanOpenSubData/dialogue_length3_6/eval/test.dialogues.pkl')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    logging.info('loading vocab {}'.format(MERGE_DICT))
    vocab = pickle.loads(MERGE_DICT.read_bytes())
    vocab = {d[0]: d[1] for d in vocab}

    logging.info('loading ctx {}'.format(MERGE_CONTEXTS))
    MERGE_CONTEXTS = MERGE_CONTEXTS.read_text().splitlines()

    logging.info('loading ref {}'.format(MERGE_REFERENCES))
    MERGE_REFERENCES = MERGE_REFERENCES.read_text().splitlines()

    logging.info('writing dialouges to {}'.format(MERGE_OUTPUT))
    dialogues = zip(MERGE_CONTEXTS, MERGE_REFERENCES)
    dialogues = [' '.join(arg) for arg in dialogues]
    dialogues = [[vocab.get(word, 0) for word in line.split()] for line in dialogues]
    MERGE_OUTPUT.write_bytes(pickle.dumps(dialogues))
