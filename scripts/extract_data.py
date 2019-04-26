from agenda import extractor
from agenda.extractor import Field

if __name__ == '__main__':
    extractor.main([
        Field('epoch', r'Epoch: (\d+)'),
        Field('acc_word_ppl', r'acc_word_perplexity = ([\d.]+)'),
        Field('cur_word_ppl', r'cur_word_perplexity = ([\d.]+)'),
    ])
