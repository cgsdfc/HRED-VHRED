import re

LSDSCC_SEP = '<EOS>#TAB#'
LSDSCC_SEP_RE = re.compile(LSDSCC_SEP)
EOS = '</s>'
EOS = EOS.join('  ')
CONVERTER = 'convert_text2dict.py'


def replace_eos_on_the_whole_file(input, output):
    def replace_each_line():
        with open(input) as f:
            for line in f:
                yield LSDSCC_SEP_RE.sub(EOS, line)

    with open(output, 'w') as out:
        for line in replace_each_line():
            out.write(line)


def pickle_dataset():
    pass
