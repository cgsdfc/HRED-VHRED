import re
import argparse
import os
import csv

EPOCH = re.compile(r'Epoch: (\d+)')
ACC_WORD_PPL = re.compile(r'acc_word_perplexity = ([\d.]+)')
CUR_WORD_PPL = re.compile(r'cur_word_perplexity = ([\d.]+)')

FIELDS = ('epoch', 'acc_word_ppl', 'cur_word_ppl')


def grep(lines, regex, func):
    matches = []
    for line in lines:
        m = regex.search(line)
        if m:
            matches.append(func(m.group(1)))
    return matches


def extract(filename):
    with open(filename) as f:
        lines = f.readlines()
    epoch = grep(lines, EPOCH, int)
    acc_ppl = grep(lines, ACC_WORD_PPL, float)
    cur_ppl = grep(lines, CUR_WORD_PPL, float)
    assert len(epoch) == len(acc_ppl) == len(cur_ppl)
    return list(zip(epoch, acc_ppl, cur_ppl))


def write_csv(data, output):
    with open(output, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for row in data:
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile')
    parser.add_argument('output')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join('.', os.path.basename(args.logfile) + '.csv')
    data = extract(args.logfile)
    write_csv(data, args.output)
