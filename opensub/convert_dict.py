import argparse


def load_dict(file):
    with open(file) as f:
        return {string: idx for idx, string in enumerate(f.read().splitlines())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='output filename for pkl dict')
    parser.add_argument('input', help='input plain text dict file')
