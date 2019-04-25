from pathlib import Path
import subprocess
import logging
import argparse

DICT_FILENAME = 'train.dict.pkl'
CONVERTER = 'convert_text2dict.py'

HARD_CODED_DIR = '/home/cgsdfc/Serban_OpenSubData'


def convert_one_dir(dir):
    dir = Path(dir)

    def get_all_words_files(root: Path):
        return sorted(root.glob('*.txt'), key=lambda p: p.stat().st_size, reverse=True)

    def get_prefix(path: Path):
        return path.name.split('.')[0]

    def get_args():
        for subdir in dir.iterdir():
            dict_file = None
            for file in get_all_words_files(subdir):
                prefix = get_prefix(file)
                if not dict_file:
                    dict_file = subdir.joinpath(prefix + '.dict.pkl')
                    has_dict = False
                else:
                    has_dict = True
                output = subdir.joinpath(prefix)
                input = file.absolute()
                logging.info('has_dict: %s', has_dict)
                yield input, output, has_dict, dict_file

    def run_converter(input, output, has_dict, dict_file):
        args = [CONVERTER, input, output]
        if has_dict:
            args += ['--dict', dict_file]
        logging.info('convert %s to %s', input, output)
        subprocess.check_call(args)

    for args in get_args():
        run_converter(*args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', default=HARD_CODED_DIR)
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    convert_one_dir(args.dir)
