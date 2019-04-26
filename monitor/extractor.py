import os
import csv
import subprocess
import argparse
import json
import logging
import re
from typing import List

LOG_PATH = 'LogPath'


class Field:
    def __init__(self, name, regex, func=None):
        self.name = name
        if isinstance(regex, str):
            regex = re.compile(regex)
        self.regex = regex
        self.func = func or float

    def __repr__(self):
        return '<Field %s>' % self.name

    def match(self, line):
        match = self.regex.search(line)
        if match:
            return self.func(match.group(1))


class Extractor:
    def __init__(self, logfile, fields: List[Field]):
        self.logfile = logfile
        self.fields = fields

    def write_csv(self, output):
        field_names = [f.name for f in self.fields]
        data = self.extract_data()
        with open(output, 'w') as out:
            writer = csv.DictWriter(out, field_names)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    def _extract_data(self):
        data = {f.name: [] for f in self.fields}
        with open(self.logfile) as f:
            for line in f:
                for field in self.fields:
                    res = field.match(line)
                    if res:
                        data[field.name].append(res)
                        break
        return data

    def extract_data(self):
        data = self._extract_data()
        value_tuples = zip(*[data[name] for name in data.keys()])
        return [dict(zip(data.keys(), value_tuple)) for value_tuple in value_tuples]


def docker_inspect(name):
    cmd = ['docker', 'inspect', name]
    bytes = subprocess.check_output(cmd)
    return json.loads(bytes.decode(), encoding='utf-8')


def get_logfile(name):
    inspect_res = docker_inspect(name)
    logpath = inspect_res[0][LOG_PATH]
    return logpath


def gen_csv_from_docker_logfile(container_name, fields, output=None):
    if output is None:
        output = os.path.join('.', container_name + '.csv')
    logging.info('output: %s', output)
    logfile = get_logfile(container_name)
    logging.info('extract from logfile: %s', logfile)
    extractor = Extractor(logfile, fields)
    extractor.write_csv(output)


def main(fields):
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('-o', '--output')
    parser.add_argument('-v', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.v else logging.ERROR)
    gen_csv_from_docker_logfile(args.name, fields, args.output)
