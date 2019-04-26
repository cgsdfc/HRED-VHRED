#!/usr/bin/env python3
import subprocess
import argparse
import shutil
import json

LOG_PATH = 'LogPath'


def docker_inspect(name):
    cmd = ['docker', 'inspect', name]
    bytes = subprocess.check_output(cmd)
    return json.loads(bytes, encoding='utf-8')


def get_logfile(name, dest):
    inspect_res = docker_inspect(name)
    logpath = inspect_res[0][LOG_PATH]
    shutil.copy(logpath, dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='container name')
    parser.add_argument('dest', help='dest for the logfile')
    args = parser.parse_args()
    get_logfile(args.name, args.dest)
