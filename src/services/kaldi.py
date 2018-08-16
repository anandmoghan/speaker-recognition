from subprocess import Popen, PIPE
from os.path import abspath

import re
import numpy as np

from constants.app_constants import KALDI_PATH_FILE


class Kaldi:
    def __init__(self, path_file=KALDI_PATH_FILE):
        self.command = 'source {}'.format(path_file)

    def run_command(self, cmd):
        cmd = '{} && ({})'.format(self.command, cmd)
        output, error = Popen(cmd, stdout=PIPE, shell=True).communicate()
        return output.decode("utf-8"), error.decode("utf-8") if error is not None else error


def read_feat(scp_file, n_features):
    output, _ = Kaldi().run_command('copy-feats scp:{} ark,t:'.format(scp_file))
    output = re.split('[\[]+', output)[1][1:-3]
    return np.fromstring(output, dtype=float, sep=' \n').reshape([-1, n_features]).T


def read_vector(scp_file, dtype=np.float):
    output, _ = Kaldi().run_command('copy-vector scp:{} ark,t:'.format(scp_file))
    output = re.split('[\[]+', output)[1][1:-3]
    return np.fromstring(output, dtype, sep=' ')


def scp_to_dict(scp_file):
    scp_dict = dict()
    with open(scp_file, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[\s]+', line.strip())
            scp_dict[tokens[0]] = tokens[1]
    return scp_dict


def write_vector(arr, utt_id, ark_file):
    scp_file = '{}.scp'.format(ark_file[:-4])
    arr = np.array(arr)
    Kaldi().run_command('echo [ {} ] | copy-vector - - > {}'.format(np.array_str(arr)[1:-1], ark_file))

    with open(scp_file, 'w') as f:
        f.write('{} {}\n'.format(utt_id, abspath(ark_file)))
