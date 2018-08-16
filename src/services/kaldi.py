from subprocess import Popen, PIPE
from os.path import abspath, join as join_path

import numpy as np
import re

from constants.app_constants import KALDI_PATH_FILE, EMB_DIR, PLDA_DIR, EMB_SCP_FILE, SPK_UTT_FILE, UTT_SPK_FILE
from services.common import run_parallel, load_array


class Kaldi:
    def __init__(self, path_file=KALDI_PATH_FILE):
        self.command = 'source {}'.format(path_file)

    def run_command(self, cmd, decode=True):
        cmd = '{} && ({})'.format(self.command, cmd)
        output, _ = Popen(cmd, stdout=PIPE, shell=True).communicate()
        if decode:
            return output.decode("utf-8")
        return output


class PLDA:
    def __init__(self, total_covariance_factor=0.0, save_loc='../save'):
        self.tcf = total_covariance_factor
        self.save_loc = save_loc
        self.embedding_loc = join_path(save_loc, EMB_DIR)
        self.plda_loc = join_path(save_loc, PLDA_DIR)

    def fit(self, index_list, speaker_list, n_jobs=10):
        spk_utt_file = join_path(self.save_loc, SPK_UTT_FILE)
        utt_spk_file = join_path(self.save_loc, UTT_SPK_FILE)
        make_spk_to_utt(index_list, speaker_list, spk_utt_file)
        make_utt_to_spk(index_list, speaker_list, utt_spk_file)

        print('Converting python arrays to kaldi ark...')
        scp_list = run_parallel(self.make_kaldi_ark, index_list, n_jobs, p_bar=True)
        embedding_scp = join_path(self.save_loc, EMB_SCP_FILE)
        with open(embedding_scp, 'w') as f:
            for scp in scp_list:
                f.write(scp)

    def make_kaldi_ark(self, key):
        embedding = load_array(join_path(self.embedding_loc, '{}.npy'.format(key)))
        ark_file = join_path(self.embedding_loc, '{}.ark'.format(key))
        return write_vector(embedding, key, ark_file)


def append_vector(arr, utt_id, ark_file, offset=None):
    arr = np.array(arr)
    output = Kaldi().run_command('echo [ {} ] | copy-vector - -'.format(np.array_str(arr)[1:-1]), decode=False)

    if offset is None:
        with open(ark_file, 'rb') as f:
            offset = len(f.read())

    content = bytes('{} '.format(utt_id).encode('utf-8'))
    offset = offset + len(content)
    content = content + output

    with open(ark_file, 'ab') as f:
        f.write(content)

    return '{} {}:{}\n'.format(utt_id, abspath(ark_file), offset), offset


def make_spk_to_utt(index_list, speaker_list, spk_utt_file):
    with open(spk_utt_file, 'w') as f:
        for u, s in zip(index_list, speaker_list):
            f.write('{} {}\n'.format(s, u))


def make_utt_to_spk(index_list, speaker_list, utt_spk_file):
    with open(utt_spk_file, 'w') as f:
        for u, s in zip(index_list, speaker_list):
            f.write('{} {}\n'.format(u, s))


def read_feat(scp_file, n_features):
    output = Kaldi().run_command('copy-feats scp:{} ark,t:'.format(scp_file))
    output = re.split('\[', output)[1][1:-2]
    return np.fromstring(output, dtype=float, sep=' \n').reshape([-1, n_features]).T


def read_vectors(scp_file, dtype=np.float):
    vectors = Kaldi().run_command('copy-vector scp:{} ark,t:'.format(scp_file))
    vectors = re.split('\n', vectors)[:-1]
    utt_list = []
    vector_list = []
    for vector in vectors:
        vector = re.split('\[', vector)
        utt_id = vector[0][:-1]
        vector = vector[1][1:-2]
        utt_list.append(utt_id)
        vector_list.append(np.fromstring(vector, dtype, sep=' '))
    return (utt_list, vector_list) if len(vector_list) > 1 else (utt_list[0], vector_list[0])


def scp_to_dict(scp_file):
    scp_dict = dict()
    with open(scp_file, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[\s]+', line.strip())
            scp_dict[tokens[0]] = tokens[1]
    return scp_dict


def write_vector(arr, utt_id, ark_file):
    arr = np.array(arr)
    output = Kaldi().run_command('echo [ {} ] | copy-vector - -'.format(np.array_str(arr)[1:-1]), decode=False)
    content = bytes('{} '.format(utt_id).encode('utf-8'))
    offset = len(content)
    content = content + output
    with open(ark_file, 'wb') as f:
            f.write(content)
    return '{} {}:{}\n'.format(utt_id, abspath(ark_file), offset), offset
