import re
from os import remove as remove_file
from os.path import exists, join as join_path

import numpy as np

from constants.app_constants import DATA_SCP_FILE, MFCC_DIR, VAD_DIR
from services.common import load_array, run_parallel, save_array
from services.kaldi import read_feat, read_vector, scp_to_dict, Kaldi


class MFCC:
    def __init__(self, fs=8000, fl=100, fh=4000, frame_len_ms=25, n_jobs=20, n_ceps=20, save_loc='../save'):
        mfcc_loc = join_path(save_loc, MFCC_DIR)
        params_file = join_path(mfcc_loc, 'mfcc.params')
        config_file = join_path(mfcc_loc, 'mfcc.conf')

        with open(params_file, 'w') as f:
            f.write('nj={}\n'.format(n_jobs))
            f.write('compress={}\n'.format('true'))
            f.write('mfcc_loc={}\n'.format(mfcc_loc))
            f.write('mfcc_config={}\n'.format(config_file))

        with open(config_file, 'w') as f:
            f.write('--sample-frequency={}\n'.format(fs))
            f.write('--low-freq={}\n'.format(fl))
            f.write('--high-freq={}\n'.format(fh))
            f.write('--frame-length={}\n'.format(frame_len_ms))
            f.write('--num-ceps={}\n'.format(n_ceps))
            f.write('--snip-edges={}\n'.format('false'))

        self.mfcc_loc = mfcc_loc
        self.params_file = params_file
        self.n_ceps = n_ceps
        self.n_jobs = n_jobs

    def apply_vad_and_save(self, feats_scp, vad_scp):
        feats_dict = scp_to_dict(feats_scp)
        vad_dict = scp_to_dict(vad_scp)
        index_list = []
        feature_list = []
        vad_list = []
        save_list = []
        scp_list = []
        for key in feats_dict.keys():
            try:
                vad_list.append(vad_dict[key])
                index_list.append(key)
                feature_list.append(feats_dict[key])
                scp_list.append('{}/{}.scp'.format(self.mfcc_loc, key))
                save_list.append('{}/{}.npy'.format(self.mfcc_loc, key))
            except KeyError:
                pass
        args_list = np.vstack([index_list, feature_list, vad_list, scp_list, save_list]).T
        frames = run_parallel(self.run_vad_and_save, args_list, self.n_jobs, p_bar=False)
        frame_dict = dict()
        for i, key in enumerate(args_list[:, 0]):
            frame_dict[key] = frames[i]
        return frame_dict

    def extract(self, data_scp):
        return Kaldi().run_command('sh ./kaldi/make_mfcc.sh {} {}'.format(data_scp, self.params_file))

    def run_vad_and_save(self, args):
        if not exists(args[4]):
            with open(args[3], 'w') as f:
                f.write('{} {}'.format(args[0], args[1]))
            features = read_feat(args[3], self.n_ceps)

            with open(args[3], 'w') as f:
                f.write('{} {}'.format(args[0], args[2]))
            vad = read_vector(args[3])

            remove_file(args[3])
            features = features[:, vad]
            features = cmvn(features)
            features = window_cmvn(features, window_len=301, var_norm=False)
            save_array(args[4], features)
        else:
            features = load_array(args[4])
        return features.shape[1]


class VAD:
    def __init__(self, threshold=5.5, mean_scale=0.5, n_jobs=20, save_loc='../save'):
        vad_loc = join_path(save_loc, VAD_DIR)
        params_file = join_path(vad_loc, 'vad.params')
        config_file = join_path(vad_loc, 'vad.conf')

        with open(params_file, 'w') as f:
            f.write('nj={}\n'.format(n_jobs))
            f.write('vad_loc={}\n'.format(vad_loc))
            f.write('vad_config={}\n'.format(config_file))

        with open(config_file, 'w') as f:
            f.write('--vad-energy-threshold={}\n'.format(threshold))
            f.write('--vad-energy-mean-scale={}\n'.format(mean_scale))

        self.params_file = params_file

    def compute(self, feats_scp):
        return Kaldi().run_command('sh ./kaldi/compute_vad.sh {} {}'.format(feats_scp, self.params_file))


def add_frames_to_args(args_list, frame_dict):
    frames = []
    for key in args_list[:, 0]:
        frames.append(frame_dict[key])
    return np.vstack([args_list.T, frames]).T


def cmvn(x, var_norm=True):
    y = x - x.mean(1, keepdims=True)
    if var_norm:
        y /= (x.std(1, keepdims=True) + 1e-20)
    return y


def generate_data_scp(save_loc, args_list, append=False):
    data_scp_file = join_path(save_loc, DATA_SCP_FILE)
    with open(data_scp_file, 'a' if append else 'w') as f:
        for args in args_list:
            f.write('{} {} |\n'.format(args[0], args[4]))


def get_frame(file_loc):
    return load_array(file_loc).shape[1]


def get_mfcc_frames(save_loc, args, n_jobs=10):
    mfcc_loc = join_path(save_loc, MFCC_DIR)
    file_loc = []
    for a in args:
        file_loc.append(join_path(mfcc_loc, a + '.npy'))
    return np.array(run_parallel(get_frame, file_loc, n_jobs)).reshape([-1, 1])


def load_feature(file_name):
    return load_array(file_name)


def remove_present_from_scp(save_loc, n_jobs=10):
    data_scp_file = join_path(save_loc, DATA_SCP_FILE)
    mfcc_loc = join_path(save_loc, MFCC_DIR)
    file_list = []
    index_list = []
    scp_list = []
    with open(data_scp_file, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[\s]+', line.strip())
            file_list.append('{}/{}.npy'.format(mfcc_loc, tokens[0]))
            index_list.append(tokens[0])
            scp_list.append(line)
    absent = np.invert(run_parallel(exists, file_list, n_jobs, p_bar=False), dtype=bool)
    scp_list = np.array(scp_list)[absent]
    with open(data_scp_file, 'w') as f:
        f.writelines(scp_list)
    return sum(absent)


def window_cmvn(x, window_len=301, var_norm=True):
    if window_len < 3 or (window_len & 1) != 1:
        raise ValueError('Window length should be an odd integer >= 3')
    n_dim, n_obs = x.shape
    if n_obs < window_len:
        return cmvn(x, var_norm)
    h_len = int((window_len - 1) / 2)
    y = np.zeros((n_dim, n_obs), dtype=x.dtype)
    y[:, :h_len] = x[:, :h_len] - x[:, :window_len].mean(1, keepdims=True)
    for ix in range(h_len, n_obs-h_len):
        y[:, ix] = x[:, ix] - x[:, ix-h_len:ix+h_len+1].mean(1)
    y[:, n_obs-h_len:n_obs] = x[:, n_obs-h_len:n_obs] - x[:, n_obs - window_len:].mean(1, keepdims=True)
    if var_norm:
        y[:, :h_len] /= (x[:, :window_len].std(1, keepdims=True) + 1e-20)
        for ix in range(h_len, n_obs-h_len):
            y[:, ix] /= (x[:, ix-h_len:ix+h_len+1].std(1) + 1e-20)
        y[:, n_obs-h_len:n_obs] /= (x[:, n_obs - window_len:].std(1, keepdims=True) + 1e-20)
    return y
