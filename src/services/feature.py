from os import remove as remove_file
from os.path import exists, join as join_path

import numpy as np

from constants.app_constants import DATA_SCP_FILE, MFCC_DIR, VAD_DIR
from services.common import run_command, run_parallel, load_array, save_array
from services.kaldi import scp_to_dict


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

    def extract(self, data_scp):
        return run_command('sh ./kaldi/make_mfcc.sh {} {}'.format(data_scp, self.params_file))


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
        return run_command('sh ./kaldi/compute_vad.sh {} {}'.format(feats_scp, self.params_file))


def add_frames_to_args(args_list, frame_dict):
    frames = []
    for key in args_list[:, 0]:
        frames.append(frame_dict[key])
    return np.vstack([args_list.T, frames]).T


def apply_vad_and_save(feats_scp, vad_scp, n_workers=10, save_loc="../save"):
    mfcc_loc = join_path(save_loc, MFCC_DIR)
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
            scp_list.append('{}/{}.scp'.format(mfcc_loc, key))
            save_list.append('{}/{}.npy'.format(mfcc_loc, key))
        except KeyError:
            pass
    args_list = np.vstack([index_list, feature_list, vad_list, scp_list, save_list]).T
    frames = run_parallel(run_vad_and_save, args_list, n_workers, p_bar=False)
    frame_dict = dict()
    for i, key in enumerate(args_list[:, 0]):
        frame_dict[key] = frames[i]
    return frame_dict


def generate_data_scp(save_loc, args_list, append=False):
    data_scp_file = join_path(save_loc, DATA_SCP_FILE)
    with open(data_scp_file, 'a' if append else 'w') as f:
        for args in args_list:
            f.write('{} {} |\n'.format(args[0], args[4]))


def get_frame(file_loc):
    return load_array(file_loc).shape[1]


def get_mfcc_frames(save_loc, args):
    mfcc_loc = join_path(save_loc, MFCC_DIR)
    file_loc = []
    for a in args:
        file_loc.append(join_path(mfcc_loc, a + '.npy'))
    return np.array(run_parallel(get_frame, file_loc)).reshape([-1, 1])


def load_feature(file_name):
    return load_array(file_name)


def run_vad_and_save(args):
    if not exists(args[4]):
        idx = len(args[0])
        with open(args[3], 'w') as f:
            f.write('{} {}'.format(args[0], args[1]))
        features, _ = run_command('source ./kaldi/path.sh && copy-feats scp:{} ark,t:'.format(args[3]))
        features = np.fromstring(features[idx + 6:-3], dtype=float, sep=' \n').reshape([-1, 20]).T

        with open(args[3], 'w') as f:
            f.write('{} {}'.format(args[0], args[2]))
        vad, _ = run_command('source ./kaldi/path.sh && copy-vector scp:{} ark,t:'.format(args[3]))
        vad = np.fromstring(vad[idx + 4:-3], dtype=bool, sep=' ')
        remove_file(args[3])
        features = features[:, vad]
        save_array(args[4], features)
    else:
        features = load_array(args[4])
    return features.shape[1]
