import re
from os.path import exists, join as join_path

import numpy as np

from constants.app_constants import DATA_SCP_FILE, MFCC_DIR, VAD_DIR, FEATS_SCP_FILE, UTT2NUM_FRAMES_FILE, TMP_DIR, \
    VAD_SCP_FILE
from services.common import load_array, run_parallel, run_command
from services.kaldi import Kaldi, spaced_file_to_dict


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
        self.save_loc = save_loc
        self.params_file = params_file
        self.n_ceps = n_ceps
        self.n_jobs = n_jobs

    def extract(self, data_scp):
        return Kaldi().run_command('sh ./kaldi/make_mfcc.sh {} {}'.format(data_scp, self.params_file))

    def extract_with_vad_and_normalization(self, data_scp, threshold=5.5, mean_scale=0.5, cmvn_window=300, var_norm=False):
        vad_loc = join_path(self.save_loc, VAD_DIR)
        tmp_loc = join_path(self.save_loc, TMP_DIR)

        feats_scp = join_path(self.save_loc, FEATS_SCP_FILE)
        vad_scp = join_path(self.save_loc, VAD_SCP_FILE)

        print('MFCC: Extracting features...')
        self.extract(data_scp)

        print('MFCC: Computing VAD...')
        vad = VAD(threshold, mean_scale, n_jobs=self.n_jobs, save_loc=self.save_loc)
        vad.compute(feats_scp)

        print('MFCC: Normalizing features and selecting voiced frames..')
        feats_scp_dict = spaced_file_to_dict(feats_scp)
        vad_scp_dict = spaced_file_to_dict(vad_scp)

        splits = np.array_split(list(feats_scp_dict.keys()), self.n_jobs)
        for i in range(self.n_jobs):
            split_feat_scp = open(join_path(tmp_loc, 'feats.{}.scp'.format(i + 1)), 'w')
            split_vad_scp = open(join_path(tmp_loc, 'vad.{}.scp'.format(i + 1)), 'w')
            for key in splits[i]:
                split_feat_scp.write('{} {}\n'.format(key, feats_scp_dict[key]))
                split_vad_scp.write('{} {}\n'.format(key, vad_scp_dict[key]))
            split_feat_scp.close()
            split_vad_scp.close()

        Kaldi().queue('JOB=1:{nj} {mfcc_loc}/log/voiced_feats.JOB.log '
                      'apply-cmvn-sliding --norm-vars={var_norm} --center=true --cmn-window={window} scp:{tmp_loc}/feats.JOB.scp ark:- \| '
                      'select-voiced-frames ark:- scp,ns,cs:{tmp_loc}/vad.JOB.scp ark:- \| '
                      'copy-feats --compress=false --write-num-frames=ark,t:{mfcc_loc}/log/utt2num_frames.JOB ark:- '
                      'ark,scp:{mfcc_loc}/voiced_feats.JOB.ark,{mfcc_loc}/voiced_feats.JOB.scp || exit 1;'
                      .format(mfcc_loc=self.mfcc_loc, tmp_loc=tmp_loc, vad_loc=vad_loc, var_norm='true' if var_norm else 'false',
                              nj=self.n_jobs, window=cmvn_window))

        run_command('for n in $(seq {nj}); do \n'
                    '   cat {mfcc_loc}/voiced_feats.$n.scp || exit 1;\n'
                    'done > {mfcc_loc}/feats.scp || exit 1'.format(mfcc_loc=self.mfcc_loc, nj=self.n_jobs))

        run_command('for n in $(seq {nj}); do \n'
                    '   cat {mfcc_loc}/log/utt2num_frames.$n || exit 1;\n'
                    'done > {mfcc_loc}/utt2num_frames || exit 1'.format(mfcc_loc=self.mfcc_loc, nj=self.n_jobs))


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


def generate_data_scp(save_loc, args_list, append=False):
    data_scp_file = join_path(save_loc, DATA_SCP_FILE)
    with open(data_scp_file, 'a' if append else 'w') as f:
        for args in args_list:
            f.write('{} {} |\n'.format(args[0], args[4]))


def get_frame(file_loc):
    return load_array(file_loc).shape[1]


def get_mfcc_frames(save_loc, args):
    utt2num_frames = join_path(save_loc, UTT2NUM_FRAMES_FILE)
    utt2num_frames_dict = spaced_file_to_dict(utt2num_frames)
    frames = []
    count = 0
    for a in args:
        try:
            frames.append(utt2num_frames_dict[a])
        except KeyError:
            count = count + 1
    if count > 0:
        print('MFCC FRAMES: Can not find {} utterances in utt2num_frames'.format(count))
    return np.array(frames).reshape([-1, 1])


def load_feature(file_name):
    return load_array(file_name)


def remove_bad_files(args_list, save_loc='../save'):
    feats_scp = join_path(save_loc, FEATS_SCP_FILE)
    feats_scp_dict = spaced_file_to_dict(feats_scp)

    bad_files = []
    for i, key in enumerate(args_list[:, 0]):
        try:
            _ = feats_scp_dict[key]
        except KeyError:
            bad_files.append(i)
    return np.delete(args_list, bad_files, axis=0)


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
