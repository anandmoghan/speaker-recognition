from collections import Counter
from subprocess import Popen, PIPE
from os.path import abspath, join as join_path

import numpy as np
import re

from constants.app_constants import KALDI_QUEUE_FILE, KALDI_PATH_FILE, DATA_DIR, EMB_DIR, LOGS_DIR, PLDA_DIR, \
    EMB_SCP_FILE, SPK_UTT_FILE, UTT_SPK_FILE, TRAIN_SPLIT, ENROLL_SPLIT, TEST_SPLIT, NUM_UTT_FILE, SCORES_FILE, \
    TRIALS_FILE, UNLABELLED_SPLIT, EER_INPUT_FILE
from services.common import run_parallel, load_array


class Kaldi:
    def __init__(self, path_file=KALDI_PATH_FILE):
        self.command = 'source {}'.format(path_file)

    def run_command(self, cmd, decode=True, print_error=False):
        cmd = '{} && ({})'.format(self.command, cmd)
        output, error = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
        if error is not None:
            error = error.decode("utf-8")
            if print_error and not error == '':
                print(error)
        if decode:
            return output.decode("utf-8")
        return output

    def queue(self, cmd, queue_loc=KALDI_QUEUE_FILE, decode=True, print_error=True):
        cmd = '{} && ({} {})'.format(self.command, queue_loc, cmd)
        return self.run_command(cmd, decode, print_error)


class PLDA:
    def __init__(self, model_tag, total_covariance_factor=0.0, save_loc='../save'):
        self.model_tag = model_tag
        self.tcf = total_covariance_factor
        self.save_loc = save_loc
        self.data_loc = join_path(save_loc, DATA_DIR)
        self.logs_loc = join_path(save_loc, LOGS_DIR)
        self.embedding_loc = join_path(save_loc, join_path(EMB_DIR, model_tag))
        self.plda_loc = join_path(save_loc, PLDA_DIR)

        self.centering_mean = '{}/mean_{}.vec'.format(self.plda_loc, self.model_tag)
        self.transform_matrix = '{}/transform_{}.mat'.format(self.plda_loc, model_tag)
        self.plda_model = '{}/plda_{}'.format(self.plda_loc, model_tag)
        self.scores_file = '{}/{}_{}'.format(self.plda_loc, model_tag, SCORES_FILE)
        self.eer_file = '{}/{}_{}'.format(self.plda_loc, model_tag, EER_INPUT_FILE)
        self.trials_file = join_path(save_loc, TRIALS_FILE)

    def fit(self, index_list, speaker_list, lda_dim=150, split=TRAIN_SPLIT, centering_split=UNLABELLED_SPLIT):
        spk_utt_file = join_path(self.save_loc, '{}_{}'.format(SPK_UTT_FILE, split))
        utt_spk_file = join_path(self.save_loc, '{}_{}'.format(UTT_SPK_FILE, split))
        make_spk_to_utt(index_list, speaker_list, spk_utt_file)
        make_utt_to_spk(index_list, speaker_list, utt_spk_file)

        embedding_scp = join_path(self.save_loc, get_embedding_scp(EMB_SCP_FILE, self.model_tag, split))
        centering_embedding_scp = join_path(self.save_loc, get_embedding_scp(EMB_SCP_FILE, self.model_tag, centering_split))

        print("PLDA: Computing mean...")  # SRE MAJOR
        Kaldi().queue('{}/compute_mean.log ivector-mean scp:{} {} || exit 1;'
                      .format(self.logs_loc, centering_embedding_scp, self.centering_mean), print_error=True)

        print("PLDA: Computing LDA...")
        Kaldi().queue('{}/lda.log ivector-compute-lda --total-covariance-factor={} --dim={} '
                      '"ark:ivector-subtract-global-mean scp:{} ark:- |" ark:{} {} || exit 1;'
                      .format(self.logs_loc, self.tcf, lda_dim, embedding_scp, utt_spk_file, self.transform_matrix), print_error=True)

        print('PLDA: Fitting PLDA model...')
        Kaldi().queue('{}/plda.log ivector-compute-plda ark:{} "ark:ivector-subtract-global-mean scp:{} ark:- | '
                      'transform-vec {} ark:- ark:- | ivector-normalize-length ark:- ark:- |" {} || exit 1;'
                      .format(self.logs_loc, spk_utt_file, embedding_scp, self.transform_matrix, self.plda_model), print_error=True)

    def score(self, enroll_index_list, enroll_speaker_list, enroll_split=ENROLL_SPLIT, test_split=TEST_SPLIT):
        num_utterances_file = join_path(self.save_loc, '{}_{}'.format(NUM_UTT_FILE, enroll_split))
        enroll_spk_utt_file = join_path(self.save_loc, '{}_{}'.format(SPK_UTT_FILE, enroll_split))
        enroll_utt_spk_file = join_path(self.save_loc, '{}_{}'.format(UTT_SPK_FILE, enroll_split))
        make_num_utterance(enroll_speaker_list, num_utterances_file)
        make_spk_to_utt(enroll_index_list, enroll_speaker_list, enroll_spk_utt_file)
        make_utt_to_spk(enroll_index_list, enroll_speaker_list, enroll_utt_spk_file)

        enroll_embedding_scp = join_path(self.save_loc, get_embedding_scp(EMB_SCP_FILE, self.model_tag, enroll_split))
        test_embedding_scp = join_path(self.save_loc, get_embedding_scp(EMB_SCP_FILE, self.model_tag, test_split))

        print('PLDA: Computing scores...')
        Kaldi().queue('{}/sre16_eval_scoring.log ivector-plda-scoring --normalize-length=true --num-utts=ark:{} '
                      '"ivector-copy-plda --smoothing=0.0 {} - |" "ark:ivector-mean ark:{} scp:{} ark:- | '
                      'ivector-subtract-global-mean {} ark:- ark:- | transform-vec {} ark:- ark:- | '
                      'ivector-normalize-length ark:- ark:- |" "ark:ivector-subtract-global-mean {} scp:{} ark:- | '
                      'transform-vec {} ark:- ark:- | ivector-normalize-length ark:- ark:- |" '
                      '"cat {} | cut -d\  --fields=1,2 |" {} || exit 1;'
                      .format(self.logs_loc, num_utterances_file, self.plda_model, enroll_spk_utt_file,
                              enroll_embedding_scp, self.centering_mean, self.transform_matrix,
                              self.centering_mean, test_embedding_scp, self.transform_matrix,
                              self.trials_file, self.scores_file), print_error=True)
        return self.scores_file

    def compute_eer(self, enroll_index_list, enroll_speaker_list, enroll_split=ENROLL_SPLIT, test_split=TEST_SPLIT):
        self.score(enroll_index_list, enroll_speaker_list, enroll_split, test_split)
        print('PLDA: Making score file...')
        scores = []
        with open(self.scores_file) as f:
            for line in f.readlines():
                score = re.split('[\s]+', line.strip())[2]
                scores.append(score)

        target_list = []
        with open(self.trials_file) as f:
            for line in f.readlines():
                t = re.split('[\s]+', line.strip())[2]
                target_list.append(t)

        with open(self.eer_file, 'w') as f:
            for s, t in zip(scores, target_list):
                f.write('{} {}\n'.format(s, t))

        print('PLDA: Computing EER...')
        output = Kaldi().run_command('cat ' + self.eer_file + ' | compute-eer -')
        return output


def append_label_to_idx_scp(scp_file, index_list, label_list):
    scp_dict = spaced_file_to_dict(scp_file)

    with open(scp_file, 'w') as f:
        for i, l in zip(index_list, label_list):
            try:
                scp_content = scp_dict[i]
                f.write('{}_{} {}\n'.format(l, i, scp_content))
            except KeyError:
                pass


def append_vector(arr, utt_id, ark_file, offset=None):
    if offset is None:
        with open(ark_file, 'rb') as f:
            offset = len(f.read())

    np.set_printoptions(threshold=np.nan, linewidth=np.nan)
    arr = np.array(arr)
    output = Kaldi().run_command('echo [ {} ] | copy-vector - -'.format(np.array_str(arr)[1:-1]), decode=False)

    with open(ark_file, 'ab') as f:
        f.write(output)

    return '{} {}:{}\n'.format(utt_id, abspath(ark_file), offset), offset + len(output)


def convert_embeddings(index_list, model_tag, split=TRAIN_SPLIT, save_loc='../save', n_jobs=10):
    embedding_loc = join_path(save_loc, join_path(EMB_DIR, model_tag))
    embedding_scp = join_path(save_loc, get_embedding_scp(EMB_SCP_FILE, model_tag, split))

    npy_list = []
    ark_list = []
    for key in index_list:
        npy_list.append(join_path(embedding_loc, '{}.npy'.format(key)))
        ark_list.append(join_path(embedding_loc, '{}.ark'.format(key)))

    args_list = np.vstack([index_list, npy_list, ark_list]).T

    scp_list = run_parallel(get_kaldi_ark, args_list, n_jobs, p_bar=True)
    with open(embedding_scp, 'w') as f:
        for scp in scp_list:
            f.write(scp)
    return embedding_scp


def get_embedding_scp(embedding_scp, model_tag, split):
    return '{}_{}_{}'.format(embedding_scp, model_tag, split)


def get_kaldi_ark(args):
    vector = load_array(args[1])
    return write_vector(vector, args[0], args[2])


def make_num_utterance(speaker_list, num_utt_file):
    speaker_counter = Counter(speaker_list)
    with open(num_utt_file, 'w') as f:
        for speaker in speaker_counter.keys():
            f.write('{} {}\n'.format(speaker, speaker_counter[speaker]))


def make_spk_to_utt(index_list, speaker_list, spk_utt_file):
    idx = np.argsort(speaker_list)
    index_list = index_list[idx]
    speaker_list = speaker_list[idx]
    spk_to_utt_dict = dict()
    for u, s in zip(index_list, speaker_list):
        try:
            utt = spk_to_utt_dict[s]
            utt = '{} {}'.format(utt, u)
        except KeyError:
            utt = u
        spk_to_utt_dict[s] = utt

    with open(spk_utt_file, 'w') as f:
        for s, u in spk_to_utt_dict.items():
            f.write('{} {}\n'.format(s, u))


def make_utt_to_spk(index_list, speaker_list, utt_spk_file):
    idx = np.argsort(index_list)
    index_list = index_list[idx]
    speaker_list = speaker_list[idx]
    with open(utt_spk_file, 'w') as f:
        for u, s in zip(index_list, speaker_list):
            f.write('{} {}\n'.format(u, s))


def read_feat(scp_file, n_features):
    output = Kaldi().run_command('copy-feats scp:{} ark,t:'.format(scp_file))
    output = re.split('\[', output)[1][1:-2]
    return np.fromstring(output, dtype=float, sep=' \n').reshape([-1, n_features]).T


def read_feats(scp_file, n_features, print_error=False):
    output = Kaldi().run_command('copy-feats scp:{} ark,t:'.format(scp_file), print_error=print_error)
    features = re.split('\]', output)[:-1]
    utt_list = []
    feature_list = []
    for i, feature in enumerate(features):
        feature = re.split('\[', feature)
        utt_id = feature[0][:-1]
        feature = np.fromstring(feature[1][1:-2], dtype=float, sep=' \n').reshape([-1, n_features]).T
        utt_list.append(utt_id)
        feature_list.append(feature)
    return utt_list, feature_list


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


def spaced_file_to_dict(scp_file):
    file_dict = dict()
    with open(scp_file, 'r') as f:
        for line in f.readlines():
            tokens = re.split('[\s]+', line.strip())
            file_dict[tokens[0]] = tokens[1]
    return file_dict


def write_vector(arr, utt_id, ark_file):
    arr = np.array(arr)
    np.set_printoptions(threshold=np.nan, linewidth=np.nan)
    output = Kaldi().run_command('echo [ {} ] | copy-vector - -'.format(np.array_str(arr)[1:-1]), decode=False)
    with open(ark_file, 'wb') as f:
        f.write(output)
    return '{} {}:{}\n'.format(utt_id, abspath(ark_file), 0)
