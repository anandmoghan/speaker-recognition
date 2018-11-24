from os.path import join as join_path, abspath

from constants.app_constants import DATA_DIR, MFCC_DIR, VAD_DIR
from services.common import make_directory
from services.feature import MFCC

import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--num-features', type=int, default=23, help='Number of MFCC Co-efficients')
parser.add_argument('-nj', '--num-jobs', type=int, default=40, help='Number of parallel jobs')
parser.add_argument('--sample-rate', type=int, default=8000, help='Sampling Rate')
parser.add_argument('--save', default='../save', help='Save Location')
args = parser.parse_args()


def make_feats(mfcc, split, save_loc):
    data_loc = join_path(join_path(save_loc, DATA_DIR), split)
    mfcc.extract_with_vad_and_normalization(data_loc, split)


if __name__ == '__main__':
    args.save = abspath(args.save)
    mfcc_loc = join_path(args.save, MFCC_DIR)
    vad_loc = join_path(args.save, VAD_DIR)
    make_directory(mfcc_loc)
    make_directory(vad_loc)
    mfcc_ = MFCC(fs=args.sample_rate, fl=20, fh=3700, frame_len_ms=25, n_ceps=args.num_features,
                 n_jobs=args.num_jobs, save_loc=args.save)
    for split_ in ['train_data', 'sre_unlabelled', 'sre_dev_enroll', 'sre_dev_test', 'sre_eval_enroll',
                   'sre_eval_test']:
        print('Making features for {}..'.format(split_))
        make_feats(mfcc_, split_, args.save)
        print('Finished making features for {}..'.format(split_))
