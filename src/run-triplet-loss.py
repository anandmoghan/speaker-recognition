from services.common import get_file_list, run_parallel
from services.features import MFCC
from services.logger import Logger

import argparse as ap


SAMPLE_RATE = 8000
DATA_LOCATION = '/home/data/SRE04_TEST/test/data'
# DATA_LOCATION = '../data/'


parser = ap.ArgumentParser()
parser.add_argument('-s', '--stage', type=int, default=0, help='Set Stage')
args = parser.parse_args()

logger = Logger()

if args.stage <= 0:
    logger.start_timer('Stage 0: Extracting Features...')
    file_list = get_file_list(DATA_LOCATION)
    mfcc = MFCC(fs=SAMPLE_RATE, nchannels=24, fl=100, fh=4000, nceps=40)
    output = run_parallel(mfcc.extract_files, file_list, batch_size=256, n_workers=40)
    logger.end_timer('{} files processed'.format(len(file_list)))
