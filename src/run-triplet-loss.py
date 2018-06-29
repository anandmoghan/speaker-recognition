from services.common import load_object, run_parallel, save_object
from services.sre_data import get_sre_swbd_data, make_sre16_eval_data, make_sre16_unlabeled_data
from services.data import check_data_dir
from services.feature import MFCC
from services.logger import Logger

from os.path import join as join_path

import argparse as ap

SAVE_LOCATION = '../save'
SRE_CONFIG = '../configs/sre_data.json'


parser = ap.ArgumentParser()
parser.add_argument('-s', '--stage', type=int, default=0, help='Set Stage')
parser.add_argument('-sr', '--sample_rate', type=int, default=8000, help='Sampling Rate')
args = parser.parse_args()

logger = Logger(filename='../logs/run-triplet-loss.log', append=False)

if args.stage <= 0:
    logger.start_timer('Stage 0: Making data...')
    sre_swbd = get_sre_swbd_data(SRE_CONFIG)
    sre16_eval_enrollment, sre16_eval_test = make_sre16_eval_data(SRE_CONFIG)
    print(sre_swbd.shape)
    logger.info('Saving data lists..')
    save_object(join_path(SAVE_LOCATION, 'sre_swbd.pkl'), sre_swbd)
    save_object(join_path(SAVE_LOCATION, 'sre16_eval_enrollment.pkl'), sre16_eval_enrollment)
    save_object(join_path(SAVE_LOCATION, 'sre16_eval_test.pkl'), sre16_eval_test)
    logger.info('Data lists saved at: {}'.format(SAVE_LOCATION))
    logger.end_timer('Stage 0:')
else:
    logger.start_timer('Loading data lists from: {}'.format(SAVE_LOCATION))
    sre_swbd = load_object(join_path(SAVE_LOCATION, 'sre_swbd.pkl'))
    sre16_eval_enrollment = load_object(join_path(SAVE_LOCATION, 'sre16_eval_enrollment.pkl'))
    sre16_eval_test = load_object(join_path(SAVE_LOCATION, 'sre16_eval_test.pkl'))
    logger.end_timer('Data loaded.')

exit(1)

if args.stage <= 1:
    logger.start_timer('Stage 1: Extracting Features...')
    mfcc = MFCC(fs=args.sample_rate, n_channels=24, fl=20, fh=3700, n_ceps=23)
    combined_features = run_parallel(mfcc.extract_sph_files_with_sad_and_cmvn, sre_swbd, batch_size=32, n_workers=20)
    save_object(join_path(SAVE_LOCATION, 'combined_features.pkl'), combined_features)
    logger.info('Features saved at: {}'.format(SAVE_LOCATION))
    logger.end_timer('Stage 1:')
else:
    logger.start_timer('Loading Features...')
    features = load_object(join_path(SAVE_LOCATION, 'feature.pkl'))
    logger.end_timer()

if args.stage <= 2:
    logger.start_timer('Stage 2: Neural Net Model...')
    logger.end_timer()
