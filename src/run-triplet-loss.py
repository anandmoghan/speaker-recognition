from collections import Counter

from models.x_vector import XVectorModel
from services.checks import check_mfcc, check_sad
from services.common import load_object, tensorflow_debug, run_parallel, save_object
from services.loader import SRESplitBatchLoader
from services.sre_data import get_sre_swbd_data, make_sre16_eval_data
from services.feature import MFCC, generate_sad_list, get_mfcc_frames
from services.logger import Logger

from os.path import join as join_path

import argparse as ap
import numpy as np

SRE_CONFIG = '../configs/sre_data.json'

parser = ap.ArgumentParser()
parser.add_argument('--bg', action="store_true", default=False, help='Background Option')
parser.add_argument('--batch-size', type=int, default=64, help='Batch Size')
parser.add_argument('--epochs', type=int, default=8, help='Number of Epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--num-features', type=int, default=24, help='Batch Size')
parser.add_argument('--sample-rate', type=int, default=8000, help='Sampling Rate')
parser.add_argument('--save', default='../save', help='Save Location')
parser.add_argument('--stage', type=int, default=2, help='Set Stage')
args = parser.parse_args()

logger = Logger()
logger.set_config(filename='../logs/run-triplet-loss.log', append=False)

tensorflow_debug(False)

if args.stage <= 0:
    logger.start_timer('Stage 0: Making data...')
    sre_swbd = get_sre_swbd_data(SRE_CONFIG)
    logger.info('Stage 0: Made {:d} files from sre_swbd.'.format(sre_swbd.shape[0]))
    sre16_eval_enrollment, sre16_eval_test = make_sre16_eval_data(SRE_CONFIG)
    logger.info('Stage 0: Saving data lists..')
    save_object(join_path(args.save, 'sre_swbd.pkl'), sre_swbd)
    save_object(join_path(args.save, 'sre16_eval_enrollment.pkl'), sre16_eval_enrollment)
    save_object(join_path(args.save, 'sre16_eval_test.pkl'), sre16_eval_test)
    # TODO: Make sre2016 unlabeled data
    logger.info('Stage 0: Data lists saved at: {}'.format(args.save))
    logger.info('Stage 0: Generating SAD list file.')
    generate_sad_list(args.save, sre_swbd)
    generate_sad_list(args.save, sre16_eval_enrollment, append=True)
    generate_sad_list(args.save, sre16_eval_test, append=True)
    logger.info('Stage 0: Saved SAD list file at: {}'.format(args.save))
    logger.end_timer('Stage 0:')
else:
    logger.start_timer('Load: Data lists from: {}'.format(args.save))
    sre_swbd = load_object(join_path(args.save, 'sre_swbd.pkl'))
    sre16_eval_enrollment = load_object(join_path(args.save, 'sre16_eval_enrollment.pkl'))
    sre16_eval_test = load_object(join_path(args.save, 'sre16_eval_test.pkl'))
    logger.end_timer('Load:')


if args.stage <= 1:
    logger.start_timer('Stage 1: Extracting Features...')
    success, fail = check_sad(args.save, sre_swbd)
    if fail > 0:
        print('Warning: SAD files not present for {:d} utterances.'.format(fail))
    mfcc = MFCC(fs=args.sample_rate, n_channels=args.num_features, fl=20, fh=3700, n_ceps=40, save_loc=args.save)
    frames = run_parallel(mfcc.extract_sph_file_with_sad_and_cmvn, sre_swbd, n_workers=16, p_bar=(not args.bg))
    frames = np.array(frames).reshape(-1, 1)
    logger.info('Stage 1: {:d} files processed.'.format(frames.shape[0]))
    sre_swbd = np.hstack([sre_swbd, frames])
    save_object(join_path(args.save, 'sre_swbd.pkl'), sre_swbd)
    logger.info('Stage 1: Indices saved.')
    logger.end_timer('Stage 1:')
else:
    logger.start_timer('Check: Looking for features...')
    success, fail = check_mfcc(args.save, sre_swbd)
    if fail > 0:
        raise Exception('No features for {:d} files. Execute Stage 1 before proceeding.'.format(fail))
    elif sre_swbd.shape[1] < 5:
        logger.info('Check: Fetching frames...')
        frames = get_mfcc_frames(args.save, sre_swbd[:, 0])
        logger.info('Check: Appending and saving...')
        sre_swbd = np.hstack([sre_swbd, frames])
        save_object(join_path(args.save, 'sre_swbd.pkl'), sre_swbd)
    logger.end_timer('Check:')


if args.stage <= 2:
    logger.start_timer('Stage 2: Pre-processing...')
    logger.info('Stage 2: Filtering out short duration utterances and sorting by duration...')
    sre_swbd = sre_swbd[np.array(sre_swbd[:, 4], dtype=int) >= 300]
    sre_swbd = sre_swbd[np.argsort(np.array(sre_swbd[:, 4], dtype=int))]
    logger.info('Stage 2: Filtering out speakers having lesser training data...')
    speakers = sre_swbd[:, 3]
    unique_speakers = set(speakers)
    logger.info('Stage 2: Total Speakers before filtering: {:d}'.format(len(unique_speakers)))
    speaker_counter = Counter(speakers)
    good_speakers = []
    for speaker in unique_speakers:
        if speaker_counter[speaker] >= 5:
            good_speakers.append(speaker)
    sre_swbd = sre_swbd[np.in1d(speakers, good_speakers), :]
    speakers = sre_swbd[:, 3]
    unique_speakers = set(speakers)
    logger.info('Stage 2: Total Speakers after filtering: {:d}'.format(len(unique_speakers)))
    logger.info('Stage 2: Training Utterances after filtering: {:d}'.format(sre_swbd.shape[0]))
    logger.info('Stage 2: Making Speaker dictionaries...')
    n_speakers = len(unique_speakers)
    speaker_to_idx = dict(zip(unique_speakers, range(len(unique_speakers))))
    idx_to_speaker = dict(zip(range(len(unique_speakers)), unique_speakers))
    sre_swbd[:, 3] = np.array(map(lambda x: speaker_to_idx[x], speakers))
    save_object(join_path(args.save, 'sre_swbd.pkl'), sre_swbd)
    save_object(join_path(args.save, 'speaker_to_idx.pkl'), speaker_to_idx)
    save_object(join_path(args.save, 'idx_to_speaker.pkl'), idx_to_speaker)
    logger.end_timer('Stage 2:')
else:
    logger.start_timer('Load: Speaker dictionaries from: {}'.format(args.save))
    speakers = sre_swbd[:, 3]
    n_speakers = len(set(speakers))
    speaker_to_idx = load_object(join_path(args.save, 'speaker_to_idx.pkl'))
    idx_to_speaker = load_object(join_path(args.save, 'idx_to_speaker.pkl'))
    logger.end_timer('Load:')


if args.stage <= 3:
    logger.start_timer('Stage 3: x-Vector Model Training...')
    batch_loader = SRESplitBatchLoader(location=args.save, args=sre_swbd, n_features=args.num_features,
                                       splits=[300, 1000, 3000, 6000], batch_size=args.batch_size)
    model = XVectorModel(batch_size=args.batch_size, n_features=args.num_features, n_classes=n_speakers,
                         learning_rate=args.lr)
    model.start_train(batch_loader, args.epochs)
    batch_x, batch_y = batch_loader.next()
    logger.end_timer('Stage 3:')
