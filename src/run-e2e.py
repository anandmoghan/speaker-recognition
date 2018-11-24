from collections import Counter
from os.path import join as join_path, abspath

import argparse as ap
import numpy as np

from constants.app_constants import DATA_DIR, DATA_SCP_FILE, TRAIN_SPLIT, ENROLL_SPLIT, TEST_SPLIT, UNLABELLED_SPLIT, \
    TRIALS_FILE
from models.attention_model import AttentionModel
from models.hierarchical import HGRUModel
from models.x_vector import XVectorModel
from services.checks import check_mfcc, check_embeddings
from services.common import create_directories, load_object, save_object, tensorflow_debug, use_gpu
from services.feature import MFCC, generate_data_scp, get_mfcc_frames, remove_bad_files
from services.kaldi import PLDA, convert_embeddings
from services.loader import LabelExtractLoader
from services.logger import Logger
from services.sre_data import get_train_data, make_sre16_eval_data, make_sre18_unlabelled_data, make_sre16_trials_file, \
    split_trials_file

DATA_CONFIG = '../configs/sre_data.json'

parser = ap.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, help='Training Batch Size')
parser.add_argument('--cont', action='store_true', help='Continue Training')
parser.add_argument('--decay', type=float, default=0.9, help='Decay Rate')
parser.add_argument('--epochs', type=int, default=20, help='Number of Epochs')
parser.add_argument('--gpu', type=int, default=0, help='Select GPU')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--model-tag', default='XVECTOR', help='Model Tag')
parser.add_argument('--num-features', type=int, default=23, help='Number of MFCC Co-efficients')
parser.add_argument('--num-jobs', type=int, default=20, help='Number of parallel jobs')
parser.add_argument('--sample-rate', type=int, default=8000, help='Sampling Rate')
parser.add_argument('--save', default='../save', help='Save Location')
parser.add_argument('-sc', '--skip-check', action="store_true", default=False, help='Skip Check')
parser.add_argument('--stage', type=int, default=0, help='Set Stage')
args = parser.parse_args()

logger = Logger()
logger.set_config(filename='../logs/run-model.log', append=False)

args.save = abspath(args.save)
data_loc = join_path(args.save, DATA_DIR)
create_directories(args.save)

tensorflow_debug(False)
use_gpu(args.gpu)

if args.stage <= 0:
    logger.start_timer('Stage 0: Making data...')
    train_data = get_train_data(DATA_CONFIG)
    logger.info('Stage 0: Made {:d} files for training.'.format(train_data.shape[0]))
    sre_unlabelled = make_sre18_unlabelled_data(DATA_CONFIG)
    logger.info('Stage 0: Made {:d} unlabelled files.'.format(sre_unlabelled.shape[0]))
    sre_enroll, sre_test = make_sre16_eval_data(DATA_CONFIG)
    logger.info('Stage 0: Saving data lists..')
    save_object(join_path(data_loc, 'train_data.pkl'), train_data)
    save_object(join_path(data_loc, 'sre_unlabelled.pkl'), sre_unlabelled)
    save_object(join_path(data_loc, 'sre_enroll.pkl'), sre_enroll)
    save_object(join_path(data_loc, 'sre_test.pkl'), sre_test)
    logger.info('Stage 0: Data lists saved at: {}'.format(data_loc))
    logger.info('Stage 0: Making trials file..')
    trials_file = join_path(args.save, TRIALS_FILE)
    make_sre16_trials_file(DATA_CONFIG, trials_file)
    logger.end_timer('Stage 0:')
else:
    logger.start_timer('Load: Data lists from: {}'.format(data_loc))
    train_data = load_object(join_path(data_loc, 'train_data.pkl'))
    sre_unlabelled = load_object(join_path(data_loc, 'sre_unlabelled.pkl'))
    sre_enroll = load_object(join_path(data_loc, 'sre_enroll.pkl'))
    sre_test = load_object(join_path(data_loc, 'sre_test.pkl'))
    logger.end_timer('Load:')

if args.stage <= 1:
    logger.start_timer('Stage 1: Feature Extraction.')
    logger.info('Stage 1: Generating data scp file...')
    generate_data_scp(args.save, train_data)
    generate_data_scp(args.save, sre_unlabelled, append=True)
    generate_data_scp(args.save, sre_enroll, append=True)
    generate_data_scp(args.save, sre_test, append=True)
    logger.info('Stage 1: Saved data scp file at: {}'.format(data_loc))

    data_scp_file = join_path(args.save, DATA_SCP_FILE)
    mfcc = MFCC(fs=args.sample_rate, fl=20, fh=3700, frame_len_ms=25, n_ceps=args.num_features, n_jobs=args.num_jobs,
                save_loc=args.save)
    mfcc.extract_with_vad_and_normalization(data_scp_file)

    logger.info('Stage 1: Removing bad training audio...')
    train_data = remove_bad_files(train_data, args.save)

    logger.info('Stage 1: Appending Frame counts..')
    frames = get_mfcc_frames(args.save, train_data[:, 0])
    train_data = np.hstack([train_data, frames])
    frames = get_mfcc_frames(args.save, sre_unlabelled[:, 0])
    sre_unlabelled = np.hstack([sre_unlabelled, frames])
    frames = get_mfcc_frames(args.save, sre_enroll[:, 0])
    sre_enroll = np.hstack([sre_enroll, frames])
    frames = get_mfcc_frames(args.save, sre_test[:, 0])
    sre_test = np.hstack([sre_test, frames])

    logger.info('Stage 1: Updating data lists..')
    save_object(join_path(data_loc, 'train_data.pkl'), train_data)
    save_object(join_path(data_loc, 'sre_unlabelled.pkl'), sre_unlabelled)
    save_object(join_path(data_loc, 'sre_enroll.pkl'), sre_enroll)
    save_object(join_path(data_loc, 'sre_test.pkl'), sre_test)
    logger.info('Stage 1: Data lists saved at: {}'.format(data_loc))
    logger.end_timer('Stage 1:')
elif args.stage < 5 and not args.skip_check:
    logger.start_timer('Check: Looking for features...')
    _, fail1 = check_mfcc(args.save, train_data)
    _, fail2 = check_mfcc(args.save, sre_unlabelled)
    _, fail3 = check_mfcc(args.save, sre_enroll)
    _, fail4 = check_mfcc(args.save, sre_test)
    fail = fail1 + fail2 + fail3 + fail4
    if fail > 0:
        raise Exception('No features for {:d} file(s). Execute Stage 1 before proceeding.'.format(fail))
    else:
        if train_data.shape[1] < 6:
            logger.info('Check: Fetching train_data frames counts...')
            frames = get_mfcc_frames(args.save, train_data[:, 0])
            logger.info('Check: Appending and saving...')
            train_data = np.hstack([train_data, frames])
            save_object(join_path(data_loc, 'train_data.pkl'), train_data)
        if sre_unlabelled.shape[1] < 6:
            logger.info('Check: Fetching sre_unlabelled frames counts...')
            frames = get_mfcc_frames(args.save, sre_unlabelled[:, 0])
            logger.info('Check: Appending and saving...')
            sre_unlabelled = np.hstack([sre_unlabelled, frames])
            save_object(join_path(data_loc, 'sre_unlabelled.pkl'), sre_unlabelled)
        if sre_enroll.shape[1] < 6:
            logger.info('Check: Fetching sre_enroll frames counts...')
            frames = get_mfcc_frames(args.save, sre_enroll[:, 0])
            logger.info('Check: Appending and saving...')
            sre_enroll = np.hstack([sre_enroll, frames])
            save_object(join_path(data_loc, 'sre_enroll.pkl'), sre_enroll)
        if sre_test.shape[1] < 7:
            logger.info('Check: Fetching sre_test frames counts...')
            frames = get_mfcc_frames(args.save, sre_test[:, 0])
            logger.info('Check: Appending and saving...')
            sre_test = np.hstack([sre_test, frames])
            save_object(join_path(data_loc, 'sre_test.pkl'), sre_test)
    logger.end_timer('Check:')

if args.stage <= 2:
    logger.start_timer('Stage 2: Pre-processing...')
    logger.info('Stage 2: Filtering out short duration utterances and sorting by duration...')
    train_data = train_data[np.array(train_data[:, -1], dtype=int) >= 500]
    train_data = train_data[np.argsort(np.array(train_data[:, -1], dtype=int))]
    logger.info('Stage 2: Filtering out speakers having lesser training data...')
    speakers = train_data[:, 3]
    unique_speakers = set(speakers)
    logger.info('Stage 2: Total Speakers before filtering: {:d}'.format(len(unique_speakers)))
    speaker_counter = Counter(speakers)
    good_speakers = []
    for speaker in unique_speakers:
        if speaker_counter[speaker] >= 8:
            good_speakers.append(speaker)
    train_data = train_data[np.in1d(speakers, good_speakers), :]
    speakers = train_data[:, 3]
    unique_speakers = set(speakers)
    logger.info('Stage 2: Total Speakers after filtering: {:d}'.format(len(unique_speakers)))
    logger.info('Stage 2: Training Utterances after filtering: {:d}'.format(train_data.shape[0]))
    save_object(join_path(data_loc, 'train_data.pkl'), train_data)

if args.stage <= 4:
    model = AttentionModel(n_features=args.num_features, n_classes=2, model_tag=args.model_tag)
    if args.stage <= 3:
        logger.start_timer('Stage 3: Train Neural Net.')
        model.start_train(args_list=train_data, epochs=args.epochs, lr=args.lr,
                          decay=args.decay, batch_size=args.batch_size, save_loc=args.save, cont=args.cont)
        logger.end_timer('Stage 3:')

    logger.start_timer('Stage 4: Scoring.')
    logger.info('Stage 4: Evaluating..')
    trials_file = join_path(args.save, TRIALS_FILE)
    model.evaluate(trials_file, sre_test, args.batch_size, save_loc=args.save)
    logger.end_timer('Stage 4:')
