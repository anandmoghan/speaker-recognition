from collections import Counter
from os.path import join as join_path, abspath

import argparse as ap
import numpy as np

from constants.app_constants import DATA_DIR, DATA_SCP_FILE, FEATS_SCP_FILE, VAD_SCP_FILE, EMB_DIR
from models.hierarchical import HGRUTripletModel
from services.checks import check_embeddings, check_mfcc
from services.common import create_directories, load_object, save_object
from services.feature import MFCC, VAD, generate_data_scp, get_mfcc_frames, remove_present_from_scp
from services.kaldi import PLDA
from services.loader import SRETestLoader, SRESplitBatchLoader
from services.logger import Logger
from services.sre_data import get_train_data, make_sre16_eval_data

DATA_CONFIG = '../configs/sre_data.json'

parser = ap.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, help='Training Batch Size')
parser.add_argument('--decay', type=float, default=0.1, help='Decay Rate')
parser.add_argument('--epochs', type=int, default=5, help='Number of Epochs')
parser.add_argument('--extract-batch-size', type=int, default=32, help='Extract Batch Size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--num-features', type=int, default=20, help='Number of MFCC Co-efficients')
parser.add_argument('--num-jobs', type=int, default=20, help='Number of parallel jobs')
parser.add_argument('--sample-rate', type=int, default=8000, help='Sampling Rate')
parser.add_argument('--save', default='../save', help='Save Location')
parser.add_argument('-sc', '--skip-check', action="store_true", default=False, help='Skip Check')
parser.add_argument('--stage', type=int, default=0, help='Set Stage')
args = parser.parse_args()

logger = Logger()
logger.set_config(filename='/logs/run-triplet-loss.log', append=False)

args.save = abspath(args.save)
data_loc = join_path(args.save, DATA_DIR)
create_directories(args.save)

if args.stage <= 0:
    logger.start_timer('Stage 0: Making data...')
    train_data = get_train_data(DATA_CONFIG)
    logger.info('Stage 0: Made {:d} files for training.'.format(train_data.shape[0]))
    sre16_enroll, sre16_test = make_sre16_eval_data(DATA_CONFIG)
    logger.info('Stage 0: Saving data lists..')
    save_object(join_path(data_loc, 'train_data.pkl'), train_data)
    save_object(join_path(data_loc, 'sre16_enroll.pkl'), sre16_enroll)
    save_object(join_path(data_loc, 'sre16_test.pkl'), sre16_test)
    # TODO: Make sre2016 unlabeled data
    logger.info('Stage 0: Data lists saved at: {}'.format(data_loc))
    logger.end_timer('Stage 0:')
else:
    logger.start_timer('Load: Data lists from: {}'.format(data_loc))
    train_data = load_object(join_path(data_loc, 'train_data.pkl'))
    sre16_enroll = load_object(join_path(data_loc, 'sre16_enroll.pkl'))
    sre16_test = load_object(join_path(data_loc, 'sre16_test.pkl'))
    logger.end_timer('Load:')


if args.stage <= 1:
    logger.start_timer('Stage 1: Feature Extraction.')
    logger.info('Stage 1: Generating data scp file...')
    generate_data_scp(args.save, train_data)
    generate_data_scp(args.save, sre16_enroll, append=True)
    generate_data_scp(args.save, sre16_test, append=True)
    logger.info('Stage 1: Removing already present files from data scp file...')
    count = remove_present_from_scp(args.save, n_jobs=args.num_jobs)
    logger.info('Stage 1: Saved data scp file at: {}'.format(data_loc))

    if count > 0:
        n_jobs = count if count < args.num_jobs else args.num_jobs
        mfcc = MFCC(fs=args.sample_rate, fl=20, fh=3700, frame_len_ms=25, n_ceps=args.num_features, n_jobs=n_jobs,
                    save_loc=args.save)
        vad = VAD(n_jobs=n_jobs, save_loc=args.save)
        logger.info('Stage 1: Extracting raw mfcc features for {} file(s)...'.format(count))
        data_scp_file = join_path(args.save, DATA_SCP_FILE)
        mfcc.extract(data_scp_file)
        logger.info('Stage 1: Computing VAD...')
        feats_scp_file = join_path(args.save, FEATS_SCP_FILE)
        vad.compute(feats_scp_file)
        logger.info('Stage 1: Applying VAD...')
        vad_scp_file = join_path(args.save, VAD_SCP_FILE)
        mfcc.apply_vad_and_save(feats_scp_file, vad_scp_file)
    else:
        logger.info('Stage 1: All files has features.')

    logger.info('Stage 1: Appending Frame counts..')
    frames = get_mfcc_frames(args.save, train_data[:, 0], args.num_jobs)
    train_data = np.hstack([train_data, frames])
    frames = get_mfcc_frames(args.save, sre16_enroll[:, 0], args.num_jobs)
    sre16_enroll = np.hstack([sre16_enroll, frames])
    frames = get_mfcc_frames(args.save, sre16_test[:, 0], args.num_jobs)
    sre16_test = np.hstack([sre16_test, frames])
    logger.info('Stage 1: Updating data lists..')
    save_object(join_path(data_loc, 'train_data.pkl'), train_data)
    save_object(join_path(data_loc, 'sre16_enroll.pkl'), sre16_enroll)
    save_object(join_path(data_loc, 'sre16_test.pkl'), sre16_test)
    logger.info('Stage 1: Data lists saved at: {}'.format(data_loc))
    logger.end_timer()
elif args.stage < 4 and not args.skip_check:
    logger.start_timer('Check: Looking for features...')
    _, fail1 = check_mfcc(args.save, train_data)
    _, fail2 = check_mfcc(args.save, sre16_enroll)
    _, fail3 = check_mfcc(args.save, sre16_test)
    fail = fail1 + fail2 + fail3
    if fail > 0:
        raise Exception('No features for {:d} file(s). Execute Stage 1 before proceeding.'.format(fail))
    else:
        if train_data.shape[1] < 6:
            logger.info('Check: Fetching train_data frames counts...')
            frames = get_mfcc_frames(args.save, train_data[:, 0], args.num_jobs)
            logger.info('Check: Appending and saving...')
            train_data = np.hstack([train_data, frames])
            save_object(join_path(args.save, 'train_data.pkl'), train_data)
        if sre16_enroll.shape[1] < 6:
            logger.info('Check: Fetching sre16_enroll frames counts...')
            frames = get_mfcc_frames(args.save, sre16_enroll[:, 0], args.num_jobs)
            logger.info('Check: Appending and saving...')
            sre16_enroll = np.hstack([sre16_enroll, frames])
            save_object(join_path(args.save, 'sre16_enroll.pkl'), sre16_enroll)
        if sre16_test.shape[1] < 8:
            logger.info('Check: Fetching sre16_test frames counts...')
            frames = get_mfcc_frames(args.save, sre16_test[:, 0], args.num_jobs)
            logger.info('Check: Appending and saving...')
            sre16_test = np.hstack([sre16_test, frames])
            save_object(join_path(args.save, 'sre16_test.pkl'), sre16_test)
    logger.end_timer('Check:')


if args.stage <= 2:
    logger.start_timer('Stage 2: Pre-processing...')
    logger.info('Stage 2: Filtering out short duration utterances and sorting by duration...')
    train_data = train_data[np.array(train_data[:, -1], dtype=int) >= 300]
    train_data = train_data[np.argsort(np.array(train_data[:, -1], dtype=int))]
    logger.info('Stage 2: Filtering out speakers having lesser training data...')
    speakers = train_data[:, 3]
    unique_speakers = set(speakers)
    logger.info('Stage 2: Total Speakers before filtering: {:d}'.format(len(unique_speakers)))
    speaker_counter = Counter(speakers)
    good_speakers = []
    for speaker in unique_speakers:
        if speaker_counter[speaker] >= 5:
            good_speakers.append(speaker)
    train_data = train_data[np.in1d(speakers, good_speakers), :]
    speakers = train_data[:, 3]
    unique_speakers = set(speakers)
    logger.info('Stage 2: Total Speakers after filtering: {:d}'.format(len(unique_speakers)))
    logger.info('Stage 2: Training Utterances after filtering: {:d}'.format(train_data.shape[0]))
    save_object(join_path(args.save, 'train_data.pkl'), train_data)

    logger.info('Stage 2: Making Speaker dictionaries...')
    n_speakers = len(unique_speakers)
    speaker_to_idx = dict(zip(unique_speakers, range(n_speakers)))
    idx_to_speaker = dict(zip(range(n_speakers), unique_speakers))
    train_data[:, 3] = np.array([speaker_to_idx[s] for s in speakers])
    save_object(join_path(args.save, 'speaker_to_idx.pkl'), speaker_to_idx)
    save_object(join_path(args.save, 'idx_to_speaker.pkl'), idx_to_speaker)
    logger.end_timer('Stage 2:')
else:
    logger.start_timer('Load: Speaker dictionaries from: {}'.format(args.save))
    speakers = train_data[:, 3]
    n_speakers = len(set(speakers))
    speaker_to_idx = load_object(join_path(args.save, 'speaker_to_idx.pkl'))
    idx_to_speaker = load_object(join_path(args.save, 'idx_to_speaker.pkl'))
    train_data[:, 3] = np.array([speaker_to_idx[s] for s in speakers])
    logger.end_timer('Load:')


if args.stage <= 4:
    model = HGRUTripletModel(n_features=args.num_features, n_classes=n_speakers, attention=False)
    if args.stage <= 3:
        logger.start_timer('Stage 3: Train Neural Net.')
        logger.info('Stage 3: Initializing batch loader...')
        # batch_loader = SREFixedLoader(location=args.save, args=train_data, n_features=args.num_features,
        #                               batch_size=args.batch_size, duration=300, stride=200, n_jobs=args.num_jobs)
        batch_loader = SRESplitBatchLoader(location=args.save, args=train_data, n_features=args.num_features,
                                           splits=[300, 1000, 3000, 6000],
                                           batch_size=[args.batch_size * 10, args.batch_size * 3, args.batch_size])
        logger.info('Stage 3: Training model...')
        model.start_train_with_splits(args.save, batch_loader, args.epochs, args.lr, args.decay, cont=False)
        logger.end_timer('Stage 3:')

    logger.start_timer('Stage 4: Embedding Extraction.')
    logger.info('Stage 4: Processing train_data...')
    train_loader = SRETestLoader(args.save, train_data, args.num_features, batch_size=args.extract_batch_size)
    model.extract(args.save, train_loader)
    logger.info('Stage 4: Processing sre16_enroll...')
    enroll_loader = SRETestLoader(args.save, sre16_enroll, args.num_features, batch_size=args.extract_batch_size)
    model.extract(args.save, enroll_loader)
    save_object(join_path(args.save, 'sre16_enroll.pkl'), sre16_enroll[:last_idx, :])
    logger.info('Stage 4: Processing sre16_test...')
    test_loader = SRETestLoader(args.save, sre16_test, args.num_features, batch_size=args.extract_batch_size)
    model.extract(args.save, test_loader)
    save_object(join_path(args.save, 'sre16_test.pkl'), sre16_test[:last_idx, :])
    logger.end_timer('Stage 4:')
elif not args.skip_check:
    _, fail1 = check_embeddings(args.save, sre16_enroll)
    _, fail2 = check_embeddings(args.save, sre16_test)
    fail = fail1 + fail2
    if fail > 0:
        print('No embeddings for {:d} files. Execute Stage 4 before proceeding.'.format(fail))

if args.stage <= 5:
    logger.start_timer('Stage 5: PLDA Training...')
    plda = PLDA(save_loc=args.save)
    plda.fit(train_data[:, 0], train_data[:, 3])
    logger.end_timer('Stage 5:')


if args.stage <= 6:
    logger.start_timer('Stage 6: PLDA Scoring...')
    logger.end_timer('Stage 6:')
