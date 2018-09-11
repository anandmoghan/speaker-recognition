from collections import Counter
from os.path import join as join_path, abspath

import argparse as ap
import numpy as np

from constants.app_constants import DATA_DIR, DATA_SCP_FILE, TRAIN_SPLIT, ENROLL_SPLIT, TEST_SPLIT, UNLABELLED_SPLIT, \
    TRIALS_FILE
from models.x_vector import XVectorModel
from services.checks import check_mfcc, check_embeddings
from services.common import create_directories, load_object, save_object, tensorflow_debug, use_gpu
from services.feature import MFCC, generate_data_scp, get_mfcc_frames, remove_bad_files
from services.kaldi import PLDA, convert_embeddings
from services.loader import SRESplitKaldiBatchLoader, SREKaldiTestLoader
from services.logger import Logger
from services.sre_data import get_train_data, make_sre16_eval_data, make_sre16_unlabelled_data, make_sre16_trials_file

DATA_CONFIG = '../configs/sre_data.json'

parser = ap.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, help='Training Batch Size')
parser.add_argument('--cont', action='store_true', help='Continue Training')
parser.add_argument('--decay', type=float, default=0.2, help='Decay Rate')
parser.add_argument('--epochs', type=int, default=3, help='Number of Epochs')
parser.add_argument('--extract-batch-size', type=int, default=56, help='Extract Batch Size')
parser.add_argument('--gpu', type=int, default=0, help='Select GPU')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--model-tag', default='XVECTOR', help='Model Tag')
parser.add_argument('--num-features', type=int, default=20, help='Number of MFCC Co-efficients')
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
    sre_unlabelled = make_sre16_unlabelled_data(DATA_CONFIG)
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
            save_object(join_path(args.save, 'train_data.pkl'), train_data)
        if sre_unlabelled.shape[1] < 6:
            logger.info('Check: Fetching sre_unlabelled frames counts...')
            frames = get_mfcc_frames(args.save, sre_unlabelled[:, 0])
            logger.info('Check: Appending and saving...')
            sre_unlabelled = np.hstack([sre_unlabelled, frames])
            save_object(join_path(args.save, 'sre_unlabelled.pkl'), sre_unlabelled)
        if sre_enroll.shape[1] < 6:
            logger.info('Check: Fetching sre_enroll frames counts...')
            frames = get_mfcc_frames(args.save, sre_enroll[:, 0])
            logger.info('Check: Appending and saving...')
            sre_enroll = np.hstack([sre_enroll, frames])
            save_object(join_path(args.save, 'sre_enroll.pkl'), sre_enroll)
        if sre_test.shape[1] < 8:
            logger.info('Check: Fetching sre_test frames counts...')
            frames = get_mfcc_frames(args.save, sre_test[:, 0])
            logger.info('Check: Appending and saving...')
            sre_test = np.hstack([sre_test, frames])
            save_object(join_path(args.save, 'sre_test.pkl'), sre_test)
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
    save_object(join_path(data_loc, 'train_data.pkl'), train_data)

    logger.info('Stage 2: Making Speaker dictionaries...')
    n_speakers = len(unique_speakers)
    speaker_to_idx = dict(zip(unique_speakers, range(n_speakers)))
    idx_to_speaker = dict(zip(range(n_speakers), unique_speakers))
    train_data[:, 3] = np.array([speaker_to_idx[s] for s in speakers])
    save_object(join_path(args.save, 'speaker_to_idx.pkl'), speaker_to_idx)
    save_object(join_path(args.save, 'idx_to_speaker.pkl'), idx_to_speaker)
    logger.end_timer('Stage 2:')
else:
    logger.start_timer('Load: Loading speaker dictionaries...')
    speaker_to_idx = load_object(join_path(args.save, 'speaker_to_idx.pkl'))
    idx_to_speaker = load_object(join_path(args.save, 'idx_to_speaker.pkl'))
    speakers = train_data[:, 3]
    train_data[:, 3] = [speaker_to_idx[s] for s in speakers]
    logger.end_timer('Load:')

if args.stage <= 4:
    n_speakers = len(set(train_data[:, 3]))
    model = XVectorModel(n_features=args.num_features, n_classes=n_speakers, attention=False, model_tag=args.model_tag)
    # model = HGRUTripletModel(n_features=args.num_features, n_classes=n_speakers, attention=True, model_tag=args.model_tag)
    if args.stage <= 3:
        logger.start_timer('Stage 3: Train Neural Net.')
        logger.info('Stage 3: Initializing batch loader...')
        batch_loader = SRESplitKaldiBatchLoader(location=args.save, args=train_data, n_features=args.num_features,
                                                splits=[200, 300, 400, 500], batch_size=args.batch_size)
        logger.info('Stage 3: Training model...')
        model.start_train_with_splits(args.save, batch_loader, args.epochs, args.lr, args.decay, cont=args.cont)
        logger.end_timer('Stage 3:')

    logger.start_timer('Stage 4: Embedding Extraction.')
    logger.info('Stage 4: Processing train_data...')
    train_loader = SREKaldiTestLoader(args.save, train_data, args.num_features, 10000, args.extract_batch_size,
                                      model_tag=args.model_tag)
    model.extract(args.save, train_loader)
    logger.info('Stage 4: Processing sre_unlabelled...')
    unlabelled_loader = SREKaldiTestLoader(args.save, sre_unlabelled, args.num_features, 10000, args.extract_batch_size,
                                           model_tag=args.model_tag)
    model.extract(args.save, unlabelled_loader)
    logger.info('Stage 4: Processing sre_enroll...')
    enroll_loader = SREKaldiTestLoader(args.save, sre_enroll, args.num_features, 10000, args.extract_batch_size,
                                       model_tag=args.model_tag)
    model.extract(args.save, enroll_loader)
    logger.info('Stage 4: Processing sre_test...')
    test_loader = SREKaldiTestLoader(args.save, sre_test, args.num_features, 10000, args.extract_batch_size,
                                     model_tag=args.model_tag)
    model.extract(args.save, test_loader)
    logger.end_timer('Stage 4:')
elif not args.skip_check and args.stage < 6:
    logger.start_timer('Check: Looking for Embeddings...')
    _, fail1 = check_embeddings(args.save, args.model_tag, train_data)
    _, fail2 = check_embeddings(args.save, args.model_tag, sre_unlabelled)
    _, fail3 = check_embeddings(args.save, args.model_tag, sre_enroll)
    _, fail4 = check_embeddings(args.save, args.model_tag, sre_test)
    fail = fail1 + fail2 + fail3 + fail4
    if fail > 0:
        logger.info('No embeddings for {:d} files. Execute Stage 4 before proceeding.'.format(fail))
    logger.end_timer('Check:')

if args.stage <= 5:
    logger.start_timer('Stage 5: Kaldi ARK embedding conversion.')
    logger.info('Stage 5: Processing train_data embeddings...')
    convert_embeddings(train_data[:, 0], args.model_tag, split=TRAIN_SPLIT, save_loc=args.save, n_jobs=args.num_jobs)
    logger.info('Stage 5: Processing sre_unlabelled embeddings...')
    convert_embeddings(sre_unlabelled[:, 0], args.model_tag, split=UNLABELLED_SPLIT, save_loc=args.save,
                       n_jobs=args.num_jobs)
    logger.info('Stage 5: Processing sre_enroll embeddings...')
    convert_embeddings(sre_enroll[:, 0], args.model_tag, split=ENROLL_SPLIT, save_loc=args.save, n_jobs=args.num_jobs)
    logger.info('Stage 5: Processing sre_test embeddings...')
    convert_embeddings(sre_test[:, 0], args.model_tag, split=TEST_SPLIT, save_loc=args.save, n_jobs=args.num_jobs)
    logger.end_timer('Stage 5:')

if args.stage <= 6:
    logger.start_timer('Stage 6: PLDA Training.')
    plda = PLDA(model_tag=args.model_tag, save_loc=args.save)
    plda.fit(train_data[:, 0], train_data[:, 3])
    logger.end_timer('Stage 6:')

if args.stage <= 7:
    logger.start_timer('Stage 7: PLDA Scoring.')
    plda = PLDA(model_tag=args.model_tag, save_loc=args.save)
    eer = plda.compute_eer(sre_enroll[:, 0], sre_enroll[:, 3], test_split=TEST_SPLIT)
    logger.info('Net EER: {}'.format(eer))
    logger.end_timer('Stage 7:')
