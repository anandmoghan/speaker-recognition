from os.path import join as join_path

KALDI_PATH_FILE = './kaldi/path.sh'
KALDI_QUEUE_FILE = './kaldi/queue.pl'

DATA_DIR = 'data'
EMB_DIR = 'embeddings'
EGS_DIR = 'egs'
LOGS_DIR = 'logs'
MFCC_DIR = 'mfcc'
MODELS_DIR = 'models'
PLDA_DIR = 'plda'
VAD_DIR = 'vad'
TMP_DIR = 'tmp'

TRAIN_SPLIT = 'train'
DEV_SPLIT = 'dev'
ENROLL_SPLIT = 'enroll'
UNLABELLED_SPLIT = 'unlabelled'
TEST_SPLIT = 'test'

LATEST_MODEL_FILE = 'latest.json'

DATA_SCP_FILE = join_path(DATA_DIR, 'data.scp')
SPK_EMB_SCP_FILE = join_path(DATA_DIR, 'spk_xvector.scp')
EMB_SCP_FILE = join_path(DATA_DIR, 'embeddings.scp')
ENROLL_SPK_EMB_SCP_FILE = join_path(DATA_DIR, 'enroll_spk_xvector.scp')
TMP_SCP_FILE = join_path(TMP_DIR, 'tmp_{}.scp')

FEATS_SCP_FILE = 'feats.scp'
UTT2NUM_FRAMES_FILE = 'utt2num_frames'
VAD_SCP_FILE = 'vad.scp'

NUM_UTT_FILE = join_path(DATA_DIR, 'num_utt')
SPK_UTT_FILE = join_path(DATA_DIR, 'spk2utt')
TRIALS_FILE = join_path(DATA_DIR, 'trials')
UTT_SPK_FILE = join_path(DATA_DIR, 'utt2spk')

BATCH_LOADER_FILE = join_path(TMP_DIR, 'batch_loader_{}.pkl')

SCORES_FILE = 'score.txt'
EER_INPUT_FILE = 'eer_input.txt'
