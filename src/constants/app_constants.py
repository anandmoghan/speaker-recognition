from os.path import join as join_path

KALDI_PATH_FILE = './path.sh'

DATA_DIR = 'data'
EMB_DIR = 'embeddings'
MFCC_DIR = 'mfcc'
MODELS_DIR = 'models'
VAD_DIR = 'vad'

LATEST_MODEL_FILE = 'latest.json'

DATA_SCP_FILE = join_path(DATA_DIR, 'data.scp')
FEATS_SCP_FILE = join_path(MFCC_DIR, 'feats.scp')
VAD_SCP_FILE = join_path(VAD_DIR, 'vad.scp')
