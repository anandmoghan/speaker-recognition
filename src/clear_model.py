from os.path import abspath, join as join_path

import argparse as ap

from constants.app_constants import LOGS_DIR, MODELS_DIR, TMP_DIR, EMB_DIR, PLDA_DIR
from services.common import delete_directory

parser = ap.ArgumentParser()
parser.add_argument('--model-tag', help='Model Tag')
parser.add_argument('--save', default='../save', help='Save Location')
args = parser.parse_args()

save_loc = abspath(args.save)
emb_loc = join_path(save_loc, '{}/{}'.format(EMB_DIR, args.model_tag))
logs_loc = join_path(save_loc, '{}/{}'.format(LOGS_DIR, args.model_tag))
model_loc = join_path(save_loc, '{}/{}'.format(MODELS_DIR, args.model_tag))
plda_loc = join_path(save_loc, '{}/{}'.format(PLDA_DIR, args.model_tag))
tmp_loc = join_path(save_loc, '{}/{}'.format(TMP_DIR, args.model_tag))

print('Clearing model: {}'.format(args.model_tag))
print('Deleting embeddings..')
delete_directory(emb_loc)
print('Deleting logs..')
delete_directory(logs_loc)
print('Deleting saved models..')
delete_directory(model_loc)
print('Deleting plda..')
delete_directory(plda_loc)
print('Deleting tmp dirs..')
delete_directory(tmp_loc)
print('Finished.')
