from os.path import abspath, join as join_path

import argparse as ap

from constants.app_constants import DATA_DIR, NUM_FEATURES, SAVE_LOC, TMP_DIR
from kaldi.split_scp import split_scp
from services.common import make_directory
from services.distributed import submit_extract_worker_job, watch_jobs

parser = ap.ArgumentParser()
parser.add_argument('--iteration', type=int, default=72, help='Saved model iteration.')
parser.add_argument('--max-chunk-size', type=int, default=5000, help='Max chunk size of utterance, after which it will be averaged.')
parser.add_argument('--model-tag', default='HGRU', help='Model Tag')
parser.add_argument('--num-features', type=int, default=NUM_FEATURES, help='Number of MFCC Co-efficients')
parser.add_argument('--save', default=SAVE_LOC, help='Save Location')
parser.add_argument('--splits', default="train_data_full,sre_unlabelled,sre_dev_enroll,sre_dev_test,sre_eval_enroll,sre_eval_test", help='Splits')
parser.add_argument('--num-workers', type=int, default=4, help='Number of Workers')
args = parser.parse_args()

save_loc = abspath(args.save)
data_loc = join_path(save_loc, DATA_DIR)
tmp_loc = join_path(save_loc, '{}/{}'.format(TMP_DIR, args.model_tag))
make_directory(tmp_loc)

splits = args.splits.split(',')

for split in splits:
    print('Extracting embeddings for {} from {} model.'.format(split, args.model_tag))
    split_loc = join_path(data_loc, split)
    feats_scp = join_path(split_loc, 'voiced_feats.scp')
    split_scp(feats_scp, args.num_workers, tmp_loc, prefix='feats.{}'.format(split))
    job_ids = []
    for worker_id in range(args.num_workers):
        split_feats_scp = join_path(tmp_loc, 'feats.{}.{}.scp'.format(split, worker_id + 1))
        job_id = submit_extract_worker_job(args.model_tag, args.iteration, worker_id, split_feats_scp, args.max_chunk_size, save_loc, compute=None)
        print('Submitted job {} to worker {}.'.format(job_id, worker_id))
        job_ids.append(job_id)
    watch_jobs(job_ids)
