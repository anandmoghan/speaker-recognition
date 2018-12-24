from os.path import join as join_path, abspath

import tensorflow as tf
import argparse as ap
import numpy as np
import sys
import os

from constants.app_constants import EGS_DIR, NUM_CLASSES, NUM_CPU_WORKERS, NUM_FEATURES, TMP_DIR, SAVE_LOC
from constants.tf_constants import LATEST_CHECKPOINT, MODEL_CHECKPOINT
from input_model import get_model
from services.common import get_time_stamp, make_directory, print_script_args, run_parallel, use_gpu
from services.distributed import get_model_path
from services.kaldi import parse_egs_scp, read_egs

parser = ap.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, help='Batch Size')
parser.add_argument('--egs-index', type=int, default=1, help='Example file to process')
parser.add_argument('--gpu', type=int, default=-1, help='Select GPU')
parser.add_argument('--iteration', type=int, default=1, help='Current Iteration')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--meta-graph', help='Meta graph path.')
parser.add_argument('--model-tag', help='Model Tag')
parser.add_argument('--num-classes', type=int, default=NUM_CLASSES, help='Number of classification labels.')
parser.add_argument('--num-features', type=int, default=NUM_FEATURES, help='Dimension of input features.')
parser.add_argument('--num-jobs', type=int, default=NUM_CPU_WORKERS, help='Number of CPU Workers')
parser.add_argument('--save', default=SAVE_LOC, help='Save location.')
parser.add_argument('--worker-id', type=int, default=0, help='Worker Id')
args = parser.parse_args()

print_script_args(sys.argv)
print('Started at: {}\n'.format(get_time_stamp()))

save_loc = abspath(args.save)
tmp_loc = join_path(save_loc, '{}/{}'.format(TMP_DIR, args.model_tag))
make_directory(tmp_loc)

egs_scp = join_path(save_loc, '{}/egs.{}.scp'.format(EGS_DIR, args.egs_index))
read_scp = join_path(tmp_loc, 'read_egs.{}.'.format(args.worker_id) + '{}.scp')

initial_path = get_model_path(args.iteration - 1, args.model_tag, save_loc)
model_path = get_model_path(args.iteration, args.model_tag, save_loc, args.worker_id)


def get_batch(items):
    batch_list, batch_id = items
    labels = []
    scp_file = read_scp.format(batch_id)
    with open(scp_file, 'w') as f:
        for utt, ark, l in batch_list:
            f.write('{} {}\n'.format(utt, ark))
            labels.append(l)

    egs_feats = read_egs(scp_file, args.num_features)
    os.remove(scp_file)
    return np.array(egs_feats), np.array(labels, dtype=int)


model = get_model(args.num_features, args.num_classes, args.model_tag)

egs_data = parse_egs_scp(egs_scp, shuffle_egs=True)
num_batches = int(len(egs_data) / args.batch_size)
egs_batches_list = list(zip(np.array_split(egs_data, num_batches), range(num_batches)))

print('Loading batches from: {}'.format(egs_scp))
egs_batches = run_parallel(get_batch, egs_batches_list, n_workers=args.num_jobs, p_bar=False)

use_gpu(args.gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    print('Restoring model parameters: {}'.format(initial_path))
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(initial_path, latest_filename=LATEST_CHECKPOINT))

    count = 0
    avg_loss = 0
    for b in range(num_batches):
        batch_x, batch_y = egs_batches[b]
        loss, global_step = model.train_step(batch_x, batch_y, args.lr, sess)
        count += 1
        avg_loss += loss
        if global_step % 50 == 0:
            print('Global Step: {} | Egs: {} | Loss: {:.3f}'.format(global_step, args.egs_index, avg_loss / count))
            count = 0
            avg_loss = 0

    print('Saving model parameters.')
    model_path = join_path(model_path, MODEL_CHECKPOINT)
    saver.save(sess, model_path, latest_filename=LATEST_CHECKPOINT)
    print('Saved at: {}'.format(model_path))

print('\nFinished at: {}'.format(get_time_stamp()))
