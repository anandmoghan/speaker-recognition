from os.path import abspath, join as join_path

from constants.app_constants import TMP_DIR, EMB_DIR
from constants.tf_constants import LATEST_CHECKPOINT
from input_model import get_model
from services.common import make_directory, use_gpu, save_array, run_parallel
from services.kaldi import read_feat

import tensorflow as tf
import argparse as ap
import numpy as np

parser = ap.ArgumentParser()
parser.add_argument('--feats-scp', help='Feats scp file.')
parser.add_argument('--gpu', type=int, default=-1, help='Select GPU')
parser.add_argument('--max-chunk-size', type=int, default=5000, help='Max chunk size of utterance, after which it will be averaged.')
parser.add_argument('--model-path', help='Model path.')
parser.add_argument('--model-tag', default='XVECTOR', help='Model Tag')
parser.add_argument('--num-features', type=int, default=23, help='Number of MFCC Co-efficients')
parser.add_argument('--num-classes', type=int, default=3769, help='Number of MFCC Co-efficients')
parser.add_argument('--num-workers', type=int, default=10, help='Number of Workers')
parser.add_argument('--save', default='../save', help='Save Location')
parser.add_argument('--worker-id', type=int, default=0, help='Worker Id')
args = parser.parse_args()

save_loc = abspath(args.save)
embedding_loc = join_path(save_loc, '{}/{}'.format(EMB_DIR, args.model_tag))
make_directory(embedding_loc)

tmp_loc = join_path(save_loc, '{}/{}'.format(TMP_DIR, args.model_tag))
make_directory(tmp_loc)

read_scp = join_path(tmp_loc, 'extract_read.{}'.format(args.worker_id) + '.{}.scp')


def get_batch(items):
    utt, ark, batch_id = items
    scp_file = read_scp.format(batch_id)

    with open(scp_file, 'w') as scp:
        scp.write('{} {}'.format(utt, ark))

    feat = read_feat(scp_file, args.num_features)
    num_frames = feat.shape[1]

    if num_frames > args.max_chunk_size:
        num_splits = int(num_frames / args.max_chunk_size) * 2
        num_frames = int(num_frames / num_splits) * num_splits
        feats = np.array_split(feat[:, :num_frames], num_splits, axis=1)
    else:
        feats = [feat]

    return np.array(feats), utt


feats_list = []
with open(args.feats_scp) as f:
    for i, line in enumerate(f.readlines()):
        tokens = line.strip().split()
        feats_list.append((tokens[0], tokens[1], i))

num_batches = len(feats_list)

print('Loading batches from: {}'.format(args.feats_scp))
batches = run_parallel(get_batch, feats_list, n_workers=args.num_workers, p_bar=False)

use_gpu(args.gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = get_model(args.num_features, args.num_classes, args.model_tag)

with tf.Session(config=config) as sess:
    model_path = tf.train.latest_checkpoint(args.model_path, latest_filename=LATEST_CHECKPOINT)
    print('Restoring model: {}'.format(model_path))
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    for b in range(num_batches):
        batch_x, key = batches[b]
        emb = sess.run(model.embeddings, feed_dict={
            model.input_: batch_x,
            model.batch_size: batch_x.shape[0]
        })

        emb = np.mean(emb, axis=0)

        embedding_file = join_path(embedding_loc, '{}.npy'.format(key))
        save_array(embedding_file, emb)
        print('{}/{}: Saved embedding for {}'.format(b + 1, num_batches, key))
