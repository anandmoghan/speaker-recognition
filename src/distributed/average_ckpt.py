from os.path import join as join_path

import tensorflow as tf
import argparse as ap
import numpy as np
import sys

from constants.tf_constants import LATEST_CHECKPOINT, MODEL_CHECKPOINT
from services.common import get_time_stamp, print_script_args, use_gpu


def average_from_paths(meta_graph_, checkpoints_, final_path_, sess_=None, ignore=None):
    print('Loading graph.')
    final_path_ = join_path(final_path_, MODEL_CHECKPOINT)
    saver = tf.train.import_meta_graph(meta_graph_)

    print('Fetching checkpoint values.')
    total_values = []
    for checkpoint in checkpoints_:
        print('Processing checkpoint from: {}'.format(checkpoint))
        saver.restore(sess_, checkpoint)
        var_values = get_var_values(tf.global_variables(), sess_, ignore=ignore)
        if len(total_values) > 0:
            for i in range(len(total_values)):
                total_values[i] += var_values[i]
        else:
            total_values = var_values

    print('Averaging the values.')
    tensor_list = tf.global_variables()
    for t, v in zip(tensor_list, total_values):
        d_type = t.dtype
        v = v / len(checkpoints_)
        if str(d_type) == 'tf.int32_ref':
            v = np.array(v, dtype=np.int32)
        elif str(d_type) == 'tf.int64_ref':
            v = np.array(v, dtype=np.int64)
        elif str(d_type) == 'tf.float32_ref':
            v = np.array(v, dtype=np.float32)
        elif str(d_type) == 'tf.float64_ref':
            v = np.array(v, dtype=np.float64)
        t.load(v, sess_)

    print('Saving the final model.')
    saver.save(sess_, final_path_, latest_filename=LATEST_CHECKPOINT)
    print('Saved at {}'.format(final_path_))


def average_from_latest_checkpoints(meta_graph_, checkpoint_dirs_, final_path_, sess_=None, ignore=None):
    checkpoints_ = [tf.train.latest_checkpoint(checkpoint_dir, latest_filename=LATEST_CHECKPOINT) for checkpoint_dir in checkpoint_dirs_]
    average_from_paths(meta_graph_, checkpoints_, final_path_, sess_, ignore)


def get_var_values(var, sess_, ignore=None):
    return [sess_.run(v) for v in var]


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help='Selects GPU. -1 to select the available gpu')
    parser.add_argument('--meta-graph', help='Meta graph file path')
    parser.add_argument('--checkpoint-dirs', help='Checkpoints directories to be averaged separated by commas')
    parser.add_argument('--final-path', help='Final checkpoint directory path')
    return parser.parse_args()


if __name__ == '__main__':
    print_script_args(sys.argv)
    print('Started at: {}\n'.format(get_time_stamp()))

    args = parse_args()
    use_gpu(args.gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    checkpoint_dirs = args.checkpoint_dirs.split(',')

    with tf.Session(config=config) as sess:
        average_from_latest_checkpoints(args.meta_graph, checkpoint_dirs, args.final_path, sess)

    print('\nFinished at: {}'.format(get_time_stamp()))
