from os.path import join as join_path

import tensorflow as tf
import argparse as ap
import sys

from constants.app_constants import NUM_CLASSES, NUM_FEATURES
from constants.tf_constants import LATEST_CHECKPOINT, MODEL_CHECKPOINT
from input_model import get_model
from services.common import get_time_stamp, print_script_args, use_gpu

parser = ap.ArgumentParser()
parser.add_argument('--gpu', type=int, default=-1, help='Selects GPU.')
parser.add_argument('--model-path', help='Model save path.')
parser.add_argument('--model-tag', help='Model Tag')
parser.add_argument('--num-classes', type=int, default=NUM_CLASSES, help='Number of classification labels.')
parser.add_argument('--num-features', type=int, default=NUM_FEATURES, help='Dimension of input features.')
args = parser.parse_args()

print_script_args(sys.argv)
print('Started at: {}\n'.format(get_time_stamp()))

use_gpu(args.gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = get_model(args.num_features, args.num_classes, args.model_tag)
model_path = join_path(args.model_path, MODEL_CHECKPOINT)

with tf.Session(config=config) as sess:
    print('Initializing global variables.')
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print('Saving model parameters.')
    saver.save(sess, model_path, latest_filename=LATEST_CHECKPOINT)
    print('Saved at {}'.format(model_path))

print('\nFinished at: {}'.format(get_time_stamp()))
