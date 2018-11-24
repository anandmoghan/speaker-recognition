from os.path import abspath, join

from constants.app_constants import EGS_DIR, MODELS_DIR, TMP_DIR
from models.x_vector import XVectorModel
from services.common import make_directory, use_gpu
from services.kaldi import parse_egs_scp, read_egs
from services.logger import Logger

import tensorflow as tf
import argparse as ap
import numpy as np
import time

parser = ap.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, help='Batch Size')
parser.add_argument('--cont', action='store_true', help='Continue Training')
parser.add_argument('--decay', type=float, default=0.8, help='Decay Rate')
parser.add_argument('--epochs', type=int, default=20, help='Number of Epochs')
parser.add_argument('--gpu', type=int, default=0, help='Select GPU')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--model-tag', default='XVECTOR', help='Model Tag')
parser.add_argument('--num-features', type=int, default=23, help='Number of MFCC Co-efficients')
parser.add_argument('--ps', help='Parameter Server(s)')
parser.add_argument('--save', default='../save', help='Save Location')
parser.add_argument('--steps', type=int, default=200000, help='Total global steps')
parser.add_argument('--type', default='ps', help='Instance Type')
parser.add_argument('--task-index', type=int, default=0, help='Task Index')
parser.add_argument('--workers', help='Worker Nodes')
args = parser.parse_args()

logger = Logger()
logger.set_config(filename='../logs/run-model.log', append=False)

save_loc = abspath(args.save)
egs_loc = join(save_loc, EGS_DIR)
model_loc = join(save_loc, MODELS_DIR)
make_directory(model_loc)
model_loc = join(model_loc, args.model_tag)
make_directory(model_loc)

tmp_loc = join(save_loc, TMP_DIR)
make_directory(tmp_loc)

ps = args.ps.split(',')
workers = args.workers.split(',')
num_workers = len(workers)

use_gpu(args.gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

cluster = tf.train.ClusterSpec({"ps": ps, "worker": workers})
server = tf.train.Server(cluster, job_name=args.type, task_index=args.task_index)

if args.type == 'ps':
    print('Running on {} as parameter server.'.format(ps[args.task_index]))
    server.join()
elif args.type == 'worker':
    print('Running on {} as worker node.'.format(workers[args.task_index]))
    total_egs = 64
    total_classes = 3769

    batch_index = 0
    current_index = 0
    current_lr = args.lr
    current_egs = np.array_split(np.arange(1, total_egs + 1, dtype=int), num_workers)[args.task_index]

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:{:d}".format(args.task_index), cluster=cluster)):
        model = XVectorModel(args.num_features, total_classes, attention=False, model_tag=args.model_tag)

    hooks = [tf.train.StopAtStepHook(last_step=args.steps)]
    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(args.task_index == 0),
                                           checkpoint_dir=model_loc, hooks=hooks, config=config) as mon_sess:
        egs_data = []
        egs_batches = 0
        num_batches = 0
        scp_file = join(tmp_loc, 'read_egs.{}.scp'.format(args.task_index))
        while not mon_sess.should_stop():
            start_time = time.time()
            if batch_index == 0:
                egs_data = parse_egs_scp(current_egs[current_index], save_loc)
                num_batches = int(len(egs_data) / args.batch_size)
                egs_batches = np.array_split(egs_data, num_batches)
                current_index = (current_index + 1) if current_index < len(current_egs) - 1 else 0

            current_batch = egs_batches[batch_index]
            batch_index = (batch_index + 1) if batch_index < num_batches - 1 else 0

            batch_y = []
            with open(scp_file, 'w') as f:
                for utt, ark, label in current_batch:
                    f.write('{} {}\n'.format(utt, ark))
                    batch_y.append(label)

            batch_x = read_egs(scp_file, args.num_features)
            batch_y = np.array(batch_y)

            loss, global_step, _ = mon_sess.run([model.loss, model.global_step, model.optimizer], feed_dict={
                model.input_: batch_x,
                model.labels: batch_y,
                model.batch_size: batch_x.shape[0],
                model.lr: current_lr
            })

            # current_lr *= args.decay ** int(global_step * 10 / args.steps)

            end_time = time.time()
            print('Global Step: {} | Loss: {:.3f} | Time Elapsed: {} seconds'.format(global_step, loss,
                                                                                     int(end_time - start_time)))
