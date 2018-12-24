from os.path import abspath, join

from constants.app_constants import EGS_DIR, MODELS_DIR, NUM_CLASSES, NUM_EGS, NUM_FEATURES, TMP_DIR
from input_model import get_model
from services.common import make_directory, use_gpu, save_array
from services.kaldi import parse_egs_scp, read_egs

import tensorflow as tf
import argparse as ap
import numpy as np
import time

parser = ap.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64, help='Batch Size')
parser.add_argument('--cont', action='store_true', help='Continue Training')
parser.add_argument('--decay', type=float, default=0.9, help='Decay Rate')
parser.add_argument('--epochs', type=int, default=3, help='Number of Epochs')
parser.add_argument('--gpu', type=int, default=0, help='Select GPU')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate')
parser.add_argument('--model-tag', default='XVECTOR', help='Model Tag')
parser.add_argument('--num-classes', type=int, default=NUM_CLASSES, help='Number of classification labels')
parser.add_argument('--num-egs', type=int, default=NUM_EGS, help='Number of Example Files')
parser.add_argument('--num-features', type=int, default=NUM_FEATURES, help='Dimension of input features')
parser.add_argument('--ps', help='Parameter Server(s)')
parser.add_argument('--save', default='../save', help='Save Location')
parser.add_argument('--steps', type=int, default=500000, help='Total global steps')
parser.add_argument('--type', default='ps', help='Instance Type')
parser.add_argument('--task-index', type=int, default=0, help='Task Index')
parser.add_argument('--workers', help='Worker Nodes')
args = parser.parse_args()

save_loc = abspath(args.save)
egs_loc = join(save_loc, EGS_DIR)
model_loc = join(save_loc, MODELS_DIR)
make_directory(model_loc)
model_loc = join(model_loc, args.model_tag)
make_directory(model_loc)

early_stop_file = join(model_loc, 'early_stop')

tmp_loc = join(save_loc, TMP_DIR)
make_directory(tmp_loc)

ps = args.ps.split(',')
workers = args.workers.split(',')
num_workers = len(workers)

use_gpu(args.gpu)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def get_batch(batch_list):
    labels = []
    scp_file = join(tmp_loc, 'read_egs.{}.scp'.format(args.task_index))
    with open(scp_file, 'w') as f:
        for utt, ark, label in batch_list:
            f.write('{} {}\n'.format(utt, ark))
            labels.append(label)

    egs = read_egs(scp_file, args.num_features)
    return np.array(egs), np.array(labels, dtype=int)


def get_diagnostic_loss(diagnostic_data, model_, sess):
    diagnostic_loss = 0
    diagnostic_count = 0
    global_step_ = model_.get_global_step(sess)
    for diagnostic_batch, diagnostic_labels in diagnostic_data:
        diagnostic_batch_size = diagnostic_batch.shape[0]
        loss_, global_step_ = model_.compute_loss(diagnostic_batch, diagnostic_labels, sess)
        diagnostic_loss += loss_ * diagnostic_batch_size
        diagnostic_count += diagnostic_batch_size

    return diagnostic_loss / diagnostic_count, global_step_


cluster = tf.train.ClusterSpec({"ps": ps, "worker": workers})
server = tf.train.Server(cluster, job_name=args.type, task_index=args.task_index)

if args.type == 'ps':
    print('Running on {} as parameter server.'.format(ps[args.task_index]))
    server.join()
elif args.type == 'worker':
    print('Running on {} as worker node.'.format(workers[args.task_index]))

    batch_index = 0
    current_index = 0
    current_lr = args.lr
    current_egs = np.arange(args.task_index, args.num_egs, num_workers, dtype=int) + 1
    current_egs = current_egs[current_egs <= args.num_egs]

    diagnostic_egs = [1, 2, 3]
    train_diagnostic = [get_batch(parse_egs_scp(join(egs_loc, 'train_diagnostic_egs.{}.scp'.format(d)), shuffle_egs=False))
                        for d in diagnostic_egs]
    val_diagnostic = [get_batch(parse_egs_scp(join(egs_loc, 'valid_egs.{}.scp'.format(d)), shuffle_egs=False)) for d in
                      diagnostic_egs]

    val_loss = np.ones([5, 1]) * np.finfo(float).max
    save_array(early_stop_file, val_loss)

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:{:d}".format(args.task_index),
                                                  cluster=cluster)):
        model = get_model(args.num_features, args.num_classes, args.model_tag)

    hooks = [tf.train.StopAtStepHook(last_step=args.steps)]
    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(args.task_index == 0),
                                           checkpoint_dir=model_loc, hooks=hooks, config=config) as mon_sess:
        egs_data = []
        egs_batches = 0
        num_batches = 0
        global_step = 0
        last_check = 0
        while not mon_sess.should_stop():
            start_time = time.time()
            if batch_index == 0:
                egs_file = join(egs_loc, 'egs.{}.scp').format(current_egs[current_index])
                print('Reading from: {}'.format(egs_file))
                egs_data = parse_egs_scp(egs_file, shuffle_egs=True)
                num_batches = int(len(egs_data) / args.batch_size)
                egs_batches = np.array_split(egs_data, num_batches)
                current_index = (current_index + 1) if current_index < len(current_egs) - 1 else 0

            current_batch = egs_batches[batch_index]
            batch_index = (batch_index + 1) if batch_index < num_batches - 1 else 0
            batch_x, batch_y = get_batch(current_batch)

            loss, global_step = model.train_step(batch_x, batch_y, current_lr, mon_sess)

            current_lr = args.lr * args.decay ** int(global_step * 10 / args.steps)

            end_time = time.time()
            print('Global Step: {} | Loss: {:.3f} | Time Elapsed: {:.1f} seconds'.format(global_step, loss,
                                                                                         end_time - start_time))
            if global_step - last_check > 5000:
                train_diagnostic_loss, global_step = get_diagnostic_loss(train_diagnostic, model, mon_sess)
                print('Train Diagnostic Loss at {}th step: {:.3f}'.format(global_step, train_diagnostic_loss))
                val_diagnostic_loss, global_step = get_diagnostic_loss(val_diagnostic, model, mon_sess)
                print('Validation Loss at {}th step: {:.3f}'.format(global_step, val_diagnostic_loss))

                last_check = global_step
                idx = np.argmax(val_loss)
                if val_loss[idx] >= val_diagnostic_loss:
                    val_loss[idx] = val_diagnostic_loss
                    save_array(early_stop_file, val_loss)
                elif val_diagnostic_loss > val_loss[idx] + 0.2:
                    mon_sess.close()
                    break
