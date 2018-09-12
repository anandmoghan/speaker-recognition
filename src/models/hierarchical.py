from os.path import join as join_path
from time import time as get_time

import tensorflow as tf
import numpy as np
import json

from constants.app_constants import EMB_DIR, LATEST_MODEL_FILE, MODELS_DIR
from models.layers.attention import variable_attention
from services.common import make_directory, save_batch_array, tensorflow_debug, use_gpu, split_args_list
from services.loader import SplitBatchLoader, FixedBatchLoader
from services.logger import Logger

tensorflow_debug(False)
use_gpu(0)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

HOP = 10
LAYER_1_HIDDEN_UNITS = 512
LAYER_2_HIDDEN_UNITS = 512
LAYER_3_HIDDEN_UNITS = 512
ATTENTION_SIZE = 256
EMBEDDING_SIZE = 512
DENSE_SIZE = 512

logger = Logger()
logger.set_config(filename='../logs/run-model.log', append=True)

MODEL_TAG = 'HGRUModel'


class HGRUModel:
    def __init__(self, n_features, n_classes, attention=False, model_tag=MODEL_TAG):
        self.n_classes = n_classes
        self.n_features = n_features
        self.model_tag = model_tag
        self.input_ = tf.placeholder(tf.float32, [None, n_features, None])
        self.labels = tf.placeholder(tf.int32, [None, ])
        self.batch_size = tf.Variable(32, dtype=tf.int32, trainable=False)
        self.lr = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        # Sequence information in not important. So, to improve performance, every HOP frames are given parallel.
        input_ = tf.transpose(self.input_, [0, 2, 1])
        input_ = tf.reshape(input_, [-1, HOP, n_features])

        with tf.variable_scope('layer_1'):
            rnn_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(LAYER_1_HIDDEN_UNITS), input_, dtype=tf.float32)
            if attention:
                rnn_output = variable_attention(rnn_output, size=ATTENTION_SIZE)
            else:
                rnn_output = rnn_output[:, -1, :]
            rnn_output = tf.reshape(rnn_output, [-1, HOP, LAYER_1_HIDDEN_UNITS])

        with tf.variable_scope('layer_2'):
            rnn_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(LAYER_2_HIDDEN_UNITS), rnn_output,
                                              dtype=tf.float32)
            if attention:
                rnn_output = variable_attention(rnn_output, size=ATTENTION_SIZE)
            else:
                rnn_output = rnn_output[:, -1, :]
            rnn_output = tf.reshape(rnn_output, [self.batch_size, -1, LAYER_2_HIDDEN_UNITS])

        with tf.variable_scope('layer_3'):
            rnn_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(LAYER_3_HIDDEN_UNITS), rnn_output,
                                              dtype=tf.float32)
            if attention:
                rnn_output = variable_attention(rnn_output, size=ATTENTION_SIZE)
            else:
                rnn_output = tf.reduce_mean(rnn_output, axis=1)

        with tf.variable_scope('layer_4'):
            self.embeddings = tf.layers.dense(rnn_output, EMBEDDING_SIZE, activation=None)
            dense_output = tf.nn.relu(self.embeddings)

        with tf.variable_scope('layer_5'):
            dense_output = tf.layers.dense(dense_output, DENSE_SIZE, activation=tf.nn.relu)

        self.logits = tf.layers.dense(dense_output, n_classes, activation=None)
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def extract(self, args_list, batch_size, save_loc='../save'):
        model_loc = join_path(join_path(save_loc, MODELS_DIR), self.model_tag)
        save_json = join_path(model_loc, LATEST_MODEL_FILE)
        with open(save_json, 'r') as f:
            model_json = json.load(f)
        model_path = join_path(model_loc, '{}_Epoch{:d}_Batch{:d}_Loss{:.2f}.ckpt'
                               .format(self.model_tag, model_json['e'] + 1, model_json['b'] + 1, model_json['loss']))

        embedding_loc = join_path(join_path(save_loc, EMB_DIR), self.model_tag)
        make_directory(embedding_loc)

        # To enable proper reshaping in the 2 layers.
        batch_loader = FixedBatchLoader(args_list, self.n_features, batch_size, 10000, self.model_tag, HOP**2, False, save_loc)
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            print('{}: Restoring Model...'.format(self.model_tag))
            saver.restore(sess, model_path)
            print('{}: Extracting embeddings in {} batches.'.format(self.model_tag, batch_loader.total_batches()))
            for b in range(batch_loader.total_batches()):
                batch_x, args_idx = batch_loader.next()
                print('{}: Extracting Batch {:d} embeddings...'.format(self.model_tag, b + 1))
                embeddings = sess.run(self.embeddings, feed_dict={
                    self.input_: batch_x,
                    self.batch_size: batch_loader.get_batch_size()
                })
                save_batch_array(embedding_loc, args_idx, embeddings, ext='.npy')
                print('{}: Saved Batch {:d} embeddings at: {}'.format(self.model_tag, b + 1, embedding_loc))

    def start_train(self, save_loc, batch_loader, epochs, lr, decay, cont=True):
        model_loc = join_path(join_path(save_loc, MODELS_DIR), self.model_tag)
        make_directory(model_loc)
        save_json = join_path(model_loc, LATEST_MODEL_FILE)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        with tf.Session(config=config) as sess:
            sess.run(init)
            if cont:
                with open(save_json, 'r') as f:
                    model_json = json.load(f)
                model_path = join_path(model_loc, '{}_Epoch{:d}_Batch{:d}_Loss{:.2f}.ckpt'
                                       .format(self.model_tag, model_json['e'] + 1, model_json['b'] + 1,
                                               model_json['loss']))
                saver.restore(sess, model_path)
                ne = model_json['e']
                nb = model_json['b']
            else:
                ne = 0
                nb = 0

            batch_size = batch_loader.get_batch_size()
            n_batches = batch_loader.total_batches()
            for e in range(ne, epochs):
                current_lr = lr * (decay ** e)
                for b in range(nb, n_batches):
                    start_time = get_time()
                    batch_x, batch_y = batch_loader.next()
                    _, loss = sess.run([self.optimizer, self.loss], feed_dict={
                        self.input_: batch_x,
                        self.labels: batch_y,
                        self.batch_size: batch_size,
                        self.lr: current_lr
                    })
                    end_time = get_time()
                    logger.info('{}: Epoch {:d} | Batch {:d}/{:d} | Loss: {:.3f} | Time Elapsed: {:d} seconds'
                                .format(self.model_tag, e + 1, b + 1, n_batches, loss, int(end_time - start_time)))
                    if (e * n_batches + b + 1) % 200 == 0:
                        model_path = join_path(model_loc, '{}_Epoch{:d}_Batch{:d}_Loss{:.2f}.ckpt'
                                               .format(self.model_tag, e + 1, b + 1, loss))
                        model_json = {
                            'e': e,
                            'b': b,
                            'lr': float(current_lr),
                            'loss': float(loss)
                        }
                        saver.save(sess, model_path)
                        with open(save_json, 'w') as f:
                            f.write(json.dumps(model_json))
                        logger.info('Model Saved at Epoch: {:d}, Batch: {:d} with Loss: {:.3f}'.format(e + 1, b + 1,
                                                                                                       loss))
                nb = 0

    def start_train_with_splits(self, args_list, splits, epochs, lr, decay, batch_size, save_loc='../save', cont=True):
        model_loc = join_path(join_path(save_loc, MODELS_DIR), self.model_tag)
        make_directory(model_loc)
        save_json = join_path(model_loc, LATEST_MODEL_FILE)

        dev_list, train_list = split_args_list(args_list, split=0.10, shuffle=True)

        logger.info('{}: Initialising batch loaders...'.format(self.model_tag))
        batch_loader = SplitBatchLoader(train_list, self.n_features, batch_size, splits, self.model_tag, HOP**2, True,
                                        save_loc)
        dev_loader = FixedBatchLoader(dev_list, self.n_features, batch_size, 1000, self.model_tag, HOP**2, False,
                                      save_loc)

        logger.info('{}: Starting training session...'.format(self.model_tag))
        min_dev_loss = np.array([100] * 5, dtype=float)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        with tf.Session(config=config) as sess:
            sess.run(init)
            if cont:
                with open(save_json, 'r') as f:
                    model_json = json.load(f)
                model_path = join_path(model_loc, '{}_Epoch{:d}_Batch{:d}_Loss{:.2f}.ckpt'
                                       .format(self.model_tag, model_json['e'] + 1, model_json['b'] + 1,
                                               model_json['loss']))
                saver.restore(sess, model_path)
                ne = model_json['e']
                nb = model_json['b'] + 1
                ns = model_json['s']
            else:
                ne = 0
                nb = 0
                ns = 0
            for s in batch_loader.get_splits()[ns:]:
                batch_loader.set_split(s)
                n_batches = batch_loader.total_batches()
                for e in range(ne, epochs):
                    current_lr = lr * (decay ** e)
                    for b in range(nb, n_batches):
                        start_time = get_time()
                        batch_x, batch_y = batch_loader.next()
                        _, loss = sess.run([self.optimizer, self.loss], feed_dict={
                            self.input_: batch_x,
                            self.labels: batch_y,
                            self.batch_size: batch_x.shape[0],
                            self.lr: current_lr
                        })
                        end_time = get_time()
                        logger.info(
                            '{}: Split: {:d} | Epoch {:d} | Batch {:d}/{:d} | Loss: {:.3f} | Time Elapsed: {:d} seconds'
                            .format(self.model_tag, s, e + 1, b + 1, n_batches, loss, int(end_time - start_time)))
                        if (e * n_batches + b + 1) % 100 == 0:
                            logger.info('{}: Calculating Dev Loss for {} dev batches...'.format(self.model_tag, dev_loader.total_batches()))
                            dev_loss = 0.0
                            start_time = get_time()
                            for _ in range(dev_loader.total_batches()):
                                batch_x, batch_y = dev_loader.next()
                                loss = sess.run(self.loss, feed_dict={
                                    self.input_: batch_x,
                                    self.labels: batch_y,
                                    self.batch_size: batch_x.shape[0]
                                })
                                dev_loss = dev_loss + float(loss)
                            dev_loss = dev_loss / dev_loader.total_batches()
                            end_time = get_time()
                            logger.info(
                                '{}: Split: {:d} | Epoch {:d} | Batch {:d} | Dev Loss: {:.3f} | Time Elapsed: {:d} seconds'
                                .format(self.model_tag, s, e + 1, b + 1, dev_loss, int(end_time - start_time)))

                            idx = np.argmax(min_dev_loss)
                            if dev_loss <= min_dev_loss[idx]:
                                model_path = join_path(model_loc, '{}_Epoch{:d}_Batch{:d}_Loss{:.2f}.ckpt'
                                                       .format(self.model_tag, e + 1, b + 1, dev_loss))
                                model_json = {
                                    'e': e,
                                    'b': b,
                                    's': s,
                                    'lr': float(current_lr),
                                    'loss': float(dev_loss)
                                }
                                saver.save(sess, model_path)
                                with open(save_json, 'w') as f:
                                    f.write(json.dumps(model_json))
                                logger.info('{}: Model Saved at Epoch: {:d}, Batch: {:d} with Loss: {:.3f}'
                                            .format(self.model_tag, e + 1, b + 1, dev_loss))
                                min_dev_loss[idx] = dev_loss
                    nb = 0
                ne = 0
