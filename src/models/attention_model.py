from os.path import join as join_path
from time import time as get_time
from sklearn.metrics import accuracy_score

import numpy as np
import tensorflow as tf
import json

from constants.app_constants import MODELS_DIR, LATEST_MODEL_FILE
from models.layers.attention import supervised_attention
from services.common import make_directory
from services.loader import AttentionBatchLoader
from services.logger import Logger

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

HOP = 10
LAYER_1_HIDDEN_UNITS = 128
LAYER_2_HIDDEN_UNITS = 256
LAYER_3_HIDDEN_UNITS = 512
LAYER_4_DENSE_SIZE = 512
LAYER_5_DENSE_SIZE = 256
LAYER_6_DENSE_SIZE = 128
ATTENTION_SIZE = 128
EMBEDDING_SIZE = 512

logger = Logger()
logger.set_config(filename='../logs/run-model.log', append=True)

MODEL_TAG = 'HGRUModel'


class AttentionModel:
    def __init__(self, n_features, n_classes, model_tag=MODEL_TAG):
        self.n_classes = n_classes
        self.n_features = n_features
        self.model_tag = model_tag
        self.input_ = tf.placeholder(tf.float32, [None, n_features, None])
        self.context_vector = tf.placeholder(tf.float32, [1, EMBEDDING_SIZE])
        self.labels = tf.placeholder(tf.int32, [None, ])
        self.batch_size = tf.Variable(32, dtype=tf.int32, trainable=False)
        self.lr = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        # Sequence information in not important. So, to improve performance, every HOP frames are given parallel.
        input_ = tf.transpose(self.input_, [0, 2, 1])
        input_ = tf.reshape(input_, [-1, HOP, n_features])

        with tf.variable_scope('layer_1'):
            rnn_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(LAYER_1_HIDDEN_UNITS, activation=tf.nn.relu),
                                              input_, dtype=tf.float32)
            rnn_output = supervised_attention(rnn_output, self.context_vector, size=ATTENTION_SIZE)
            rnn_output = tf.reshape(rnn_output, [-1, HOP, LAYER_1_HIDDEN_UNITS])

        with tf.variable_scope('layer_2'):
            rnn_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(LAYER_2_HIDDEN_UNITS, activation=tf.nn.relu),
                                              rnn_output, dtype=tf.float32)
            rnn_output = supervised_attention(rnn_output, self.context_vector, size=ATTENTION_SIZE)
            rnn_output = tf.reshape(rnn_output, [self.batch_size, -1, LAYER_2_HIDDEN_UNITS])

        with tf.variable_scope('layer_3'):
            rnn_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(LAYER_3_HIDDEN_UNITS, activation=tf.nn.relu),
                                              rnn_output, dtype=tf.float32)
            rnn_output = supervised_attention(rnn_output, self.context_vector, size=ATTENTION_SIZE)

        with tf.variable_scope('layer_4'):
            dense_output = tf.layers.dense(rnn_output, LAYER_4_DENSE_SIZE, activation=tf.nn.relu)
            dense_output = tf.layers.batch_normalization(dense_output)

        with tf.variable_scope('layer_5'):
            dense_output = tf.layers.dense(dense_output, LAYER_5_DENSE_SIZE, activation=tf.nn.relu)
            dense_output = tf.layers.batch_normalization(dense_output)

        with tf.variable_scope('layer_6'):
            dense_output = tf.layers.dense(dense_output, LAYER_6_DENSE_SIZE, activation=tf.nn.relu)

        self.logits = tf.layers.dense(dense_output, n_classes, activation=None)
        self.predicted_labels = tf.argmax(self.logits, axis=1)
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def start_train(self, args_list, idx_to_label, epochs, lr, decay, batch_size, save_loc, cont=False):
        model_loc = join_path(join_path(save_loc, MODELS_DIR), self.model_tag)
        make_directory(model_loc)
        save_json = join_path(model_loc, LATEST_MODEL_FILE)

        batch_loader = AttentionBatchLoader(args_list, idx_to_label, self.n_features, batch_size, 300, 500,
                                            self.model_tag, 10, HOP ** 2, True, cont, save_loc)
        dev_loader = batch_loader.get_dev_loader()

        logger.info('{}: Starting training session...'.format(self.model_tag))
        min_dev_loss = np.array([100] * 5, dtype=float)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        with tf.Session(config=config) as sess:
            sess.run(init)
            ne = 0
            nb = 0
            n_batches = batch_loader.total_batches()

            if cont:
                with open(save_json, 'r') as f:
                    model_json = json.load(f)

                model_name = '{}_Epoch{:d}_Batch{:d}_Loss{:.2f}.ckpt'.format(self.model_tag, model_json['e'] + 1,
                                                                             model_json['b'] + 1, model_json['loss'])
                model_path = join_path(model_loc, model_name)
                saver.restore(sess, model_path)
                ne = model_json['e']
                nb = model_json['b'] + 1

            for e in range(ne, epochs):
                current_lr = lr * (decay ** e)
                for b in range(nb, n_batches):
                    start_time = get_time()
                    batch_x, batch_y, context_vector = batch_loader.next()
                    _, loss, predicted_labels = sess.run([self.optimizer, self.loss, self.predicted_labels], feed_dict={
                        self.input_: batch_x,
                        self.labels: batch_y,
                        self.context_vector: context_vector,
                        self.batch_size: batch_x.shape[0],
                        self.lr: current_lr
                    })
                    accuracy = 100 * accuracy_score(predicted_labels, batch_y)
                    end_time = get_time()
                    logger.info('{}: Epoch {:d} | Batch {:d}/{:d} | Loss: {:.3f} | Accuracy: {:0.2f}% | Time Elapsed: {:d} seconds'
                        .format(self.model_tag, e + 1, b + 1, n_batches, loss, accuracy, int(end_time - start_time)))
                    if (e * n_batches + b + 1) % 100 == 0:
                        logger.info('{}: Calculating Dev Loss for {} dev batches...'
                                    .format(self.model_tag, dev_loader.total_batches()))
                        dev_loss = 0.0
                        dev_accuracy = 0
                        start_time = get_time()
                        for _ in range(dev_loader.total_batches()):
                            batch_x, batch_y, context_vector = dev_loader.next()
                            loss, predicted_labels = sess.run([self.loss, self.predicted_labels], feed_dict={
                                self.input_: batch_x,
                                self.labels: batch_y,
                                self.context_vector: context_vector,
                                self.batch_size: batch_x.shape[0]
                            })
                            dev_accuracy = dev_accuracy + accuracy_score(batch_y, predicted_labels)
                            dev_loss = dev_loss + float(loss)
                        dev_loss = dev_loss / dev_loader.total_batches()
                        dev_accuracy = 100 * dev_accuracy / dev_loader.total_batches()
                        end_time = get_time()
                        logger.info('{}: Development Set | Epoch {:d} | Batch {:d} | Loss: {:.3f} | Accuracy: {:.2f}% | Time Elapsed: {:d} seconds'
                            .format(self.model_tag, e + 1, b + 1, dev_loss, dev_accuracy, int(end_time - start_time)))

                        idx = np.argmax(min_dev_loss)
                        if dev_loss <= min_dev_loss[idx]:
                            model_path = join_path(model_loc, '{}_Epoch{:d}_Batch{:d}_Loss{:.2f}.ckpt'
                                                   .format(self.model_tag, e + 1, b + 1, dev_loss))
                            model_json = {
                                'e': e,
                                'b': b,
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
        return self.n_classes
