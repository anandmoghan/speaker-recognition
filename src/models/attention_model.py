from os.path import join as join_path
from time import time as get_time
from sklearn.metrics import accuracy_score

import numpy as np
import tensorflow as tf
import json

from constants.app_constants import MODELS_DIR, LATEST_MODEL_FILE
from models.layers.attention import supervised_attention
from services.common import make_directory
from services.loader import AttentionBatchLoader, LabelExtractLoader
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

MODEL_TAG = 'ATTN_MODEL'


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
        self.global_step = tf.train.get_or_create_global_step()
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)

    def compute_loss(self, batch_x, batch_y, sess):
        loss, global_step = sess.run([self.loss, self.global_step], feed_dict={
            self.input_: batch_x,
            self.labels: batch_y,
            self.batch_size: batch_x.shape[0]
        })
        return loss, global_step

    def get_current_step(self, sess):
        return sess.run(self.global_step)

    def train_step(self, batch_x, batch_y, lr, sess):
        loss, global_step, _ = sess.run([self.loss, self.global_step, self.train_op], feed_dict={
            self.input_: batch_x,
            self.labels: batch_y,
            self.batch_size: batch_x.shape[0],
            self.lr: lr
        })
        return loss, global_step
