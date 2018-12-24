import tensorflow as tf

from models.base import BaseEmbedding
from models.layers.attention import variable_attention, multi_head_attention
from lib.triplet_loss import batch_hard_triplet_loss
from services.common import arrange_data

WINDOW1 = 10
HOP1 = 5
HOP2 = 10
LAYER_1_HIDDEN_UNITS = 256
LAYER_2_HIDDEN_UNITS = 512
LAYER_3_HIDDEN_UNITS = 256
LAYER_4_DENSE_SIZE = 1024
ATTENTION_SIZE = 128
EMBEDDING_SIZE = 256
TRIPLET_MARGIN = 0.2


def arrange_input(batch_x):
    return arrange_data(batch_x, window=WINDOW1, hop=HOP1)


class TripletModel(BaseEmbedding):
    def __init__(self, n_features, n_classes, attention=True):
        super().__init__(n_features, n_classes)

        net_hop = WINDOW1 * HOP2
        max_frames = tf.cast(tf.floor(tf.shape(self.input_)[2] / net_hop), tf.int32) * net_hop
        input_ = self.input_[:, :, :max_frames]

        # Sequence information in not important. So, to improve performance, every HOP frames are given parallel.
        input_ = tf.transpose(input_, [0, 2, 1])
        input_ = tf.reshape(input_, [-1, WINDOW1, n_features])

        with tf.variable_scope('layer_1'):
            rnn_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(LAYER_1_HIDDEN_UNITS, activation=tf.nn.tanh),
                                              input_, dtype=tf.float32)
            rnn_output = tf.reshape(rnn_output[:, -1, :], [-1, HOP2, LAYER_1_HIDDEN_UNITS])

        with tf.variable_scope('layer_2'):
            rnn_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(LAYER_2_HIDDEN_UNITS, activation=tf.nn.tanh),
                                              rnn_output, dtype=tf.float32)
            rnn_output = tf.reshape(rnn_output[:, -1, :], [self.batch_size, -1, LAYER_2_HIDDEN_UNITS])

        with tf.variable_scope('layer_3'):
            cell_fw = tf.contrib.rnn.GRUCell(LAYER_3_HIDDEN_UNITS, activation=tf.nn.tanh)
            cell_bw = tf.contrib.rnn.GRUCell(LAYER_3_HIDDEN_UNITS, activation=tf.nn.tanh)
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_output, dtype=tf.float32)
            rnn_output = tf.concat([rnn_outputs[0], rnn_outputs[1]], axis=2)
            if attention:
                pooled_output = multi_head_attention(rnn_output, sizes=[ATTENTION_SIZE] * 2)
            else:
                pooled_output = tf.reduce_mean(rnn_output, axis=1)

        with tf.variable_scope('layer_4'):
            dense_output = tf.layers.dense(pooled_output, LAYER_4_DENSE_SIZE, activation=tf.nn.relu)

        with tf.variable_scope('layer_5'):
            dense_output = tf.layers.dense(dense_output, EMBEDDING_SIZE, activation=None)
            self.embeddings = tf.nn.l2_normalize(dense_output, dim=0, name='embeddings')

        self.loss = batch_hard_triplet_loss(self.targets, self.embeddings, TRIPLET_MARGIN)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def compute_loss(self, batch_x, batch_y, sess):
        loss, global_step = super().compute_loss(arrange_input(batch_x), batch_y, sess)
        return loss * 100, global_step

    def extract(self, batch_x, sess, save_loc=None):
        return super().extract(arrange_input(batch_x), sess, save_loc)

    def train_step(self, batch_x, batch_y, lr, sess):
        loss, global_step = super().train_step(arrange_input(batch_x), batch_y, lr, sess)
        return loss * 100, global_step
