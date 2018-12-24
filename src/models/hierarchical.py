import tensorflow as tf

from models.base import BaseEmbedding
from models.layers.attention import variable_attention, multi_head_attention
from services.common import arrange_data

WINDOW1 = 20
HOP1 = 10
HOP2 = 5
LAYER_1_HIDDEN_UNITS = 256
LAYER_2_HIDDEN_UNITS = 512
LAYER_3_HIDDEN_UNITS = 256
ATTENTION_SIZE = 128
EMBEDDING_SIZE = 512
DENSE_SIZE = 1024


def arrange_input(batch_x):
    return arrange_data(batch_x, window=WINDOW1, hop=HOP1)


class HGRU(BaseEmbedding):
    def __init__(self, n_features, n_classes, attention=True):
        super().__init__(n_features, n_classes)

        net_hop = WINDOW1 * HOP2
        max_frames = tf.cast(tf.floor(tf.shape(self.input_)[2] / net_hop), tf.int32) * net_hop
        input_ = self.input_[:, :, :max_frames]

        # Sequence information in not important. So, to improve performance, every HOP frames are given parallel.
        input_ = tf.transpose(input_, [0, 2, 1])
        input_ = tf.reshape(input_, [-1, WINDOW1, n_features])

        with tf.variable_scope('layer_1'):
            rnn_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(LAYER_1_HIDDEN_UNITS, activation=tf.tanh),
                                              input_, dtype=tf.float32)
            rnn_output = tf.reshape(rnn_output[:, -1, :], [-1, HOP2, LAYER_1_HIDDEN_UNITS])

        with tf.variable_scope('layer_2'):
            rnn_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(LAYER_2_HIDDEN_UNITS, activation=tf.tanh),
                                              rnn_output, dtype=tf.float32)
            rnn_output = tf.reshape(rnn_output[:, -1, :], [self.batch_size, -1, LAYER_2_HIDDEN_UNITS])

        with tf.variable_scope('layer_3'):
            cell_fw = tf.contrib.rnn.GRUCell(LAYER_3_HIDDEN_UNITS, activation=tf.tanh)
            cell_bw = tf.contrib.rnn.GRUCell(LAYER_3_HIDDEN_UNITS, activation=tf.tanh)
            rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, rnn_output, dtype=tf.float32)
            rnn_output = tf.concat([rnn_outputs[0], rnn_outputs[1]], axis=2)
            if attention:
                # pooled_output = variable_attention(rnn_output, size=ATTENTION_SIZE)
                pooled_output = multi_head_attention(rnn_output, sizes=[ATTENTION_SIZE] * 2)
            else:
                pooled_output = tf.reduce_mean(rnn_output, axis=1)

        with tf.variable_scope('layer_4'):
            self.embeddings = tf.layers.dense(pooled_output, EMBEDDING_SIZE, activation=None, name='embeddings')
            dense_output = tf.tanh(self.embeddings)

        with tf.variable_scope('layer_5'):
            dense_output = tf.layers.dense(dense_output, DENSE_SIZE, activation=tf.tanh)

        self.logits = tf.layers.dense(dense_output, n_classes, activation=None)
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.targets, self.logits)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # gradients = self.optimizer.compute_gradients(self.loss)
        # clipped_gradients = [(tf.clip_by_value(grad, -0.75, 0.75), var) for grad, var in gradients]
        # self.train_op = self.optimizer.apply_gradients(clipped_gradients, global_step=self.global_step)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def compute_loss(self, batch_x, batch_y, sess):
        return super().compute_loss(arrange_input(batch_x), batch_y, sess)

    def extract(self, batch_x, sess, save_loc=None):
        return super().extract(arrange_input(batch_x), sess, save_loc)

    def train_step(self, batch_x, batch_y, lr, sess):
        return super().train_step(arrange_input(batch_x), batch_y, lr, sess)
