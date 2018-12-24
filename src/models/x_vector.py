import tensorflow as tf

from models.base import BaseEmbedding
from models.layers.attention import variable_attention
from models.layers.stats_pooling import stats_pool
from models.layers.nn import tdnn

TDNN_1_OUTPUT_SIZE = 512
TDNN_2_OUTPUT_SIZE = 512
TDNN_3_OUTPUT_SIZE = 512
TDNN_4_OUTPUT_SIZE = 512
TDNN_5_OUTPUT_SIZE = 1500
ATTENTION_SIZE = 256
EMBEDDING_SIZE = 512
DENSE_SIZE = 512

activation_fn = tf.tanh


class XVector(BaseEmbedding):
    def __init__(self, n_features, n_classes, attention=False):
        super().__init__(n_features, n_classes)

        input_ = tf.expand_dims(self.input_, axis=3)

        tdnn_output = tdnn('tdnn1', input_, [-2, -1, 0, 1, 2], TDNN_1_OUTPUT_SIZE, activation=activation_fn)
        tdnn_output = tf.layers.batch_normalization(tdnn_output)

        tdnn_output = tdnn('tdnn2', tdnn_output, [-2, 0, 2], TDNN_2_OUTPUT_SIZE, activation=activation_fn)
        tdnn_output = tf.layers.batch_normalization(tdnn_output)

        tdnn_output = tdnn('tdnn3', tdnn_output, [-3, 0, 3], TDNN_3_OUTPUT_SIZE, activation=activation_fn)
        tdnn_output = tf.layers.batch_normalization(tdnn_output)

        tdnn_output = tdnn('tdnn4', tdnn_output, [0], TDNN_4_OUTPUT_SIZE, activation=activation_fn)
        tdnn_output = tf.layers.batch_normalization(tdnn_output)

        tdnn_output = tdnn('tdnn5', tdnn_output, [0], TDNN_5_OUTPUT_SIZE, activation=activation_fn)
        tdnn_output = tf.layers.batch_normalization(tdnn_output)
        tdnn_output = tf.squeeze(tdnn_output, axis=3)

        if attention:
            tdnn_output = tf.transpose(tdnn_output, [0, 2, 1])
            stats_output = variable_attention(tdnn_output, size=ATTENTION_SIZE)
            stats_output = tf.reshape(stats_output, [-1, TDNN_5_OUTPUT_SIZE])
        else:
            stats_output = stats_pool(tdnn_output, axes=2)
            stats_output = tf.reshape(stats_output, [-1, 2 * TDNN_5_OUTPUT_SIZE])

        self.embeddings = tf.layers.dense(stats_output, EMBEDDING_SIZE, activation=None)
        dense_output = activation_fn(self.embeddings)
        dense_output = tf.layers.batch_normalization(dense_output)

        dense_output = tf.layers.dense(dense_output, DENSE_SIZE, activation=activation_fn)
        dense_output = tf.layers.batch_normalization(dense_output)

        self.logits = tf.layers.dense(dense_output, n_classes, activation=None)
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.targets, self.logits)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        # gradients = self.optimizer.compute_gradients(self.loss)
        # clipped_gradients = [(tf.clip_by_value(grad, -0.75, 0.75) if var.name[:4] == 'tdnn' else tf.clip_by_value(grad, -1.5, 1.5), var) for grad, var in gradients]
        # self.train_op = self.optimizer.apply_gradients(clipped_gradients, global_step=self.global_step)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
