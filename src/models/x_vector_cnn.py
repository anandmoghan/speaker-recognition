import tensorflow as tf

from models.base import BaseEmbedding
from models.layers.attention import variable_attention
from models.layers.stats_pooling import stats_pool

CNN_1_OUTPUT_SIZE = 512
CNN_1_CONTEXT = 5
CNN_2_OUTPUT_SIZE = 512
CNN_2_CONTEXT = 5
CNN_3_OUTPUT_SIZE = 512
CNN_3_CONTEXT = 7
CNN_4_OUTPUT_SIZE = 512
CNN_4_CONTEXT = 1
CNN_5_OUTPUT_SIZE = 1500
CNN_5_CONTEXT = 1
ATTENTION_SIZE = 512
EMBEDDING_SIZE = 512
DENSE_SIZE = 512


class XVectorCNN(BaseEmbedding):
    def __init__(self, n_features, n_classes, attention=False):
        super().__init__(n_features, n_classes)

        input_ = tf.expand_dims(self.input_, axis=3)

        cnn_output = tf.layers.conv2d(input_, filters=CNN_1_OUTPUT_SIZE, kernel_size=(n_features, CNN_1_CONTEXT),
                                      activation=tf.nn.relu)
        cnn_output = tf.transpose(cnn_output, [0, 3, 2, 1])

        cnn_output = tf.layers.conv2d(cnn_output, filters=CNN_2_OUTPUT_SIZE,
                                      kernel_size=(CNN_1_OUTPUT_SIZE, CNN_2_CONTEXT), activation=tf.nn.relu)
        cnn_output = tf.transpose(cnn_output, [0, 3, 2, 1])

        cnn_output = tf.layers.conv2d(cnn_output, filters=CNN_3_OUTPUT_SIZE,
                                      kernel_size=(CNN_2_OUTPUT_SIZE, CNN_3_CONTEXT), activation=tf.nn.relu)
        cnn_output = tf.transpose(cnn_output, [0, 3, 2, 1])

        cnn_output = tf.layers.conv2d(cnn_output, filters=CNN_4_OUTPUT_SIZE,
                                      kernel_size=(CNN_3_OUTPUT_SIZE, CNN_4_CONTEXT), activation=tf.nn.relu)
        cnn_output = tf.transpose(cnn_output, [0, 3, 2, 1])

        cnn_output = tf.layers.conv2d(cnn_output, filters=CNN_5_OUTPUT_SIZE,
                                      kernel_size=(CNN_4_OUTPUT_SIZE, CNN_5_CONTEXT), activation=tf.nn.relu)
        cnn_output = tf.transpose(tf.squeeze(cnn_output, axis=1), [0, 2, 1])

        if attention:
            cnn_output = tf.transpose(cnn_output, [0, 2, 1])
            stats_output = variable_attention(cnn_output, size=ATTENTION_SIZE)
            stats_output = tf.reshape(stats_output, [-1, CNN_5_OUTPUT_SIZE])
        else:
            stats_output = stats_pool(cnn_output, axes=2)
            stats_output = tf.reshape(stats_output, [-1, 2 * CNN_5_OUTPUT_SIZE])

        self.embeddings = tf.layers.dense(stats_output, EMBEDDING_SIZE, activation=None)
        dense_output = tf.nn.relu(self.embeddings)

        dense_output = tf.layers.dense(dense_output, DENSE_SIZE, activation=tf.nn.relu)

        self.logits = tf.layers.dense(dense_output, n_classes, activation=None)
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.targets, self.logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
