from os.path import join as join_path

import tensorflow as tf

from models.layers.stats_pooling import stats_pool
from services.common import make_directory
from services.logger import Logger

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

CNN_1_OUTPUT_SIZE = 512
CNN_1_CONTEXT = 5
CNN_2_OUTPUT_SIZE = 512
CNN_2_CONTEXT = 5
CNN_3_OUTPUT_SIZE = 512
CNN_3_CONTEXT = 5
CNN_4_OUTPUT_SIZE = 512
CNN_4_CONTEXT = 1
CNN_5_OUTPUT_SIZE = 1500
CNN_5_CONTEXT = 1

SAVE_TAG = 'xvector'

logger = Logger()
logger.set_config(filename='../logs/run-triplet-loss.log', append=True)


class XVectorModel:
    def __init__(self, batch_size, n_features, n_classes):
        self.input_ = tf.placeholder(tf.float32, [batch_size, n_features, None])
        self.labels = tf.placeholder(tf.int32, [batch_size, ])
        self.lr = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        input_ = tf.reshape(self.input_, [batch_size, n_features, -1, 1])

        cnn_output = tf.layers.conv2d(input_, filters=CNN_1_OUTPUT_SIZE, kernel_size=(n_features, CNN_1_CONTEXT))
        cnn_output = tf.transpose(cnn_output, [0, 3, 2, 1])

        cnn_output = tf.layers.conv2d(cnn_output, filters=CNN_2_OUTPUT_SIZE,
                                      kernel_size=(CNN_1_OUTPUT_SIZE, CNN_2_CONTEXT))
        cnn_output = tf.transpose(cnn_output, [0, 3, 2, 1])

        cnn_output = tf.layers.conv2d(cnn_output, filters=CNN_3_OUTPUT_SIZE,
                                      kernel_size=(CNN_2_OUTPUT_SIZE, CNN_3_CONTEXT))
        cnn_output = tf.transpose(cnn_output, [0, 3, 2, 1])

        cnn_output = tf.layers.conv2d(cnn_output, filters=CNN_4_OUTPUT_SIZE,
                                      kernel_size=(CNN_3_OUTPUT_SIZE, CNN_4_CONTEXT))
        cnn_output = tf.transpose(cnn_output, [0, 3, 2, 1])

        cnn_output = tf.layers.conv2d(cnn_output, filters=CNN_5_OUTPUT_SIZE,
                                      kernel_size=(CNN_4_OUTPUT_SIZE, CNN_5_CONTEXT))
        cnn_output = tf.transpose(tf.squeeze(cnn_output), [0, 2, 1])

        stats_output = stats_pool(cnn_output, axes=2)
        stats_output = tf.reshape(stats_output, [batch_size, 2 * CNN_5_OUTPUT_SIZE])

        self.x_vector = tf.layers.dense(stats_output, 512, activation=None)
        fc_output = tf.nn.relu(self.x_vector)
        fc_output = tf.layers.dense(fc_output, 512, activation=tf.nn.relu)
        self.logits = tf.layers.dense(fc_output, n_classes, activation=None)

        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def start_train(self, save_loc, batch_loader, epochs, lr, decay):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        save_loc = join_path(save_loc, 'models')
        make_directory(save_loc)
        with tf.Session(config=config) as sess:
            sess.run(init)
            for s in [0, 1, 2]:
                batch_loader.set_split(s)
                for e in range(epochs):
                    elr = lr * (decay ** e)
                    for b in range(batch_loader.total_batches()):
                        batch_x, batch_y = batch_loader.next()
                        _, loss = sess.run([self.optimizer, self.loss], feed_dict={
                            self.input_: batch_x,
                            self.labels: batch_y,
                            self.lr: elr
                        })
                        logger.info('Epoch {:d} | Batch {:d} | Loss: {:.2f}'.format(e + 1, b + 1, loss))
                        if (e + 1) * (b + 1) % 100 == 0:
                            model_path = join_path(save_loc, '{}_Epoch{:d}_Batch{:d}_Loss{:.2f}.ckpt'.format(SAVE_TAG, e + 1, b + 1, loss))
                            saver.save(sess, model_path)
