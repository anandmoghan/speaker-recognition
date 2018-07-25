from os.path import join as join_path

import tensorflow as tf
import json

from constants.app_constants import EMB_DIR, LATEST_MODEL_FILE, MODELS_DIR
from layers.pooling import stats_pool
from services.common import make_directory, save_batch_array, tensorflow_debug, use_gpu
from services.logger import Logger

tensorflow_debug(False)
use_gpu(0)

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
EMBEDDING_SIZE = 512
DENSE_SIZE = 512

MODEL_TAG = 'xvector'

logger = Logger()
logger.set_config(filename='../logs/run-triplet-loss.log', append=True)


class XVectorModel:
    def __init__(self, batch_size, n_features, n_classes):
        self.input_ = tf.placeholder(tf.float32, [batch_size, n_features, None])
        self.labels = tf.placeholder(tf.int32, [batch_size, ])
        self.lr = tf.Variable(0.0, dtype=tf.float64, trainable=False)

        input_ = tf.reshape(self.input_, [batch_size, n_features, -1, 1])

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
        cnn_output = tf.transpose(tf.squeeze(cnn_output), [0, 2, 1])

        stats_output = stats_pool(cnn_output, axes=2)
        stats_output = tf.reshape(stats_output, [batch_size, 2 * CNN_5_OUTPUT_SIZE])

        self.x_vector = tf.layers.dense(stats_output, EMBEDDING_SIZE, activation=None)
        dense_output = tf.layers.dense(tf.nn.relu(self.x_vector), DENSE_SIZE, activation=tf.nn.relu)
        self.logits = tf.layers.dense(dense_output, n_classes, activation=None)

        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.logits)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def extract(self, save_loc, batch_loader):
        model_loc = join_path(join_path(save_loc, MODELS_DIR), MODEL_TAG)
        save_json = join_path(model_loc, LATEST_MODEL_FILE)
        with open(save_json, 'r') as f:
            model_json = json.load(f)
        model_path = join_path(model_loc, '{}_Epoch{:d}_Batch{:d}_Loss{:.2f}.ckpt'
                               .format(MODEL_TAG, model_json['e'] + 1, model_json['b'] + 1, model_json['loss']))

        embedding_loc = join_path(join_path(save_loc, EMB_DIR), MODEL_TAG)
        make_directory(embedding_loc)

        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            print('{}: Restoring Model...'.format(MODEL_TAG))
            saver.restore(sess, model_path)
            for b in range(batch_loader.total_batches()):
                batch_x, args_idx = batch_loader.next()
                print('{}: Extracting Batch {:d} embeddings...'.format(MODEL_TAG, b + 1))
                embeddings = sess.run(self.x_vector, feed_dict={
                    self.input_: batch_x
                })
                save_batch_array(embedding_loc, args_idx, embeddings, ext='.npy')
                print('{}: Saved Batch {:d} embeddings at: {}'.format(MODEL_TAG, b + 1, embedding_loc))

        return batch_loader.get_last_idx()

    def start_train(self, save_loc, batch_loader, epochs, lr, decay):
        model_loc = join_path(join_path(save_loc, MODELS_DIR), MODEL_TAG)
        make_directory(model_loc)
        save_json = join_path(model_loc, LATEST_MODEL_FILE)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        with tf.Session(config=config) as sess:
            sess.run(init)
            for s in [0, 1, 2]:
                batch_loader.set_split(s)
                for e in range(epochs):
                    current_lr = lr * (decay ** e)
                    for b in range(batch_loader.total_batches()):
                        batch_x, batch_y = batch_loader.next()
                        _, loss = sess.run([self.optimizer, self.loss], feed_dict={
                            self.input_: batch_x,
                            self.labels: batch_y,
                            self.lr: current_lr
                        })
                        logger.info('{}: Epoch {:d} | Batch {:d} | Loss: {:.2f}'.format(MODEL_TAG, e + 1, b + 1, loss))
                        if (e + 1) * (b + 1) % 200 == 0:
                            model_path = join_path(model_loc, '{}_Epoch{:d}_Batch{:d}_Loss{:.2f}.ckpt'
                                                   .format(MODEL_TAG, e + 1, b + 1, loss))
                            model_json = {
                                'e': e,
                                'b': b,
                                's': s,
                                'lr': float(current_lr),
                                'loss': float(loss)
                            }
                            saver.save(sess, model_path)
                            with open(save_json, 'w') as f:
                                f.write(json.dumps(model_json))
                            logger.info('Model Saved at Epoch: {:d}, Batch: {:d} with Loss: {:.2f}'.format(e + 1, b + 1,
                                                                                                           loss))
