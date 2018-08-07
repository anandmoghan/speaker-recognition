import json
from os.path import join as join_path

import tensorflow as tf

from constants.app_constants import EMB_DIR, LATEST_MODEL_FILE, MODELS_DIR
from lib.triplet_loss import batch_hard_triplet_loss
from services.common import make_directory, save_batch_array, tensorflow_debug, use_gpu
from services.logger import Logger

tensorflow_debug(False)
use_gpu(0)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

HOP = 10
LAYER_1_HIDDEN_UNITS = 512
LAYER_2_HIDDEN_UNITS = 512
LAYER_3_HIDDEN_UNITS = 512
EMBEDDING_SIZE = 512
TRIPLET_MARGIN = 0.2

MODEL_TAG = 'HGRUTRIPLET'

logger = Logger()
logger.set_config(filename='../logs/run-model.log', append=True)


class HGRUTripletModel:
    def __init__(self, n_features, n_classes):
        self.input_ = tf.placeholder(tf.float32, [None, n_features, None])
        self.labels = tf.placeholder(tf.int32, [None, ])
        self.n_classes = n_classes
        self.batch_size = tf.Variable(32, dtype=tf.int32, trainable=False)
        self.lr = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        # Sequence information in not important. So, to improve performance, every HOP frames are given parallel.
        input_ = tf.transpose(self.input_, [0, 2, 1])
        input_ = tf.reshape(input_, [-1, HOP, n_features])

        with tf.variable_scope('layer_1'):
            rnn_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(LAYER_1_HIDDEN_UNITS), input_,
                                              dtype=tf.float32)
            rnn_output = tf.reshape(rnn_output[:, -1, :], [-1, HOP, LAYER_1_HIDDEN_UNITS])

        with tf.variable_scope('layer_2'):
            rnn_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(LAYER_2_HIDDEN_UNITS), rnn_output,
                                              dtype=tf.float32)
            rnn_output = tf.reshape(rnn_output[:, -1, :], [self.batch_size, -1, LAYER_2_HIDDEN_UNITS])

        with tf.variable_scope('layer_3'):
            rnn_output, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(LAYER_3_HIDDEN_UNITS), rnn_output,
                                              dtype=tf.float32)
            rnn_output = tf.reduce_mean(rnn_output, axis=1)

        with tf.variable_scope('layer_4'):
            dense_output = tf.layers.dense(rnn_output, EMBEDDING_SIZE, activation=None)
            self.embeddings = tf.nn.l2_normalize(dense_output, dim=0)

        self.loss = batch_hard_triplet_loss(self.labels, self.embeddings, TRIPLET_MARGIN)
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
                embeddings = sess.run(self.embeddings, feed_dict={
                    self.input_: batch_x,
                    self.batch_size: batch_loader.get_batch_size()
                })
                save_batch_array(embedding_loc, args_idx, embeddings, ext='.npy')
                print('{}: Saved Batch {:d} embeddings at: {}'.format(MODEL_TAG, b + 1, embedding_loc))

        return batch_loader.get_last_idx()

    def start_train(self, save_loc, batch_loader, epochs, lr, decay, cont=True):
        model_loc = join_path(join_path(save_loc, MODELS_DIR), MODEL_TAG)
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
                                       .format(MODEL_TAG, model_json['e'] + 1, model_json['b'] + 1, model_json['loss']))
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
                    batch_x, batch_y = batch_loader.next()
                    _, loss = sess.run([self.optimizer, self.loss], feed_dict={
                        self.input_: batch_x,
                        self.labels: batch_y,
                        self.batch_size: batch_size,
                        self.lr: current_lr
                    })
                    logger.info('{}: Epoch {:d} | Batch {:d} | Loss: {:.3f}'
                                .format(MODEL_TAG, e + 1, b + 1, loss))
                    if (e * n_batches + b + 1) % 200 == 0:
                        model_path = join_path(model_loc, '{}_Epoch{:d}_Batch{:d}_Loss{:.2f}.ckpt'
                                               .format(MODEL_TAG, e + 1, b + 1, loss))
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

    def start_train_with_splits(self, save_loc, batch_loader, epochs, lr, decay, cont=True):
        model_loc = join_path(join_path(save_loc, MODELS_DIR), MODEL_TAG)
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
                                       .format(MODEL_TAG, model_json['e'] + 1, model_json['b'] + 1, model_json['loss']))
                saver.restore(sess, model_path)
                ne = model_json['e']
                nb = model_json['b']
                ns = model_json['s']
            else:
                ne = 0
                nb = 0
                ns = 0

            for s in [0, 1, 2][ns:]:
                batch_loader.set_split(s)
                batch_size = batch_loader.get_batch_size()
                n_batches = batch_loader.total_batches()
                for e in range(ne, epochs):
                    current_lr = lr * (decay ** e)
                    for b in range(nb, n_batches):
                        batch_x, batch_y = batch_loader.next()
                        _, loss = sess.run([self.optimizer, self.loss], feed_dict={
                            self.input_: batch_x,
                            self.labels: batch_y,
                            self.batch_size: batch_size,
                            self.lr: current_lr
                        })
                        logger.info('{}: Split {:d} | Epoch {:d} | Batch {:d} | Loss: {:.3f}'
                                    .format(MODEL_TAG, s, e + 1, b + 1, loss))
                        if (e * n_batches + b + 1) % 200 == 0:
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
                            logger.info('Model Saved at Epoch: {:d}, Batch: {:d} with Loss: {:.3f}'.format(e + 1, b + 1,
                                                                                                           loss))
                    nb = 0
