from os.path import join as join_path
from time import time as get_time

import tensorflow as tf
import json

from constants.app_constants import EMB_DIR, LATEST_MODEL_FILE, MODELS_DIR
from models.layers.attention import variable_attention
from models.layers.stats_pooling import stats_pool
from models.layers.tdnn import tdnn2d
from services.common import make_directory, save_batch_array
from services.logger import Logger

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

TDNN_1_OUTPUT_SIZE = 512
TDNN_2_OUTPUT_SIZE = 512
TDNN_3_OUTPUT_SIZE = 512
TDNN_4_OUTPUT_SIZE = 512
TDNN_5_OUTPUT_SIZE = 1500
ATTENTION_SIZE = 256
EMBEDDING_SIZE = 512
DENSE_SIZE = 512

MODEL_TAG = 'XVECTOR'

logger = Logger()
logger.set_config(filename='../logs/{}.log'.format(MODEL_TAG), append=True)


class XVectorModel:
    def __init__(self, n_features, n_classes, attention=False, model_tag=MODEL_TAG):
        self.input_ = tf.placeholder(tf.float32, [None, n_features, None])
        self.labels = tf.placeholder(tf.int32, [None, ])
        self.batch_size = tf.Variable(32, dtype=tf.int32, trainable=False)
        self.lr = tf.Variable(0.0, dtype=tf.float64, trainable=False)
        self.model_tag = model_tag

        input_ = tf.expand_dims(self.input_, axis=3)
        input_ = tf.layers.batch_normalization(input_)

        tdnn_output = tdnn2d('tdnn1', input_, [-2, -1, 0, 1, 2], TDNN_1_OUTPUT_SIZE, activation=tf.nn.relu)
        tdnn_output = tf.layers.batch_normalization(tdnn_output)

        tdnn_output = tdnn2d('tdnn2', tdnn_output, [-2, -1, 0, 1, 2], TDNN_2_OUTPUT_SIZE, activation=tf.nn.relu)
        tdnn_output = tf.layers.batch_normalization(tdnn_output)

        tdnn_output = tdnn2d('tdnn3', tdnn_output, [-2, 0, 2], TDNN_3_OUTPUT_SIZE, activation=tf.nn.relu)
        tdnn_output = tf.layers.batch_normalization(tdnn_output)

        tdnn_output = tdnn2d('tdnn4', tdnn_output, [0], TDNN_4_OUTPUT_SIZE, activation=tf.nn.relu)
        tdnn_output = tf.layers.batch_normalization(tdnn_output)

        tdnn_output = tdnn2d('tdnn5', tdnn_output, [0], TDNN_5_OUTPUT_SIZE, activation=tf.nn.relu)
        tdnn_output = tf.layers.batch_normalization(tdnn_output)

        tdnn_output = tf.squeeze(tdnn_output)

        if attention:
            stats_output = variable_attention(tdnn_output, size=ATTENTION_SIZE)
            stats_output = tf.reshape(stats_output, [-1, TDNN_5_OUTPUT_SIZE])
        else:
            stats_output = stats_pool(tdnn_output, axes=2)
            stats_output = tf.reshape(stats_output, [-1, 2 * TDNN_5_OUTPUT_SIZE])

        self.x_vector = tf.layers.dense(stats_output, EMBEDDING_SIZE, activation=None)
        dense_output = tf.layers.batch_normalization(tf.nn.relu(self.x_vector))

        dense_output = tf.layers.dense(dense_output, DENSE_SIZE, activation=tf.nn.relu)
        dense_output = tf.layers.batch_normalization(dense_output)

        self.logits = tf.layers.dense(dense_output, n_classes, activation=None)

        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.logits)
        self.global_step = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss, global_step=self.global_step)

    def extract(self, save_loc, batch_loader, start=None, end=None):
        model_loc = join_path(join_path(save_loc, MODELS_DIR), self.model_tag)
        save_json = join_path(model_loc, LATEST_MODEL_FILE)
        with open(save_json, 'r') as f:
            model_json = json.load(f)
        model_path = join_path(model_loc, '{}_Epoch{:d}_Batch{:d}_Loss{:.2f}.ckpt'
                               .format(self.model_tag, model_json['e'] + 1, model_json['b'] + 1, model_json['loss']))

        embedding_loc = join_path(save_loc, join_path(EMB_DIR, self.model_tag))
        make_directory(embedding_loc)

        if start is None:
            start = batch_loader.get_current_batch()
        else:
            batch_loader.set_current_batch(start)

        if end is None:
            end = batch_loader.total_batches()

        if start > end:
            raise Exception('{}: Extract Embedding - start greater than end.'.format(self.model_tag))

        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            print('{}: Restoring Model...'.format(self.model_tag))
            saver.restore(sess, model_path)
            print('{}: Extracting for batches {} to {}'.format(self.model_tag, start, end))
            for b in range(start, end):
                batch_x, args_idx = batch_loader.next()
                print('{}: Extracting Batch {:d} embeddings...'.format(self.model_tag, b + 1))
                embeddings = sess.run(self.x_vector, feed_dict={
                    self.input_: batch_x,
                    self.batch_size: batch_loader.get_batch_size()
                })
                save_batch_array(embedding_loc, args_idx, embeddings, ext='.npy')
                print('{}: Saved Batch {:d} embeddings at: {}'.format(self.model_tag, b + 1, embedding_loc))

    def start_train_with_splits(self, save_loc, batch_loader, epochs, lr, decay, cont=True):
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
                                       .format(self.model_tag, model_json['e'] + 1, model_json['b'] + 1, model_json['loss']))
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
                        logger.info('{}: Split: {:d} | Epoch {:d} | Batch {:d}/{:d} | Loss: {:.3f} | Time Elapsed: {:d} seconds'
                                    .format(self.model_tag, s, e + 1, b + 1, n_batches, loss, int(end_time - start_time)))
                        if (e * n_batches + b + 1) % 200 == 0:
                            model_path = join_path(model_loc, '{}_Epoch{:d}_Batch{:d}_Loss{:.2f}.ckpt'
                                                   .format(self.model_tag, e + 1, b + 1, loss))
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
                ne = 0
