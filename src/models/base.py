import tensorflow as tf

from services.common import save_array


PREFETCH_COUNT = 2560


class Base:
    def __init__(self, n_features, n_classes):
        self.batch_size = tf.Variable(32, dtype=tf.int32, trainable=False, name='batch_size')

        self.input_ = tf.placeholder(tf.float32, [None, n_features, None], name='input')
        self.targets = tf.placeholder(tf.int32, [None, ], name='targets')

        self.lr = tf.Variable(0.001, dtype=tf.float32, trainable=False, name='lr')
        self.global_step = tf.train.get_or_create_global_step()

        self.n_features = n_features
        self.n_classes = n_classes

        self.logits = None
        self.loss = None
        self.optimizer = None
        self.train_op = None

    def compute_loss(self, batch_x, batch_y, sess):
        loss, global_step = sess.run([self.loss, self.global_step], feed_dict={
            self.input_: batch_x,
            self.targets: batch_y,
            self.batch_size: batch_x.shape[0]
        })
        return loss, global_step

    def get_global_step(self, sess):
        return sess.run(self.global_step)

    def train_step(self, batch_x, batch_y, lr, sess):
        loss, global_step, _ = sess.run([self.loss, self.global_step, self.train_op], feed_dict={
            self.input_: batch_x,
            self.targets: batch_y,
            self.batch_size: batch_x.shape[0],
            self.lr: lr
        })
        return loss, global_step


class BaseEmbedding(Base):
    def __init__(self, n_features, n_classes):
        super().__init__(n_features, n_classes)
        self.embeddings = None

    def extract(self, batch_x, sess, save_loc=None):
        batch_size = batch_x.shape[0]
        embeddings = sess.run(self.embeddings, feed_dict={
            self.input_: batch_x,
            self.batch_size: batch_size
        })

        if save_loc is not None:
            if batch_size == 1 and save_loc is not list:
                save_loc = [save_loc]
            for i in range(batch_size):
                save_array(save_loc[i], embeddings[i, :])

        return embeddings
