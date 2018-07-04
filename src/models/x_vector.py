import tensorflow as tf

from models.nn.tdnn import tdnn_2d, get_gather_idx

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class XVectorModel:
    def __init__(self, batch_size, n_features, n_classes):
        self.input_ = tf.placeholder(tf.float32, [batch_size, n_features, None])
        self.gather_idx_1 = tf.placeholder(tf.int32, shape=None)
        self.tdnn_context_1 = [-2, -1, 0, 1, 2]
        self.gather_idx_2 = tf.placeholder(tf.int32, shape=None)
        self.tdnn_context_2 = [-2, 0, 2]

        tdnn_output = tdnn_2d(self.input_, self.gather_idx_1, self.tdnn_context_1, n_features, batch_size, 512)
        tdnn_output = tdnn_2d(tdnn_output, self.gather_idx_2, self.tdnn_context_2, 512, batch_size, 512)

        self.output_ = tdnn_output

    def start_train(self, batch_loader):
        init = tf.global_variables_initializer()
        with tf.Session(config=config) as sess:
            sess.run(init)
            batch_x, batch_y = batch_loader.next()
            print(batch_x.shape[2])
            gather_idx_1 = get_gather_idx(self.tdnn_context_1, batch_loader.get_batch_size(), batch_x.shape[2])
            print(gather_idx_1.shape)
            gather_idx_2 = get_gather_idx(self.tdnn_context_2, batch_loader.get_batch_size(), gather_idx_1.shape[1])
            output_ = sess.run(self.output_, feed_dict={self.input_: batch_x,
                                                        self.gather_idx_1: gather_idx_1,
                                                        self.gather_idx_2: gather_idx_2})
            print(output_.shape)
