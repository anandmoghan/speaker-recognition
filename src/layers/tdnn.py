import tensorflow as tf
import numpy as np


def get_gather_idx(context, batch_size, n_frames):
    start = 0 if context[0] >= 0 else -1 * context[0]
    end = n_frames if context[-1] <= 0 else n_frames - context[-1]
    gather_idx = []
    for b in range(batch_size):
        for c in context:
            gather_idx.append([[b, c + f] for f in range(start, end)])
    return np.array(gather_idx)


def tdnn_2d(input_, gather_idx, context, n_features, batch_size, output_size):
    n_features = len(context) * n_features
    input_ = tf.transpose(input_, [0, 2, 1])
    input_ = tf.gather_nd(input_, gather_idx)
    input_ = tf.transpose(input_, [0, 2, 1])
    input_ = tf.reshape(input_, [batch_size, n_features, -1])
    input_ = tf.expand_dims(input_, 3)
    output_ = tf.layers.conv2d(input_, filters=output_size, kernel_size=(n_features, 1))
    output_ = tf.transpose(tf.squeeze(output_), [0, 2, 1])
    return output_
