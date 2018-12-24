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


def tdnn(name, input_, context, out_channels, activation=None):
    context = np.array(context)
    context = context if min(context) >= 0 else context - np.array([min(context)] * len(context))
    kernel_height = input_.shape[1]
    kernel_width = max(context) + 1
    in_channels = input_.shape[3]
    mask = [1 if idx in context else 0 for idx in range(kernel_width)]

    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('{}_weights'.format(name),
                                 shape=[kernel_height, out_channels, in_channels, kernel_width], dtype=tf.float32,
                                 initializer=tf.initializers.random_normal(0, 0.01), trainable=True)
        bias = tf.get_variable('{}_bias'.format(name), shape=[out_channels], dtype=tf.float32,
                               initializer=tf.zeros_initializer(), trainable=True)
        kernel = tf.multiply(kernel, tf.cast(tf.constant(mask), dtype=kernel.dtype))
        kernel = tf.transpose(kernel, [0, 3, 2, 1])
        output_ = tf.nn.convolution(input_, kernel, strides=[1, 1], padding="VALID", name=scope.name) + bias

    if activation is not None:
        output_ = activation(output_)
    return tf.transpose(output_, [0, 3, 2, 1])
