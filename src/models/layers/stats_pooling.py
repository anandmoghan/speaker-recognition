import tensorflow as tf


def stats_pool(x, axes=0):
    mean, variance = tf.nn.moments(x, axes)
    return tf.concat([mean, variance], 1)


