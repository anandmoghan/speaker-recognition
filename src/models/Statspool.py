import tensorflow as tf
import numpy as np
def statsPool(X):
    X=tf.convert_to_tensor(X, dtype=tf.float32)
    mean, variance = tf.nn.moments(X, axes=[0])
    tf.reshape(mean,[-1,1])
    tf.reshape(variance,[-1,1])
    return tf.concat([mean,variance],0)


