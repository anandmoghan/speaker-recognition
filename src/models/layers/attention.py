import tensorflow as tf


def variable_attention(input_, size):
    attention_output = tf.layers.dense(input_, size, activation=tf.tanh)
    attention_vector = tf.Variable(tf.random_normal([size], stddev=0.1))
    similarity = tf.tensordot(attention_output, attention_vector, axes=1)
    alphas = tf.nn.softmax(similarity)
    hidden_units = input_.shape[2]
    output = tf.reduce_sum(input_ * tf.expand_dims(alphas, -1), 1)
    return tf.reshape(output, [-1, hidden_units])
