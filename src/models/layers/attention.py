import tensorflow as tf


def supervised_attention(input_, context_vector, size):
    hidden_units = input_.shape[2]
    transformed_input = tf.layers.dense(input_, size, activation=tf.nn.tanh)
    transformed_vector = tf.reshape(tf.layers.dense(context_vector, size, activation=tf.nn.tanh), [-1])
    similarity = tf.tensordot(transformed_input, transformed_vector, axes=1)
    alphas = tf.nn.softmax(similarity)
    output = tf.reduce_sum(input_ * tf.expand_dims(alphas, -1), 1)
    return tf.reshape(output, [-1, hidden_units])


def variable_attention(input_, size):
    attention_output = tf.layers.dense(input_, size, activation=tf.tanh)
    attention_vector = tf.Variable(tf.random_normal([size], stddev=0.1), trainable=True)
    similarity = tf.tensordot(attention_output, attention_vector, axes=1)
    alphas = tf.nn.softmax(similarity)
    hidden_units = input_.shape[2]
    output = tf.reduce_sum(input_ * tf.expand_dims(alphas, -1), 1)
    return tf.reshape(output, [-1, hidden_units])
