import tensorflow as tf


def _append_homog(tensor):
    rank = len(tensor.shape.as_list())
    shape = tf.concat([tf.shape(tensor)[:-1], [1]], axis=0)
    ones = tf.ones(shape, dtype=tensor.dtype)
    return tf.concat([tensor, ones], axis=rank - 1)


def dense(inputs, weights):
    inputs = _append_homog(inputs)
    n_in = inputs.shape.as_list()[-1]
    particles = weights.shape.as_list()[0]
    inputs = tf.reshape(inputs, [particles, -1, n_in])
    preactivations = tf.matmul(inputs, weights)
    preactivations = tf.reshape(preactivations, [-1, weights.get_shape()[-1]])
    activations = tf.nn.relu(preactivations)

    return preactivations, activations
