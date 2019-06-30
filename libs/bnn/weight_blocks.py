from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import abc
distributions = tf.distributions


def _compute_pi_tracenorm(left_cov, right_cov):
    left_norm = tf.trace(left_cov) * right_cov.shape.as_list()[0]
    right_norm = tf.trace(right_cov) * left_cov.shape.as_list()[0]
    return tf.sqrt(left_norm / right_norm)


class WeightBlock(object):
    def __init__(self, idx, shape, coeff, ita, particles):
        self._shape = shape
        self._n_in = np.prod(shape[:-1]) + 1
        self._n_out = shape[-1]
        self._coeff = coeff
        self._ita = ita
        self._particles = particles

        self._build_weights(idx)
        self._init_random(idx, particles)

    def _build_weights(self, idx):
        self._weight = tf.get_variable(
            'w'+str(idx)+'_weight',
            shape=self._shape,
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True
        )
        self._bias = tf.get_variable(
            'w'+str(idx)+'_bias',
            shape=[self._n_out],
            initializer=tf.constant_initializer(0.),
            trainable=True
        )

    def _init_random(self, idx, particles):
        self._rand = tf.get_variable(
            'w'+str(idx)+'rand',
            shape=[particles, self._n_in, self._n_out],
            initializer=tf.constant_initializer(0.),
            trainable=False
        )

    def params(self):
        return (self._weight, self._bias)

    @property
    def _mean(self):
        weight = tf.reshape(self._weight, (self._n_in-1, self._n_out))
        bias = tf.expand_dims(self._bias, 0)
        return tf.concat([weight, bias], 0)

    @abc.abstractmethod
    def sample(self, particles):
        pass

    @abc.abstractmethod
    def update_randomness(self):
        pass


class BBBBlock(WeightBlock):
    def __init__(self, idx, shape, coeff, ita, particles):
        super(BBBBlock, self).__init__(idx, shape, coeff, ita, particles)
        self._logstd = tf.get_variable(
            'w'+str(idx)+'_logstd',
            shape=[self._n_in, self._n_out],
            initializer=tf.constant_initializer(-3.),
            trainable=True
        )

    def sample(self, mode):
        mean = self._mean
        out_mean = tf.expand_dims(mean, 0)
        if mode == "train":
            rand = tf.random_normal(shape=tf.shape(self._mean))
            out_std = tf.log(1. + tf.exp(self._logstd))
            out_rand = rand * out_std
            return out_mean + out_rand
        elif mode == "val":
            rand = tf.random_normal(shape=[self._particles, self._n_in, self._n_out])
            out_std = tf.log(1. + tf.exp(self._logstd))
            out_rand = rand * out_std
            return out_mean + out_rand
        elif mode == "random":
            rand = tf.random_normal(shape=tf.shape(self._mean))
            out_std = tf.log(1. + tf.exp(self._logstd))
            out_rand = rand * out_std
            return out_mean + out_rand
        elif mode == "mode":
            return out_mean
        elif mode == "test":
            return out_mean + self._rand

    def update_randomness(self):
        rand = tf.random_normal(shape=[self._particles, self._n_in, self._n_out])
        std = tf.log(1. + tf.exp(self._logstd))
        return self._rand.assign(rand * std)

    def kl_div(self):
        kl = distributions.kl_divergence(distributions.Normal(self._mean, tf.log(1. + tf.exp(self._logstd))),
                                         distributions.Normal(tf.zeros(tf.shape(self._mean)), tf.sqrt(self._ita)
                                                              * tf.ones(tf.shape(self._logstd))))

        return tf.reduce_sum(kl)


class MVGBlock(WeightBlock):
    def __init__(self, idx, shape, coeff, ita, particles):
        super(MVGBlock, self).__init__(idx, shape, coeff, ita, particles)
        self._u_c = tf.get_variable(
            'w'+str(idx)+'_u_c',
            initializer=1e-3 * tf.eye(self._n_in),
            trainable=False
        )
        self._v_c = tf.get_variable(
            'w'+str(idx)+'_v_c',
            initializer=1e-3 * tf.eye(self._n_out),
            trainable=False
        )

    def sample(self, mode):
        mean = self._mean
        out_mean = tf.expand_dims(mean, 0)
        if mode == "train":
            rand = tf.random_normal(shape=tf.shape(self._mean))
            out_rand = tf.matmul(self._u_c, tf.matmul(rand, self._v_c, transpose_b=True))
            return out_mean + out_rand
        elif mode == "val":
            rand = tf.random_normal(shape=[self._particles, self._n_in, self._n_out])
            u_c = tf.tile(tf.expand_dims(self._u_c, 0), [self._particles, 1, 1])
            v_c = tf.tile(tf.expand_dims(self._v_c, 0), [self._particles, 1, 1])
            out_rand = tf.matmul(u_c, tf.matmul(rand, v_c, transpose_b=True))
            return out_mean + out_rand
        elif mode == "random":
            rand = tf.random_normal(shape=tf.shape(self._mean))
            out_rand = tf.matmul(self._u_c, tf.matmul(rand, self._v_c, transpose_b=True))
            return out_mean + out_rand
        elif mode == "mode":
            return out_mean
        elif mode == "test":
            return out_mean + self._rand

    def update(self, block):
        input_factor = block._input_factor
        output_factor = block._output_factor
        pi = _compute_pi_tracenorm(input_factor.get_cov(), output_factor.get_cov())
        # check the following code
        coeff = self._coeff / block._renorm_coeff
        coeff = coeff ** 0.5
        damping = coeff / (self._ita ** 0.5)

        ue, uv = tf.self_adjoint_eig(
            input_factor.get_cov() / pi + damping * tf.eye(self._u_c.shape.as_list()[0]))
        ve, vv = tf.self_adjoint_eig(
            output_factor.get_cov() * pi + damping * tf.eye(self._v_c.shape.as_list()[0]))

        ue = coeff / tf.maximum(ue, damping)
        new_uc = uv * ue ** 0.5

        ve = coeff / tf.maximum(ve, damping)
        new_vc = vv * ve ** 0.5

        updates_op = []
        updates_op.append(self._u_c.assign(new_uc))
        updates_op.append(self._v_c.assign(new_vc))

        return tf.group(*updates_op)

    def update_randomness(self):
        rand = tf.random_normal(shape=[self._particles, self._n_in, self._n_out])
        u_c = tf.tile(tf.expand_dims(self._u_c, 0), [self._particles, 1, 1])
        v_c = tf.tile(tf.expand_dims(self._v_c, 0), [self._particles, 1, 1])
        rand = tf.matmul(u_c, tf.matmul(rand, v_c, transpose_b=True))
        return self._rand.assign(rand)

    def multiply(self, vector):
        # vector = tf.reduce_mean(vector, 0)
        # out = tf.matmul(self._u_c, tf.matmul(vector, self._v_c, transpose_b=True))
        # return out

        particles = vector.shape.as_list()[0]
        u_c = tf.tile(tf.expand_dims(self._u_c, 0), [particles, 1, 1])
        v_c = tf.tile(tf.expand_dims(self._v_c, 0), [particles, 1, 1])
        out = tf.matmul(u_c, tf.matmul(vector, v_c, transpose_b=True))

        return tf.reshape(out, [particles, -1])

