import numpy as np
import tensorflow as tf
from libs.misc import tf_networks


class MLPPolicy(object):
    def __init__(
        self,
        name,
        observation_space,
        action_space,
        init_logstd=0.0,
        network_shape=(64, 64),
        activation='tanh',
        sess=None
    ):
        self.name = name
        self.running_stats = None
        self.sess = sess

        obs_dim = observation_space.flat_dim
        act_dim = action_space.flat_dim

        network_shape = [obs_dim] + list(network_shape) + [act_dim]
        num_layer = len(network_shape) - 1
        act_type = [activation] * (num_layer - 1) + [None]

        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )
        init_data[-1]['w_init_para']['stddev'] = 0.01  # the output layer std
        self.mean_network = tf_networks.MLP(
            dims=network_shape, scope=name,
            activation=act_type, init_data=init_data
        )

        self.obs_ph = tf.placeholder(tf.float32, [None, obs_dim])

        self.mean = self.mean_network(self.obs_ph)
        with tf.variable_scope(self.name):
            self.logstd = tf.get_variable("logstd", shape=[1, act_dim],
                                     initializer=tf.constant_initializer(init_logstd),
                                     trainable=True)
            self.reset_op = tf.assign(self.logstd, tf.zeros(shape=[1, act_dim]))

            self.init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name))

    @property
    def vectorized(self):
        return True

    @property
    def trainable_variables(self):
        return tf.trainable_variables(self.name)

    def initialize_variables(self):
        self.init_op.run()

    def get_action(self, ob):
        ob = np.reshape(ob, [1, -1])
        normlaized_ob = self.running_stats.apply_norm_np(ob)
        mean, logstd = self.sess.run([self.mean, self.logstd],
                                     feed_dict={self.obs_ph: normlaized_ob})

        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(logstd) + mean
        return action[0], dict(mean=mean, logstd=logstd)

    def get_actions(self, obs):
        normlaized_obs = self.running_stats.apply_norm_np(obs)
        means, logstds = self.sess.run([self.mean, self.logstd],
                                       feed_dict={self.obs_ph: normlaized_obs})
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(logstds) + means
        return actions, dict(mean=means, logstd=np.tile(logstds, [np.shape(means)[0], 1]))

    def get_actions_tf(self, obs):
        normlaized_obs = self.running_stats.apply_norm_tf(obs)
        means = self.mean_network(normlaized_obs, reuse=True)
        rnd = tf.random_normal(shape=obs.shape)
        actions = rnd * tf.exp(self.logstd) + means
        return actions, None

    def get_dist_tf(self, obs):
        normlaized_obs = self.running_stats.apply_norm_tf(obs)
        return self.mean_network(normlaized_obs, reuse=True), self.logstd

    def add_running_stats(self, running_stats):
        if self.running_stats is not None:
            raise ValueError
        else:
            self.running_stats = running_stats
