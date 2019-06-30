import tensorflow as tf
import numpy as np
import logger
import libs.bnn.sampler as sp
from libs.bnn.layers import dense


class mlp(object):
    def __init__(self,
                 output_size,
                 scope='dynamics',
                 n_layers=2,
                 size=1000,
                 activation=tf.nn.relu,
                 output_activation=None):
        self.output_size = output_size
        self.scope = scope
        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation

    def __call__(self, input, reuse=False):
        out = input
        with tf.variable_scope(self.scope, reuse=reuse):
            l2_loss = 0.0
            for layer_i in range(self.n_layers):
                layer_name = "dense_{}".format(layer_i)
                out = tf.layers.dense(out, self.size, activation=self.activation, name=layer_name)
                with tf.variable_scope(layer_name, reuse=True):
                    weight = tf.get_variable("kernel")
                    l2_loss += tf.nn.l2_loss(weight)
            out = tf.layers.dense(out, self.output_size, activation=self.output_activation)
        return out, l2_loss


class bnn_mlp(object):
    def __init__(self,
                 sampler,
                 scope='dynamics',
                 n_layers=2,
                 activation=tf.nn.relu,
                 output_activation=None):
        assert(activation == tf.nn.relu)
        assert(output_activation == None)
        self.scope = scope
        self.n_layers = n_layers
        self.sampler = sampler
        self.activation = activation
        self.output_activation = output_activation

    def __call__(self, input, mode, layer_collection=None, var=0.5):
        l2_loss = 0.
        for l in range(1, self.n_layers+2):
            weight = self.sampler.sample(l, mode=mode)
            l2_loss += 0.5 * tf.reduce_sum(tf.reduce_mean(weight ** 2, 0))
            pre, act = dense(input, weight)

            if layer_collection is not None:
                layer_collection.register_fully_connected(self.sampler.get_params(l), input, pre)
            input = act

        if layer_collection is not None:
            layer_collection.register_normal_predictive_distribution(pre, var=var, name="mean")

        return pre, l2_loss


class Dynamics(object):
    def __init__(self):
        self.int_rewards_only = False
        self.ext_rewards_only = False

    def use_intrinsic_rewards_only(self):
        logger.info("Pre-training enabled. Using only intrinsic reward.")
        self.int_rewards_only = True

    def combine_int_and_ext_rewards(self):
        logger.info("Using a combination of external and intrinsic reward.")
        self.int_rewards_only = False
        self.ext_rewards_only = False

    def use_external_rewards_only(self):
        logger.info("Using external reward only.")
        self.int_rewards_only = False
        self.ext_rewards_only = True

    def information_gain(self, obses, acts, next_obses):
        return np.zeros([len(obses),])

    def information_gain_tf(self, obs, act, next_obs):
        raise NotImplementedError

    def process_rewards(self, ext_rewards, obses, actions, next_obses):
        weighted_intrinsic_reward = self.information_gain(obses, actions, next_obses)
        if self.int_rewards_only:
            return weighted_intrinsic_reward
        elif self.ext_rewards_only:
            return ext_rewards
        else:
            return ext_rewards + weighted_intrinsic_reward


class DynamicsModel(Dynamics):
    def __init__(self,
                 env,
                 normalization,
                 batch_size,
                 epochs,
                 val,
                 sess):
        super().__init__()
        self.env = env
        self.normalization = normalization
        self.batch_size = batch_size
        self.epochs = epochs
        self.val = val
        self.sess = sess

        self.obs_dim = env.observation_space.shape[0]
        self.acts_dim = env.action_space.shape[0]
        self.mlp = None

        self.epsilon = 1e-10

    def get_obs_dim(self):
        raise NotImplementedError

    def update_randomness(self):
        pass

    def update_normalization(self, new_normalization):
        self.normalization = new_normalization

    def _build_placeholders(self):
        self.obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_dim))
        self.acts_ph = tf.placeholder(tf.float32, shape=(None, self.acts_dim))
        self.next_obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_dim))

    def _get_feed_dict(self, obs, action, next_obs):
        feed_dict = {self.obs_ph: obs,
                     self.acts_ph: action,
                     self.next_obs_ph: next_obs}
        return feed_dict

    def _get_normalized_obs_and_acts(self, obs, acts):
        normalized_obs = (obs[:, :self.obs_dim] - self.normalization.mean_obs) / (self.normalization.std_obs + self.epsilon)
        normalized_obs = tf.concat([normalized_obs, obs[:, self.obs_dim:]], axis=1)
        normalized_acts = (acts - self.normalization.mean_acts) / (self.normalization.std_acts + self.epsilon)
        return tf.concat([normalized_obs, normalized_acts], 1)

    def _get_predicted_normalized_deltas(self, states, actions):
        normalized_obs_and_acts = self._get_normalized_obs_and_acts(states, actions)
        predicted_normalized_deltas, _ = self.mlp(normalized_obs_and_acts, reuse=True)
        return predicted_normalized_deltas

    def _get_unnormalized_deltas(self, normalized_deltas):
        return normalized_deltas * self.normalization.std_deltas + self.normalization.mean_deltas

    def _add_observations_to_unnormalized_deltas(self, states, unnormalized_deltas):
        return states[:, :self.obs_dim] + unnormalized_deltas

    def _get_normalized_deltas(self, deltas):
        return (deltas - self.normalization.mean_deltas) / (self.normalization.std_deltas + self.epsilon)


class NNDynamicsModel(DynamicsModel):
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 epochs,
                 learning_rate,
                 val,
                 sess,
                 scope="dynamics",
                 reg_coeff=None,
                 controller=None):
        """ Note: Be careful about normalization """
        # Store arguments for later.
        super().__init__(env,
                         normalization,
                         batch_size,
                         epochs,
                         val,
                         sess)
        self.scope = scope
        self.adam_scope = "adam_" + self.scope
        self.controller = controller
        if reg_coeff is None:
            self.reg_coeff = 1.0
        else:
            self.reg_coeff = reg_coeff

        # Build NN placeholders.
        assert(len(env.observation_space.shape) == 1)
        assert(len(env.action_space.shape) == 1)

        self._build_placeholders()

        # Build NN.
        with tf.variable_scope(self.scope):
            self.coeff = tf.get_variable('coeff', initializer=tf.constant(0.001), trainable=False)

        self.mlp = mlp(
            output_size=self.obs_dim,
            scope=self.scope,
            n_layers=n_layers,
            size=size,
            activation=activation,
            output_activation=output_activation)

        # Build cost function and optimizer.
        mse, l2_loss, self.predicted_unnormalized_deltas = self._get_loss()

        self.loss = mse + l2_loss * self.coeff * self.reg_coeff
        self.loss_val = mse
        with tf.variable_scope(self.adam_scope):
            self.update_op = tf.train.AdamOptimizer(learning_rate). \
                minimize(self.loss, var_list=tf.trainable_variables(self.scope))
        dyn_adam_vars = tf.trainable_variables(self.adam_scope)
        self.dyn_adam_init = tf.variables_initializer(dyn_adam_vars)

    def get_obs_dim(self):
        return self.obs_dim

    def fit(self, train_data, val_data):
        self.sess.run(self.dyn_adam_init)
        self.sess.run(tf.assign(self.coeff, 1. / len(train_data)))

        loss = 1000
        best_index = 0
        for epoch in range(self.epochs):
            for (itr, (obs, action, next_obs, _)) in enumerate(train_data):
                feed_dict = self._get_feed_dict(obs, action, next_obs)
                self.sess.run([self.update_op, self.loss], feed_dict=feed_dict)

            if epoch % 5 == 0:
                loss_list = []
                for (itr, (obs, action, next_obs, _)) in enumerate(val_data):
                    feed_dict = self._get_feed_dict(obs, action, next_obs)

                    cur_loss = self.sess.run(self.loss_val, feed_dict=feed_dict)
                    loss_list.append(cur_loss)
                logger.info("Validation loss = {}".format(np.mean(loss_list)))
                if np.mean(loss_list) < loss:
                    loss = np.mean(loss_list)
                    best_index = epoch
                # logger.info("Dynamics optimization | epoch {}/{}: Loss = {}".
                #             format(epoch, self.epochs, np.mean(loss_list)))

                if self.val:
                    if epoch - best_index >= 20:
                        break

    def predict(self, states, actions):
        assert(len(states) == len(actions))
        feed_dict = {
            self.obs_ph: states,
            self.acts_ph: actions,
        }
        unnormalized_deltas = self.sess.run(
            self.predicted_unnormalized_deltas,
            feed_dict=feed_dict)
        return np.array(states)[:, :self.obs_dim] + unnormalized_deltas

    def predict_tf(self, states, actions):
        return self._add_observations_to_unnormalized_deltas(
            states, self._get_unnormalized_deltas(self._get_predicted_normalized_deltas(states, actions)))

    def _get_loss(self):
        deltas = self.next_obs_ph - self.obs_ph
        labels = self._get_normalized_deltas(deltas)
        normalized_obs_and_acts = self._get_normalized_obs_and_acts(self.obs_ph, self.acts_ph)
        predicted_normalized_deltas, l2_loss = self.mlp(normalized_obs_and_acts, reuse=tf.AUTO_REUSE)
        predicted_unnormalized_deltas = self._get_unnormalized_deltas(predicted_normalized_deltas)
        mse_loss = tf.losses.mean_squared_error(
            labels=labels,
            predictions=predicted_normalized_deltas)

        return mse_loss, l2_loss, predicted_unnormalized_deltas


class BBBDynamicsModel(DynamicsModel):
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 epochs,
                 learning_rate,
                 val,
                 particles,
                 ita,
                 obs_var,
                 mode,
                 sess):
        """ Note: Be careful about normalization """
        # Store arguments for later.
        super().__init__(env,
                         normalization,
                         batch_size,
                         epochs,
                         val,
                         sess)

        # Build NN placeholders.
        assert(len(env.observation_space.shape) == 1)
        assert(len(env.action_space.shape) == 1)
        obs_dim = env.observation_space.shape[0]
        acts_dim = env.action_space.shape[0]
        self.obs_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
        self.acts_ph = tf.placeholder(tf.float32, shape=(None, acts_dim))
        self.next_obs_ph = tf.placeholder(tf.float32, shape=(None, obs_dim))
        self.coeff = tf.get_variable('coeff', initializer=tf.constant(0.001), trainable=False)

        # Build NN.
        self.epsilon = 1e-10
        normalized_obs = (self.obs_ph - normalization.mean_obs) / (normalization.std_obs + self.epsilon)
        normalized_acts = (self.acts_ph - normalization.mean_acts) / (normalization.std_acts + self.epsilon)
        normalized_obs_and_acts = tf.concat([normalized_obs, normalized_acts], 1)

        # BNN
        assert(n_layers == 2)
        with tf.variable_scope('dynamics'):
            sampler = sp.sampler(self.coeff, ita, particles, 'bbb')
            sampler.register_block(1, (obs_dim + acts_dim, size))
            sampler.register_block(2, (size, size))
            sampler.register_block(3, (size, obs_dim))
            self.random_update_op = sampler.update_randomness()

        self.mlp = bnn_mlp(
            sampler=sampler,
            scope="dynamics",
            n_layers=n_layers,
            activation=activation,
            output_activation=output_activation)
        predicted_normalized_deltas_train, _ = self.mlp(normalized_obs_and_acts, mode="train")
        normalized_obs_and_acts_rep = tf.tile(normalized_obs_and_acts, [particles, 1])
        predicted_normalized_deltas_val, _ = self.mlp(normalized_obs_and_acts_rep, mode="val")
        predicted_normalized_deltas_val = tf.reduce_mean(tf.reshape(predicted_normalized_deltas_val, [particles, -1, obs_dim]), 0)

        predicted_normalized_deltas_test, _ = self.mlp(normalized_obs_and_acts, mode=mode)
        self.predicted_unnormalized_deltas = predicted_normalized_deltas_test * \
                                             normalization.std_deltas + normalization.mean_deltas

        # Build cost function and optimizer.
        normalized_deltas = ((self.next_obs_ph - self.obs_ph) - normalization.mean_deltas) \
                            / (normalization.std_deltas + self.epsilon)
        mse = tf.losses.mean_squared_error(
            labels=normalized_deltas,
            predictions=predicted_normalized_deltas_train) / obs_var
        self.kl_div = sampler.kl_div()
        self.loss = mse + self.kl_div * self.coeff

        # validation
        self.loss_val = tf.losses.mean_squared_error(
            labels=normalized_deltas,
            predictions=predicted_normalized_deltas_val)

        with tf.variable_scope('adam_dynamics'):
            self.update_op = tf.train.AdamOptimizer(learning_rate).\
                minimize(self.loss, var_list=tf.trainable_variables("dynamics"))

        dyn_adam_vars = tf.global_variables('adam_dynamics')
        self.reset_opt_op = tf.variables_initializer(dyn_adam_vars)

    def fit(self, train_data, val_data):
        self.sess.run(self.reset_opt_op)
        self.sess.run(tf.assign(self.coeff, 1. / len(train_data)))

        loss = 1000
        best_index = 0
        total_iter = 0
        for epoch in range(self.epochs):
            for (itr, (obs, action, next_obs, _)) in enumerate(train_data):
                feed_dict = {
                    self.obs_ph: obs,
                    self.acts_ph: action,
                    self.next_obs_ph: next_obs,
                }
                _, elbo = self.sess.run([self.update_op, self.loss], feed_dict=feed_dict)
                total_iter += 1

            if epoch % 5 == 0:
                loss_list = []
                for (itr, (obs, action, next_obs, _)) in enumerate(val_data):
                    feed_dict = {
                        self.obs_ph: obs,
                        self.acts_ph: action,
                        self.next_obs_ph: next_obs,
                    }
                    cur_loss = self.sess.run(self.loss_val, feed_dict=feed_dict)
                    loss_list.append(cur_loss)
                if np.mean(loss_list) < loss:
                    loss = np.mean(loss_list)
                    best_index = epoch
                kl = self.sess.run(self.kl_div)
                logger.log("Dynamics optimization | epoch {}/{}: Loss = {}, KL = {}".
                           format(epoch, self.epochs, np.mean(loss_list), kl))
                if self.val:
                    if epoch - best_index >= 20 and total_iter >= 1000:
                        break
                else:
                    if total_iter >= 2000 and epoch >= 50:
                        break

    def predict(self, states, actions):
        assert(len(states) == len(actions))
        feed_dict = {
            self.obs_ph: states,
            self.acts_ph: actions,
        }
        unnormalized_deltas = self.sess.run(
            self.predicted_unnormalized_deltas,
            feed_dict=feed_dict)

        return states + unnormalized_deltas

    def predict_tf(self, states, actions):
        raise ValueError("Not Implemented.")

    def update_randomness(self):
        self.sess.run(self.random_update_op)


class NNGDynamicsModel(DynamicsModel):
    def __init__(self,
                 env,
                 n_layers,
                 size,
                 activation,
                 output_activation,
                 normalization,
                 batch_size,
                 epochs,
                 learning_rate,
                 val,
                 particles,
                 ita,
                 obs_var,
                 mode,
                 kfac_params,
                 sess):
        """ Note: Be careful about normalization """
        # Store arguments for later.
        super().__init__(env,
                         normalization,
                         batch_size,
                         epochs,
                         val,
                         sess)
        self.obs_var = obs_var
        self.controller = None
        self.particles = particles
        self.mode = mode

        # Build NN placeholders.
        assert(len(env.observation_space.shape) == 1)
        assert(len(env.action_space.shape) == 1)
        self._build_placeholders()
        self.coeff = tf.get_variable('coeff', initializer=tf.constant(0.001), trainable=False)
        # self.logvar = tf.get_variable('logvar', initializer=-tf.ones(shape=[obs_dim]))

        # Build NN.

        # BNN
        with tf.variable_scope('dynamics'):
            sampler = sp.sampler(self.coeff, ita, particles, 'kron')
            sampler.register_block(1, (self.obs_dim + self.acts_dim, size))
            for block_i in range(2, n_layers + 1):
                sampler.register_block(block_i, (size, size))
            sampler.register_block(n_layers + 1, (size, self.obs_dim))
            self.random_update_op = sampler.update_randomness()

        self.mlp = bnn_mlp(
            sampler=sampler,
            scope="dynamics",
            n_layers=n_layers,
            activation=activation,
            output_activation=output_activation)

        from libs.kfac import layer_collection as lc
        layer_collection = lc.LayerCollection()

        mse, l2_loss, predicted_normalized_deltas_val, normalized_deltas, normalized_obs_and_acts_rep = \
            self._get_loss(layer_collection)

        self.loss = mse + l2_loss * self.coeff / ita

        # validation
        self.loss_val = tf.losses.mean_squared_error(
            labels=normalized_deltas,
            predictions=predicted_normalized_deltas_val)

        # TODO(GD): write KFAC op
        from libs.kfac import optimizer as opt
        with tf.variable_scope('kfac_dynamics'):
            self.optim = optim = opt.KFACOptimizer(learning_rate=kfac_params['learning_rate'],
                                                   cov_ema_decay=kfac_params['cov_ema_decay'],
                                                   damping=kfac_params['damping'], layer_collection=layer_collection,
                                                   norm_constraint=kfac_params['kl_clip'], momentum=kfac_params['momentum'],
                                                   # cov_devices=['/cpu:0'], inv_devices=['/cpu:0'],
                                                   var_list=tf.trainable_variables("dynamics"))
            #
            # optim2 = tf.train.AdamOptimizer(learning_rate=3e-4)

        self.cov_update_op = optim.cov_update_op
        self.inv_update_op = optim.inv_update_op
        self.update_op = optim.minimize(self.loss)
        with tf.control_dependencies([self.inv_update_op]):
            self.var_update_op = sampler.update(layer_collection.get_blocks())
        self.random_update_op = sampler.update_randomness()
        self.reset_opt_op = tf.variables_initializer([optim.get_slot(var, name) for name in optim.get_slot_names() for var in tf.trainable_variables("dynamics")])

        # for compute the information gain, using the mode
        prediction_info_gain, _ = self.mlp(normalized_obs_and_acts_rep, mode="test")
        normalized_deltas_rep = tf.tile(normalized_deltas, [particles, 1])

        quad = tf.reduce_sum((prediction_info_gain - normalized_deltas_rep) ** 2, -1)
        quad = tf.reshape(quad, [particles, -1])
        quad -= tf.reduce_min(quad, 0)
        likelihood = tf.exp(-quad)
        prob = likelihood / tf.reduce_sum(likelihood, 0)
        prob = tf.check_numerics(prob, message="prob invalid number")
        entropy = - prob * tf.log(prob + 1e-8)
        entropy = tf.where(tf.is_nan(entropy), tf.zeros_like(entropy), entropy)
        self.info_gain = tf.log(tf.to_float(particles)) - tf.reduce_sum(entropy, 0)

    def fit(self, train_data, val_data):
        self.sess.run(self.reset_opt_op)
        self.sess.run(tf.assign(self.coeff, 1. / len(train_data) * self.obs_var))

        loss = 1000
        best_index = 0
        total_iter = 0
        for epoch in range(self.epochs):
            for (itr, (obs, action, next_obs, _)) in enumerate(train_data):
                feed_dict = self._get_feed_dict(obs, action, next_obs)
                self.sess.run([self.update_op, self.cov_update_op], feed_dict=feed_dict)
                total_iter += 1
                if total_iter % 20 == 0:
                    self.sess.run(self.var_update_op)

            if epoch % 5 == 0:
                loss_list = []
                for (itr, (obs, action, next_obs, _)) in enumerate(val_data):
                    feed_dict = self._get_feed_dict(obs, action, next_obs)
                    cur_loss = self.sess.run(self.loss_val, feed_dict=feed_dict)
                    loss_list.append(cur_loss)
                if np.mean(loss_list) < loss:
                    loss = np.mean(loss_list)
                    best_index = epoch
                logger.log("Dynamics optimization | epoch {}/{}: Loss = {}".
                           format(epoch, self.epochs, np.mean(loss_list)))
                if self.val:
                    if epoch - best_index >= 20 and total_iter >= 1000:
                        break
                else:
                    if total_iter >= 2000 and epoch >= 50:
                        break

    def predict(self, states, actions):
        assert(len(states) == len(actions))
        feed_dict = {
            self.obs_ph: states,
            self.acts_ph: actions,
        }
        unnormalized_deltas = self.sess.run(
            self.predicted_unnormalized_deltas,
            feed_dict=feed_dict)

        return states + unnormalized_deltas

    def predict_tf(self, states, actions):
        normalized_obs = (states - self.normalization.mean_obs) / (self.normalization.std_obs + self.epsilon)
        normalized_acts = (actions - self.normalization.mean_acts) / (self.normalization.std_acts + self.epsilon)
        normalized_obs_and_acts = tf.concat([normalized_obs, normalized_acts], 1)
        normalized_deltas, _ = self.mlp(normalized_obs_and_acts, mode="val")

        unnormalized_deltas = self.normalization.mean_deltas + self.normalization.std_deltas * normalized_deltas
        return states + unnormalized_deltas

    def update_randomness(self):
        self.sess.run(self.random_update_op)

    def information_gain(self, obs, act, next_obs):
        feed_dict = {self.obs_ph: obs, self.acts_ph: act, self.next_obs_ph: next_obs}
        return self.sess.run(self.info_gain, feed_dict)

    def _get_loss(self, layer_collection):
        normalized_obs_and_acts = self._get_normalized_obs_and_acts(self.obs_ph, self.acts_ph)
        predicted_normalized_deltas_train, l2_loss = self.mlp(normalized_obs_and_acts, mode="train",
                                                              layer_collection=layer_collection)
        normalized_obs_and_acts_rep = tf.tile(normalized_obs_and_acts, [self.particles, 1])
        predicted_normalized_deltas_val, _ = self.mlp(normalized_obs_and_acts_rep, mode="val")
        predicted_normalized_deltas_val = tf.reduce_mean(
            tf.reshape(predicted_normalized_deltas_val, [self.particles, -1, self.obs_dim]), 0)
        predicted_normalized_deltas_test, _ = self.mlp(normalized_obs_and_acts, mode=self.mode)
        self.predicted_unnormalized_deltas = self._get_unnormalized_deltas(predicted_normalized_deltas_test)

        # Build cost function and optimizer.
        deltas = self.next_obs_ph -self.obs_ph
        normalized_deltas = self._get_normalized_deltas(deltas)
        mse = tf.losses.mean_squared_error(
            labels=normalized_deltas,
            predictions=predicted_normalized_deltas_train)
        return mse, l2_loss, predicted_normalized_deltas_val, normalized_deltas, normalized_obs_and_acts_rep
