import numpy as np
import tensorflow as tf

from model.policies import MLPPolicy
from model.controllers import PolicyController


def create_policy_from_params(params, env, sess, scope=None):
    if params['algo'] in ['trpo', 'vime']:
        if not scope:
            scope = "policy"
        return MLPPolicy(
            name=scope,
            observation_space=env.observation_space,
            action_space=env.action_space,
            network_shape=params['policy']['network_shape'],
            activation=params['policy']['activation'],
            init_logstd=params['policy']['init_logstd'],
            sess=sess
        )
    return None


def create_controller_from_policy(policy):
    controller = PolicyController(policy)

    # TODO(GD): check the following reset_opt
    reset_opt = policy.reset_op
    return controller, reset_opt
