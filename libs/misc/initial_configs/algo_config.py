from envs.neural_env import NeuralNetEnv
from model.baselines import LinearFeatureBaseline
from algos.trpo import TRPO as TRPO_mbrl


def get_base_trpo_args(params, env, inner_env, policy, dyn_model, sess, scope=None):
    neural_env = NeuralNetEnv(env=env,
                              inner_env=inner_env,
                              dynamics=dyn_model)

    # add options for MLPBaseline
    baseline = LinearFeatureBaseline(name="baseline")
    return {
        "env": neural_env,
        "inner_env": inner_env,
        "policy": policy,
        "baseline": baseline,
        "batch_size": params['trpo']['batch_size'],
        "max_path_length": params['trpo']['horizon'],
        "discount": params['trpo']['gamma'],
        "target_kl": params['trpo']['step_size'],
        "gae_lambda": params['trpo']['gae'],
        "sess": sess,
        "scope": scope
    }


def get_vime_trpo_args(params, env, inner_env, policy, dyn_model, sess, scope=None):
    vime_trpo_args = get_base_trpo_args(params, env, inner_env, policy, dyn_model, sess, scope=scope)
    vime_trpo_args["dynamics"] = dyn_model
    vime_trpo_args["eta"] = params['vime']['eta']
    return vime_trpo_args


def create_trpo_algo(params, env, inner_env, policy, dyn_model, sess, scope=None):
    if params['algo'] == 'trpo':
        return TRPO_mbrl(**get_base_trpo_args(params, env, inner_env, policy, dyn_model, sess, scope=scope))
    elif params['algo'] == 'vime':
        return TRPO_mbrl(**get_vime_trpo_args(params, env, inner_env, policy, dyn_model, sess, scope=scope))
    else:
        raise ValueError("To create TRPO algo, params['algo'] must be one of ({}, {})".format("'trpo'", "'vime'"))
