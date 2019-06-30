import tensorflow as tf

from model.dynamics import NNDynamicsModel, BBBDynamicsModel, NNGDynamicsModel
from model.dynamics_ensemble import DynamicsModelEnsemble
from model.stochastic_dynamics import PNNDynamicsModel
from libs.misc.initial_configs.tf_swish import swish


def get_activation(activation_type):
    if activation_type == "relu":
        return tf.nn.relu
    elif activation_type == "selu":
        return tf.nn.selu
    elif activation_type == "swish":
        return swish
    raise NotImplementedError("Configuration for 'activation' must be one of "
                              "('relu', 'selu' and 'swish')")


def get_nn_dynamic_model_params(params, env, normalization, sess, controller):
    return {
        "env": env,
        "n_layers": params["dynamics"]["n_layers"],
        "size": params["dynamics"]["hidden_size"],
        "activation": get_activation(params["dynamics"]["activation"]),
        "output_activation": None,
        "normalization": normalization,
        "batch_size": params["dynamics"]["batch_size"],
        "epochs": params["dynamics"]["epochs"],
        "learning_rate": params["dynamics"]["learning_rate"],
        "val": params["dynamics"]["val"],
        "sess": sess,
        "controller": controller,
    }


def get_bbb_dynamic_model_params(params, env, normalization, sess):
    return {
        "env": env,
        "n_layers": params["dynamics"]["n_layers"],
        "size": params["dynamics"]["hidden_size"],
        "activation": get_activation(params["dynamics"]["activation"]),
        "output_activation": None,
        "normalization": normalization,
        "batch_size": params["dynamics"]["batch_size"],
        "epochs": params["dynamics"]["epochs"],
        "learning_rate": params["dynamics"]["learning_rate"],
        "val": params["dynamics"]["val"],
        "particles": params["dynamics"]["particles"],
        "ita": params["dynamics"]["ita"],
        "obs_var": params["dynamics"]["obs_var"],
        "mode": params["dynamics"]["mode"],
        "sess": sess
    }


def get_nng_dynamic_model_params(params, env, normalization, sess):
    return {
        "env": env,
        "n_layers": params["dynamics"]["n_layers"],
        "size": params["dynamics"]["hidden_size"],
        "activation": get_activation(params["dynamics"]["activation"]),
        "output_activation": None,
        "normalization": normalization,
        "batch_size": params["dynamics"]["batch_size"],
        "epochs": params["dynamics"]["epochs"],
        "learning_rate": params["dynamics"]["learning_rate"],
        "val": params["dynamics"]["val"],
        "particles": params["dynamics"]["particles"],
        "ita": params["dynamics"]["ita"],
        "obs_var": params["dynamics"]["obs_var"],
        "mode": params["dynamics"]["mode"],
        "kfac_params": params["dynamics"]["kfac_params"],
        "sess": sess
    }


def get_pnn_dynamic_model_params(params, env, normalization, sess):
    return {
        "env": env,
        "n_layers": params["dynamics"]["n_layers"],
        "size": params["dynamics"]["hidden_size"],
        "activation": get_activation(params["dynamics"]["activation"]),
        "output_activation": None,
        "normalization": normalization,
        "batch_size": params["dynamics"]["batch_size"],
        "epochs": params["dynamics"]["epochs"],
        "learning_rate": params["dynamics"]["learning_rate"],
        "val": params["dynamics"]["val"],
        "sess": sess
    }


def create_dynamics_model(params, env, normalization, sess, controller=None):
    if params['dynamics']['model'] == 'nn':
        nn_dynamic_model_params = get_nn_dynamic_model_params(params, env, normalization, sess, controller)
        if params['dynamics'].get('ensemble', False):
            dyn_model = DynamicsModelEnsemble(
                NNDynamicsModel,
                params['dynamics']['ensemble_model_count'],
                enable_particle_ensemble=params["dynamics"].get("enable_particle_ensemble", False),
                particles=params["dynamics"].get("particles"),
                intrinsic_reward_coeff=params["dynamics"].get("intrinsic_reward_coeff"),
                obs_var=params["dynamics"].get("obs_var")
            )
            dyn_model.init_dynamic_models(**nn_dynamic_model_params)
        else:
            dyn_model = NNDynamicsModel(**nn_dynamic_model_params)
    elif params['dynamics']['model'] == 'bbb':
        dyn_model = BBBDynamicsModel(
            **get_bbb_dynamic_model_params(params, env, normalization, sess))
    elif params['dynamics']['model'] == 'nng':
        dyn_model = NNGDynamicsModel(
            **get_nng_dynamic_model_params(params, env, normalization, sess))
    elif params['dynamics']['model'] == 'pnn':
        pnn_dynamic_model_params = get_pnn_dynamic_model_params(params, env, normalization, sess)
        if params['dynamics'].get('ensemble', False):
            dyn_model = DynamicsModelEnsemble(
                PNNDynamicsModel,
                params['dynamics']['ensemble_model_count'],
                enable_particle_ensemble=params['dynamics'].get('enable_particle_ensemble', False),
                particles=params['dynamics'].get('particles')
            )
            dyn_model.init_dynamic_models(**pnn_dynamic_model_params)
        else:
            dyn_model = PNNDynamicsModel(**pnn_dynamic_model_params)
    else:
        raise NotImplementedError("Dynamics model {} not recognized!".format(params['dynamics']['model']))
    return dyn_model
