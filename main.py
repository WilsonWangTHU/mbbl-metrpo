import numpy as np
import tensorflow as tf
from libs.misc.initial_configs.algo_config import create_trpo_algo
from libs.misc.initial_configs.policy_config import create_policy_from_params, create_controller_from_policy

import logger
from libs.misc.data_handling.utils import add_path_data_to_collection_and_update_normalization
from libs.misc.data_handling.data_collection import DataCollection
from libs.misc.data_handling.path_collection import PathCollection
from libs.misc.data_handling.rollout_sampler import RolloutSampler
from libs.misc.initial_configs.dynamics_model_config import create_dynamics_model
from libs.misc.saving_and_loading import save_cur_iter_dynamics_model, \
    confirm_restoring_dynamics_model, restore_model
from libs.misc.utils import get_session, get_env, get_inner_env
from params_preprocessing import parse_args, process_params


def log_tabular_results(returns, itr, train_collection):
    logger.clear_tabular()
    logger.record_tabular('Iteration', itr)
    logger.record_tabular('AverageReturn', np.mean(returns))
    logger.record_tabular('MinimumReturn', np.min(returns))
    logger.record_tabular('MaximumReturn', np.max(returns))
    logger.record_tabular('TotalSamples', train_collection.get_total_samples())

    logger.dump_tabular()


def get_data_from_random_rollouts(params, env, normalization_scope=None):
    train_collection = DataCollection(batch_size=params['dynamics']['batch_size'],
                                      max_size=params['max_train_data'],
                                      shuffle=True)
    val_collection = DataCollection(batch_size=params['dynamics']['batch_size'],
                                    max_size=params['max_val_data'],
                                    shuffle=False)
    rollout_sampler = RolloutSampler(env)
    random_paths = rollout_sampler.generate_random_rollouts(
        num_paths=params['num_path_random'],
        horizon=params['env_horizon']
    )
    path_collection = PathCollection()
    obs_dim = env.observation_space.shape[0]
    normalization = add_path_data_to_collection_and_update_normalization(random_paths, path_collection,
                                                                         train_collection, val_collection,
                                                                         normalization=None,
                                                                         obs_dim=obs_dim,
                                                                         normalization_scope=normalization_scope)
    return train_collection, val_collection, normalization, path_collection, rollout_sampler


def train_policy_trpo(params, algo, dyn_model, iterations):
    algo.start_worker()
    for j in range(iterations):
        paths, _ = algo.obtain_samples(j, dynamics=dyn_model)
        samples_data = algo.process_samples(j, paths)
        algo.optimize_policy(j, samples_data)
        # algo.update_stats(paths)
        algo.fit_baseline(paths)

        # dump this tabular result only for debug purposes
        # logger.dump_tabular()

    algo.shutdown_worker()


def pre_train_dynamics(params, dyn_model, policy, algo, reset_opt, sess,
                       path_collection, train_collection, val_collection, normalization, rollout_sampler):
    dyn_model.use_intrinsic_rewards_only()

    pre_train_itr = params["dynamics"].get("pre_training", {}).get("itr", 0)
    logger.info("Pre-training dynamics model for {} iterations...".format(pre_train_itr))
    tf.global_variables_initializer().run()

    for itr in range(pre_train_itr):
        logger.info('Pre-training itr #{} |'.format(itr))
        dyn_model.fit(train_collection, val_collection)
        rollout_sampler.update_dynamics(dyn_model)
        dyn_model.update_randomness()

        sess.run(reset_opt)

        if params['policy'].get('reinitialize_every_itr', False):
            logger.info("Re-initialize policy variables")
            policy.initialize_variables()

        train_policy_trpo(params, algo, dyn_model,
                          params["dynamics"]["pre_training"]["policy_itr"])
        rl_paths = rollout_sampler.sample(
            num_paths=params['num_path_onpol'],
            horizon=params['env_horizon'],
            visualize=params.get("rollout_visualization", False),
            visualize_path_no=params.get("rollout_record_path_no"),
        )

        returns = np.array([sum(path["rewards"]) for path in rl_paths])
        log_tabular_results(returns, itr, train_collection)

        normalization = add_path_data_to_collection_and_update_normalization(
            rl_paths, path_collection,
            train_collection, val_collection,
            normalization)
    logger.info("Done pre-training dynamics model.")


def train(params):

    sess = get_session(interactive=True)
    env = get_env(params['env_name'], params.get('video_dir'))
    # TODO(GD): change to replay_buffer
    inner_env = get_inner_env(env)

    train_collection, val_collection, normalization, path_collection, rollout_sampler = \
        get_data_from_random_rollouts(params, env)

    # ############################################################
    # ############### create computational graph #################
    # ############################################################
    policy = create_policy_from_params(params, env, sess)

    controller, reset_opt = create_controller_from_policy(policy)
    dyn_model = create_dynamics_model(params, env, normalization, sess)

    rollout_sampler.update_controller(controller)
    if params['algo'] not in ('trpo', 'vime'):
        raise NotImplementedError

    algo = create_trpo_algo(params, env, inner_env, policy, dyn_model, sess)

    # ############################################################
    # ######################### learning #########################
    # ############################################################

    # init global variables
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=None)
    policy_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy")
    all_var_except_policy = [v for v in all_variables if v not in policy_variables]

    train_dyn_with_intrinsic_reward_only = params["dynamics"].get("intrinsic_reward_only", False)
    logger.log("Train dynamics model with intrinsic reward only? {}".format(train_dyn_with_intrinsic_reward_only))
    if train_dyn_with_intrinsic_reward_only:
        external_evaluation_interval = params["dynamics"]["external_reward_evaluation_interval"]
        policy_ext = create_policy_from_params(params, env, sess, scope='policy_ext_reward')
        controller_ext, reset_opt_ext = create_controller_from_policy(policy_ext)
        algo_ext = create_trpo_algo(params, env, inner_env, policy_ext, dyn_model, sess, scope="trpo_ext_reward")
        rollout_sampler_ext = RolloutSampler(env, controller=controller_ext)
    else:
        external_evaluation_interval = None
        policy_ext = None
        algo_ext = None
        rollout_sampler_ext = None

    saver = tf.train.Saver(var_list=all_var_except_policy)
    tf.global_variables_initializer().run()

    start_itr = params.get("start_onpol_iter", 0)
    end_itr = params['onpol_iters']

    # Pre-training
    pretrain_mode = params["dynamics"].get("pre_training", {}).get("mode")
    pretrain_itr = params["dynamics"].get("pre_training", {}).get("itr", 0)

    if pretrain_mode == "intrinsic_reward":
        pre_train_dynamics(params, dyn_model, policy, algo, reset_opt, sess,
                           path_collection, train_collection, val_collection, normalization, rollout_sampler)

    elif pretrain_mode == "random":
        logger.log("Baseline without pre-training. Generating random rollouts to match pre-train samples.")
        rl_paths = rollout_sampler.generate_random_rollouts(
            num_paths=pretrain_itr * params['num_path_onpol'],
            horizon=params['env_horizon']
        )

        normalization = add_path_data_to_collection_and_update_normalization(
            rl_paths, path_collection,
            train_collection, val_collection, normalization)

    elif pretrain_mode == "metrpo":
        # simply start a few iterations early
        start_itr -= pretrain_itr

    if train_dyn_with_intrinsic_reward_only:
        dyn_model.use_intrinsic_rewards_only()
    else:
        dyn_model.use_external_rewards_only()

    # Main training loop
    for itr in range(start_itr, end_itr):
        logger.info('itr #%d | ' % itr)

        if confirm_restoring_dynamics_model(params):
            restore_model(params, saver, sess, itr)
        else:
            # Fit the dynamics.
            logger.info("Fitting dynamics.")

            dyn_model.fit(train_collection, val_collection)

            logger.info("Done fitting dynamics.")

            rollout_sampler.update_dynamics(dyn_model)

        # Update randomness
        logger.info("Updating randomness.")
        dyn_model.update_randomness()
        logger.info("Done updating randomness.")

        # Policy training
        logger.info("Training policy using TRPO.")

        logger.info("Re-initialize init_std.")
        sess.run(reset_opt)

        if params['policy'].get('reinitialize_every_itr', False):
            logger.info("Re-initialize policy variables.")
            policy.initialize_variables()

        train_policy_trpo(params, algo, dyn_model, params['trpo']['iterations'])

        logger.info("Done training policy.")

        # Generate on-policy rollouts.
        logger.info("Generating on-policy rollouts.")
        rl_paths = rollout_sampler.sample(
            num_paths=params['num_path_onpol'],
            horizon=params['env_horizon']
        )
        logger.info("Done generating on-policy rollouts.")

        # Update data.
        normalization = add_path_data_to_collection_and_update_normalization(rl_paths, path_collection,
                                                                             train_collection, val_collection,
                                                                             normalization)
        if train_dyn_with_intrinsic_reward_only:
            # Evaluate with external reward once in a while
            if (itr + 1) % external_evaluation_interval == 0:
                dyn_model.use_external_rewards_only()
                logger.info("Training policy with external reward to evaluate the dynamics model.")
                policy_ext.initialize_variables()
                train_policy_trpo(params, algo_ext, dyn_model, params['trpo_ext_reward']['iterations'])
                logger.info("Done training policy with external reward.")
                logger.info("Generating on-policy rollouts with external reward.")
                rl_paths_ext = rollout_sampler_ext.sample(
                    num_paths=params['num_path_onpol'],
                    horizon=params['env_horizon']
                )
                logger.info("Done generating on-policy rollouts with external reward.")
                # Compute metrics and log results
                returns = np.array([sum(path["rewards"]) for path in rl_paths_ext])
                log_tabular_results(returns, itr, train_collection)
                dyn_model.use_intrinsic_rewards_only()
        else:
            # Compute metrics and log results
            returns = np.array([sum(path["rewards"]) for path in rl_paths])
            log_tabular_results(returns, itr, train_collection)

        # save dynamics model if applicable
        save_cur_iter_dynamics_model(params, saver, sess, itr)


def get_exp_name(exp_name, seed):
    return "experiments/" + exp_name + '_seed' + str(seed)


def set_seed(seed):
    seed %= 4294967294
    global seed_
    seed_ = seed
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.set_random_seed(seed)
    except Exception as e:
        print(e)


def run_train(params, exp_name):
    for seed in params["random_seeds"]:
        # set seed
        print("Using random seed {}".format(seed))
        set_seed(seed)

        # logger
        exp_dir = get_exp_name(exp_name, seed)
        logger.configure(exp_dir)
        logger.info("Print configuration .....")
        logger.info(params)
        train(params)

    return


if __name__ == '__main__':
    options, exp_name = parse_args()
    params = process_params(options, options.param_path)
    run_train(params, exp_name)
