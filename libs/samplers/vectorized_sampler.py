import pickle
import numpy as np
import itertools

import logger
from libs.misc import tensor_utils, misc_utils
from envs.vec_env import VecEnvExecutor


class VectorizedSampler(object):

    def __init__(self, algo, n_envs=None):
        self.algo = algo
        self.n_envs = n_envs

    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(n_envs)]
            self.vec_env = VecEnvExecutor(
                envs=envs,
                max_path_length=self.algo.max_path_length
            )

    def shutdown_worker(self):
        self.vec_env.terminate()

    def obtain_samples(self, itr, dynamics=None):
        logger.info("Obtaining samples for iteration %d..." % itr)
        paths = []
        n_samples = 0
        obses = self.vec_env.reset()
        # dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy
        import time
        while n_samples < self.algo.batch_size:
            t = time.time()
            actions, agent_infos = policy.get_actions(obses)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            if dynamics:
                rewards = dynamics.process_rewards(rewards, obses, actions, next_obses)
            env_time += time.time() - t

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            for idx, obs, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                            rewards, env_infos,
                                                                            agent_infos, dones):

                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(obs)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    paths.append(dict(
                        observations=self.algo.env.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.algo.env.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            obses = next_obses

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)

        return paths, n_samples

    def update_stats(self, paths):
        obs = [path["observations"] for path in paths]
        obs = np.concatenate(obs)
        self.algo.running_stats.update_stats(np.reshape(obs, [-1, self.algo.obs_dim]))

    def process_samples(self, itr, paths):
        baselines = []
        returns = []

        all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                self.algo.discount * path_baselines[1:] - \
                path_baselines[:-1]
            path["advantages"] = misc_utils.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = misc_utils.discount_cumsum(path["rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

            # a trick to reduce variance but gives biased gradient
            path["value_targets"] = path["advantages"] + np.array(path_baselines[:-1])

        ev = misc_utils.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

        max_path_length = max([len(path["advantages"]) for path in paths])

        # make all paths the same length (pad extra advantages with 0)
        obs = [path["observations"] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        if self.algo.center_adv:
            raw_adv = np.concatenate([path["advantages"] for path in paths])
            adv_mean = np.mean(raw_adv)
            adv_std = np.std(raw_adv) + 1e-8
            adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
        else:
            adv = [path["advantages"] for path in paths]

        adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

        actions = [path["actions"] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        rewards = [path["rewards"] for path in paths]
        rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

        returns = [path["returns"] for path in paths]
        returns = tensor_utils.pad_tensor_n(returns, max_path_length)

        agent_infos = [path["agent_infos"] for path in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list(
            [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
        )

        env_infos = [path["env_infos"] for path in paths]
        env_infos = tensor_utils.stack_tensor_dict_list(
            [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
        )

        valids = [np.ones_like(path["returns"]) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        samples_data = dict(
            observations=obs,
            actions=actions,
            advantages=adv,
            rewards=rewards,
            returns=returns,
            valids=valids,
            agent_infos=agent_infos,
            env_infos=env_infos,
            paths=paths,
        )

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data
