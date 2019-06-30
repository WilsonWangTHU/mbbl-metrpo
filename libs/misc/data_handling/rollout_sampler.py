from libs.misc.utils import get_inner_env
from libs.misc.visualization import turn_off_video_recording, turn_on_video_recording
from model.controllers import RandomController
from libs.misc.data_handling.path import Path
import logger


class RolloutSampler:
    def __init__(self, env, dynamics=None, controller=None):
        self.env = env
        self.inner_env = get_inner_env(self.env)
        self.dynamics = dynamics
        self.controller = controller
        self.random_controller = RandomController(self.env)

    def update_dynamics(self, new_dynamics):
        self.dynamics = new_dynamics

    def update_controller(self, new_controller):
        self.controller = new_controller

    def generate_random_rollouts(self, num_paths, horizon=1000):
        logger.info("Generating random rollouts.")
        random_paths = self.sample(
            num_paths=num_paths,
            horizon=horizon,
            use_random_controller=True
        )
        logger.info("Done generating random rollouts.")
        return random_paths

    def sample(self,
               num_paths=3,
               horizon=1000,
               visualize=False,
               visualize_path_no=None,
               use_random_controller=False):
        # Write a sampler function which takes in an environment, a controller
        # (either random or the MPC controller), and returns rollouts by running on
        # the env. Each path can have elements for observations, next_observations,
        # rewards, returns, actions, etc.
        paths = []
        total_timesteps = 0
        path_num = 0

        controller = self._get_controller(use_random_controller=use_random_controller)

        while True:
            turn_off_video_recording()
            if visualize and not isinstance(controller, RandomController):
                if (visualize_path_no is None) or (path_num == visualize_path_no):
                    turn_on_video_recording()

            self._reset_env_for_visualization()

            logger.info("Path {} | total_timesteps {}.".format(path_num, total_timesteps))

            # update randomness
            if controller.__class__.__name__ == "MPCcontroller" \
                and hasattr(controller.dyn_model, 'update_randomness'):
                controller.dyn_model.update_randomness()

            path, total_timesteps = self._rollout_single_path(horizon, controller,
                                                             total_timesteps)
            paths.append(path)
            path_num += 1
            if total_timesteps >= num_paths * horizon:
                break

        turn_off_video_recording()

        return paths

    # ---- Private methods ----

    def _reset_env_for_visualization(self):
        # A hack for resetting env while recording videos
        if hasattr(self.env.wrapped_env, "stats_recorder"):
            setattr(self.env.wrapped_env.stats_recorder, "done", None)

    def _get_controller(self, use_random_controller=False):
        if use_random_controller:
            return self.random_controller
        return self.controller

    def _rollout_single_path(self, horizon, controller, total_timesteps):
        path = Path()
        obs = self.env.reset()
        for horizon_num in range(1, horizon + 1):
            action = controller.get_action(obs)
            next_obs, reward, done, _info = self.env.step(action)

            path.add_timestep(obs, action, next_obs, reward)
            obs = next_obs
            if done or horizon_num == horizon:
                total_timesteps += horizon_num
                break
        return path, total_timesteps
