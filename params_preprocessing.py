import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='run experiment options')
    parser.add_argument('--env')
    parser.add_argument('--exp_name')
    parser.add_argument('--sub_exp_name', default="")
    parser.add_argument('--algo', default='trpo')
    parser.add_argument('--param_path', default=None)
    parser_options = parser.parse_args()
    exp_name = '%s/%s/%s' % (parser_options.env, parser_options.sub_exp_name, parser_options.exp_name)
    return parser_options, exp_name


def process_params(options, param_path=None):
    if options.env in ['half_cheetah', 'swimmer', 'ant', 'walker2d',
                       'reacher', 'hopper', 'invertedPendulum', 'acrobot',
                       'cartpole', 'mountain', 'pendulum',
                       'gym_petsCheetah', 'gym_petsReacher', 'gym_petsPusher',
                       'gym_cheetahO01', 'gym_cheetahO001',
                       'gym_cheetahA01', 'gym_cheetahA003',
                       'gym_pendulumO01', 'gym_pendulumO001',
                       'gym_cartpoleO01', 'gym_cartpoleO001',
                       'gym_fant', 'gym_fswimmer', 'gym_fhopper', 'gym_fwalker2d',
                       'gym_humanoid', 'gym_slimhumanoid', 'gym_nostopslimhumanoid']:
        if param_path is None:
            param_path = os.path.join(os.path.curdir, 'configs/params_%s.json' % options.env)
    else:
        raise NotImplementedError

    if options.algo not in ['trpo', 'vime']:
        raise NotImplementedError

    with open(param_path, 'r') as f:
        params = json.load(f)
    params['algo'] = options.algo
    assert params['env_name'] == options.env
    # assert options.algo in options.exp_name
    return params
