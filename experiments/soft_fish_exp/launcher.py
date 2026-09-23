# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

from experiments.util import generate_base_command, generate_run_commands, hash_dict, sample_flag, RESULT_DIR

import yaml
import argparse
import numpy as np
import copy
import os
import itertools

from datetime import datetime

applicable_configs = {
    'general': ['use_wandb'],
    'env': ['time_limit', 'n_envs', 'time_limit_eval'],
    'optimizer': ['optimizer_type', 'num_samples', 'num_elites', 'num_steps', 'horizon', 'alpha'],
    'agent': ['discount', 'n_particles', 'reset_model', 'batch_size', 'train_steps', 'num_epochs', 'max_train_steps'],
    'dynamics_model': ['num_ensembles', 'hidden_layers', 'num_neurons', 'pred_diff', 'deterministic'],
    'trainer': ['eval_freq', 'total_train_steps', 'buffer_size',
                'exploration_steps', 'eval_episodes', 'train_freq', 'rollout_steps',
                'validate', 'normalize', 'action_normalize', 'record_test_video', 'validation_buffer_size',
                'validation_batch_size'],
}

search_ranges = {
}

ctrl_costs = [0]
exploration_stepss = [0]


def main(args):
    env_name = args.env_name
    file_path = os.path.dirname(os.path.abspath(__file__))
    assert env_name in ['Koi'], \
        "Only SoftFish environment works"
    EXPLORATION_STRATEGY = ['SoftAE']
    if args.task_reach:
        import active_exploration_exp_softfish_reach as active_exploration_exp
    else:
        import active_exploration_exp_softfish_forward as active_exploration_exp
    default_configs = yaml.safe_load(open(file_path + '/hyperparams.yaml', 'r'))

    default_configs['optimizer_type'] = args.optimizer_type
    default_configs['hidden_layers'] = args.hidden_layers
    default_configs['horizon'] = args.horizon
    default_configs['time_limit'] = args.time_limit
    default_configs['time_limit_eval'] = args.time_limit
    default_configs['rollout_steps'] = args.time_limit
    
    # check consistency of configuration dicts
    assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys(), *search_ranges.keys()}
    rds = np.random.RandomState(args.seed)
    assert args.num_seeds_per_hparam < 100
    init_seeds = list(rds.randint(0, 10 ** 6, size=(100,)))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    # exp_path = os.path.join(exp_base_path, f'{args.exp_name}')
    exp_path = exp_base_path

    command_list = []
    output_list = []
    for _ in range(args.num_hparam_samples):
        # transfer flags from the args
        flags = copy.deepcopy(args.__dict__)
        if flags['launch_mode'] == 'euler':
            logs_dir = '/cluster/scratch/'
            logs_dir += flags['user_name'] + '/' + args.exp_name + '/'
        else:
            logs_dir = './results/' + args.exp_name + '/'
        [flags.pop(key) for key in ['seed', 'num_hparam_samples', 'num_seeds_per_hparam', 'num_cpus',
                                    'num_gpus', 'launch_mode', 'user_name', 'task_reach', 'long_run', 'num_hours']]

        # randomly sample flags
        for flag in default_configs:
            if flag in search_ranges:
                flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
            else:
                flags[flag] = default_configs[flag]
        flags['logs_dir'] = logs_dir
        for exploration_steps in exploration_stepss:
            flags['exploration_steps'] = exploration_steps
            flags['ctrl_cost'] = 0
                    
            for exploration_strategy in EXPLORATION_STRATEGY:
            # hyperparameter sweep
                # determine subdir which holds the repetitions of the exp
                flags_hash = hash_dict(flags)
                flags['exploration_strategy'] = exploration_strategy
                flags['exp_result_folder'] = os.path.join(exp_path, flags_hash)

                for j in range(args.num_seeds_per_hparam):
                    seed = init_seeds[j]
                    cmd = generate_base_command(active_exploration_exp, flags=dict(**flags, **{'seed': seed}))
                    command_list.append(cmd)
                    now = datetime.now()
                    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
                    output_path = logs_dir + 'log' + str(seed) + exploration_strategy + dt_string + '.log'
                    output_list.append(output_path)

    # submit jobs
    generate_run_commands(command_list, output_list, num_cpus=args.num_cpus, num_gpus=args.num_gpus, mode=args.launch_mode,
                          long=args.long_run,
                          num_hours=args.num_hours,
                          promt=True,
                          mem=16000,
                          )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')
    parser.add_argument('--exp_name', type=str, default='SoftFish-test-3DRender')
    parser.add_argument('--task_reach', default=False, action="store_true")
    parser.add_argument('--env_name', type=str, default='Koi', help='Name of the environment')
    parser.add_argument('--num_cpus', type=int, default=8, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--launch_mode', type=str, default='euler', help='how to launch the experiments')
    parser.add_argument('--user_name', type=str, default='zhengh', help='name of user launching experiments')
    parser.add_argument('--num_hparam_samples', type=int, default=1)
    parser.add_argument('--num_seeds_per_hparam', type=int, default=10)
    parser.add_argument('--use_log', default=False, action="store_true")
    parser.add_argument('--use_al', default=False, action="store_true")
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--long_run', default=False, action="store_true")
    parser.add_argument('--num_hours', type=int, default=None)

    parser.add_argument('--optimizer_type', type=str, default='iCemTO')
    parser.add_argument('--hidden_layers', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--time_limit', type=int, default=200)

    parser.add_argument('--randomise_target', default=False, action="store_true")
    parser.add_argument('--local_frame', default=False, action="store_true")
    parser.add_argument('--use_angle_obs', default=False, action="store_true")
    parser.add_argument('--task0_vel_weight', type=float, default=1.0, help='base task velocity weight')
    parser.add_argument('--target_radius', type=float, default=2.0, help='init target distance when random sampling')

    args = parser.parse_args()
    main(args)
