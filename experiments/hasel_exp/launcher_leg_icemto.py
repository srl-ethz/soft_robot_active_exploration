# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

import numpy as np
import argparse
from datetime import datetime
from experiments.util import generate_run_commands, generate_base_command, dict_permutations, hash_dict

PROJECT_NAME = 'HASEL-Leg-iCem'
OPTIMIZER_TYPE = 'iCemTO'

seed = 1
num_seeds_per_hparam = 10
rds = np.random.RandomState(seed)
init_seeds = list(rds.randint(0, 10 ** 6, size=(num_seeds_per_hparam,)))

_applicable_configs_base = {
    'use_wandb': [1],
    'exp_name': [PROJECT_NAME],
    'seed': init_seeds,
    'time_limit': [500],
    'n_envs': [4],
    'n_particles': [10],
    'horizon': [100],
    'reset_model': [0],
    'num_ensembles': [5],
    'hidden_layers': [4],
    'num_neurons': [256],
    'deterministic': [0],
    'max_train_steps': [5_000],
    'pred_diff': [1],
    'batch_size': [64],
    'num_epochs': [50],
    'eval_freq': [5],
    'total_train_steps': [800_000],
    'buffer_size': [1_000_000],
    'exploration_steps': [0],
    'eval_episodes': [1],
    'train_freq': [1],
    'rollout_steps': [200],
    'normalize': [1],
    'action_normalize': [1],
    'validate': [1],
    'record_test_video': [0],
    'validation_buffer_size': [0],
    'validation_batch_size': [4_096],
    'action_cost': [0.0],
    'time_limit_eval': [500],
    'action_repeat': [1],
    'lr': [5e-5],
    'calibrate_model': [1],
    'ctrl_cost': [0.0],
    'vel_cost': [0.0],
}

_applicable_configs_icem = {
    'optimizer_type': ['iCemTO'],
    'num_samples': [200],
    'num_elites': [20],
    'num_steps': [5],
    'colored_noise_exponent': [0.25],
}

if OPTIMIZER_TYPE == 'iCemTO':
    from experiments.hasel_exp import exp_leg_icemto as exp
    _applicable_configs = _applicable_configs_base | _applicable_configs_icem
else:
    raise NotImplementedError('Selected optimizer not implemented.')

_applicable_configs_softae = {'exploration_strategy': ['SoftAE'], 'beta': [2.0]} | _applicable_configs
_applicable_configs_pets = {'exploration_strategy': ['PETS'], 'beta': [0.0]} | _applicable_configs
_applicable_configs_mean = {'exploration_strategy': ['Mean'], 'beta': [0.0]} | _applicable_configs
_applicable_configs_hucrl = {'exploration_strategy': ['HUCRL'], 'beta': [2.0]} | _applicable_configs
_applicable_configs_random = {'exploration_strategy': ['Random'], 'beta': [2.0]} | _applicable_configs


all_flags_combinations = dict_permutations(_applicable_configs_softae) + dict_permutations(_applicable_configs_pets) \
                         + dict_permutations(_applicable_configs_mean) + dict_permutations(_applicable_configs_hucrl) \
                         + dict_permutations(_applicable_configs_random)

def main(args):
    command_list = []
    output_file_list = []
    logs_dir = './' + PROJECT_NAME + '/'

    for flags in all_flags_combinations:
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        exp_logs_dir = logs_dir + str(flags['seed']) + flags['exploration_strategy'] + dt_string + '_' + hash_dict(flags) +'/'
        flags['logs_dir'] = exp_logs_dir
        if args.resume is not None:
            flags['resume'] = args.resume
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)
        output_file = exp_logs_dir + 'exp_log.log'
        output_file_list.append(output_file)

    # submit jobs
    num_hours = 23 if args.long_run else 3
    generate_run_commands(command_list, output_file_list=output_file_list, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
                          mode=args.mode, duration=f'{num_hours}:59:00', prompt=True, mem=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--mode', type=str, default='local', help='how to launch the experiments')
    parser.add_argument('--long_run', default=False, action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to stored checkpoint from which to resume training")

    args = parser.parse_args()
    main(args)