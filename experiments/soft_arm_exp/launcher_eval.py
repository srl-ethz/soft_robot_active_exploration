# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

import argparse
import os
from datetime import datetime
from experiments.util import generate_run_commands, generate_base_command, dict_permutations
import numpy as np

PROJECT_NAME = 'SoftAE-Eval-Reach'
# Default 'reach' runs simulation rollouts with the pretrained agent on the reach task
# and works out-of-the-box with the bundled checkpoints. Switch to 'model_eval' for
# dynamics-model MSE evaluation (requires a separately collected validation buffer; see
# exp_icemto_eval.py for the expected path).
TASK = 'reach'
OPTIMIZER_TYPE = 'iCemTO'

# Resolve path to bundled pretrained checkpoint. Swap the filename to evaluate a
# different method: softae_softarm.pkl, hucrl_softarm.pkl, or random_softarm.pkl.
_PRETRAINED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'pretrained')
)
PRETRAINED_CHECKPOINT = os.path.join(_PRETRAINED_DIR, 'softae_softarm.pkl')

_applicable_configs_base = {
    'use_wandb': [0],
    'exp_name': [PROJECT_NAME],
    'train_mode': [0],
    'task': [TASK],
    'seed': list(range(1)),
    'time_limit': [200],
    'n_envs': [4],
    'n_particles': [10],
    'horizon': [10],
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
    'total_train_steps': [80_000],
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
    'time_limit_eval': [200],
    'action_repeat': [1],
    'lr': [5e-5],
    'calibrate_model': [1],
    'ctrl_cost': [0.0],
    'vel_cost': [0.0],
    'obstacle_cost': [1.0],
    'use_delta_action': [0],
    'delta_action': [0.0],
    'use_action_obs': [0],
    'stack_obs_time': [1],
    'obs_state_points': [4],
    'max_rate_of_change_of_activation': [np.infty],
    'resume': [PRETRAINED_CHECKPOINT],
}

_applicable_configs_icem = {
    'optimizer_type': ['iCemTO'],
    'num_samples': [200],
    'num_elites': [20],
    'num_steps': [5],
    'colored_noise_exponent': [0.25],
}

_applicable_configs_sac = {
    'optimizer_type': ['SacOpt'],
    'sac_train_steps': [2000],
    'sac_lr_actor': [3e-4],
    'sac_lr_critic': [7e-4],
    'sac_lr_alpha': [3e-5],
    'sac_tau': [5e-4],
    'sac_batch_size': [512],
    'train_steps_per_model_update': [50],
    'transitions_per_update': [400],
    'reset_actor_params': [1],
}

if OPTIMIZER_TYPE == 'iCemTO':
    from experiments.soft_arm_exp import exp_icemto as exp
    _applicable_configs = _applicable_configs_base | _applicable_configs_icem
else:
    raise NotImplementedError('Selected optimizer not implemented.')

_applicable_configs_softae = {'exploration_strategy': ['SoftAE'], 'beta': [2.0]} | _applicable_configs
_applicable_configs_hucrl = {'exploration_strategy': ['HUCRL'], 'beta': [2.0]} | _applicable_configs
_applicable_configs_random = {'exploration_strategy': ['Random'], 'beta': [2.0]} | _applicable_configs

# Eval sweeps only the single exploration strategy that matches the pretrained
# checkpoint being loaded. The mapping is by filename prefix:
#   softae_softarm.pkl -> SoftAE config
#   hucrl_softarm.pkl  -> HUCRL  config
#   random_softarm.pkl -> Random config
# (Training, by contrast, sweeps all strategies — see launcher.py.)
_CHECKPOINT_TO_CONFIG = {
    'softae': _applicable_configs_softae,
    'hucrl':  _applicable_configs_hucrl,
    'random': _applicable_configs_random,
}
_ckpt_method = os.path.basename(PRETRAINED_CHECKPOINT).split('_')[0].lower()
if _ckpt_method not in _CHECKPOINT_TO_CONFIG:
    raise ValueError(
        f"Cannot infer exploration strategy from PRETRAINED_CHECKPOINT "
        f"'{PRETRAINED_CHECKPOINT}'. Expected filename to start with one of "
        f"{sorted(_CHECKPOINT_TO_CONFIG)} (e.g. 'softae_softarm.pkl')."
    )
all_flags_combinations = dict_permutations(_CHECKPOINT_TO_CONFIG[_ckpt_method])


def main(args):
    command_list = []
    output_file_list = []
    logs_dir = './results/' + PROJECT_NAME + '/'

    for flags in all_flags_combinations:
        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        exp_logs_dir = logs_dir + str(flags['seed']) + flags['exploration_strategy'] + dt_string + '/'
        flags['logs_dir'] = exp_logs_dir
        cmd = generate_base_command(exp, flags=flags)
        command_list.append(cmd)
        output_file = exp_logs_dir + 'exp_log.log'
        output_file_list.append(output_file)

    num_hours = 23 if args.long_run else 3
    # prompt=False so the eval runs non-interactively (works inside Docker / CI).
    generate_run_commands(command_list, output_file_list=output_file_list, num_cpus=args.num_cpus, num_gpus=args.num_gpus,
                          mode=args.mode, duration=f'{num_hours}:59:00', prompt=False, mem=16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cpus', type=int, default=1, help='number of cpus to use')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--mode', type=str, default='local', help='how to launch the experiments')
    parser.add_argument('--long_run', default=False, action="store_true")

    args = parser.parse_args()
    main(args)