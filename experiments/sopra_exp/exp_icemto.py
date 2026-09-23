# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

from gym.wrappers import RescaleAction, TimeLimit
from softae.agents.model_based.model_based_agent import ModelBasedAgent
from softae.trainer.model_based.model_based_trainer import ModelBasedTrainer as Trainer
from softae.models.active_learning_model import ActiveLearningHUCRLModel, ActiveLearningPETSModel
import numpy as np
import time
import json
import os
import sys
import argparse
from experiments.util import Logger, hash_dict, NumpyArrayEncoder
import wandb
from typing import Optional
from softae.models.hucrl_model import HUCRLModel
from softae.models.environment_models.sopra_arm_reward import SoftArmRewardModel
from softae.envs.wrappers.action_repeat import ActionRepeat


def experiment(
        logs_dir: str, use_wandb: bool, exp_name: str, time_limit: int, n_envs: int,
        num_samples: int, num_elites: int, num_steps: int, horizon: int, n_particles: int, reset_model: bool,
        num_ensembles: int, hidden_layers: int, num_neurons: int, beta: float, deterministic: bool,
        max_train_steps: int, pred_diff: bool, batch_size: int, eval_freq: int, total_train_steps: int,
        buffer_size: int, exploration_steps: int, eval_episodes: int, train_freq: int,
        num_epochs: int, rollout_steps: int, normalize: bool, action_normalize: bool, validate: bool,
        record_test_video: bool, validation_buffer_size: int, validation_batch_size: int,
        seed: int, exploration_strategy: str,
        time_limit_eval: int, action_cost: float = 0.0, action_repeat: int = 1, lr: float = 1e-3,
        colored_noise_exponent: float = 0.25, calibrate_model: bool = True,
        ctrl_cost: float = 0.0, vel_cost: float = 0.0,
        optimizer_type: str = 'iCemTO',
        resume_path: Optional[str] = None,
    ):
    optimizer_config = dict(
        optimizer_type=optimizer_type,
        num_samples=num_samples,
        num_elites=num_elites,
        num_steps=num_steps,
        n_particles=n_particles,
        horizon=horizon,
        exponent=colored_noise_exponent,
    )
    model_config = dict(
        reset_model=reset_model,
        num_ensembles=num_ensembles,
        hidden_layers=hidden_layers,
        num_neurons=num_neurons,
        beta=beta,
        deterministic=deterministic,
        max_train_steps=max_train_steps,
        pred_diff=pred_diff,
        batch_size=batch_size,
        eval_freq=eval_freq,
        num_epochs=num_epochs,
        lr=lr,
        calibrate_model=calibrate_model,
    )

    trainer_config = dict(
        total_train_steps=total_train_steps,
        buffer_size=buffer_size,
        exploration_steps=exploration_steps,
        eval_episodes=eval_episodes,
        train_freq=train_freq,
        rollout_steps=rollout_steps,
        normalize=normalize,
        action_normalize=action_normalize,
        validate=validate,
        record_test_video=record_test_video,
        validation_buffer_size=validation_buffer_size,
        validation_batch_size=validation_batch_size,
        seed=seed,
    )
    env_config = dict(
        time_limit=time_limit,
        time_limit_eval=time_limit_eval,
        action_cost=action_cost,
        ctrl_cost=ctrl_cost,
        vel_cost=vel_cost,
    )

    # define environment for train and test
    from softae.utils.vec_env.env_util import make_vec_env
    import math
    time_lim = math.ceil(time_limit / action_repeat)
    wrapper_cls = lambda x: RescaleAction(
        TimeLimit(
            ActionRepeat(x, repeat=action_repeat),
            max_episode_steps=time_lim),
        min_action=-1,
        max_action=1,
    )
    time_lim_eval = math.ceil(time_limit_eval / action_repeat)
    wrapper_cls_test = lambda x: RescaleAction(
        TimeLimit(
            ActionRepeat(x, repeat=action_repeat),
            max_episode_steps=time_lim_eval),
        min_action=-1,
        max_action=1,
    )

    from softae.envs.utils.hardware_interfaces.sopra_hardware_interface import SoPrAHWInterface
    hw_intf = SoPrAHWInterface()

    from softae.envs.sopra_env import SopraTrackingEnv as SoftArmEnv
    freq_hz = 1
    target_kwargs_train = {
        'target_position': np.array([2.69183693e-01, 6.12480003e-02, 4.78399431e-02]),
    }
    target_kwargs_backwawrd = {
        'target_position': np.array([2.77541941e-01, -5.36855794e-02, -4.34468240e-02]),
    }

    reward_model_reach = SoftArmRewardModel(target_position=target_kwargs_train['target_position'], 
                                            ctrl_cost=ctrl_cost,
                                            vel_cost=vel_cost)
    reward_model_unseen = SoftArmRewardModel(target_position=target_kwargs_backwawrd['target_position'], 
                                             ctrl_cost=ctrl_cost,
                                             vel_cost=vel_cost)
    env_kwargs_train = {
        'hw_intf': hw_intf,
        'freq_hz': freq_hz,
        'reward_model': reward_model_reach,
        'render_mode': None,
        'save_dir': logs_dir + 'save/train',
    } | target_kwargs_train

    env_kwargs_forward = {
        'hw_intf': hw_intf,
        'freq_hz': freq_hz,
        'reward_model': reward_model_reach,
        'render_mode': 'rgb_array',
        'COLLECT_DATA_FOR_POSTPROCESSING': True,
        'save_dir': logs_dir + 'save/task0',
        'max_env_step': time_lim_eval,
    } | target_kwargs_train

    env_kwargs_backward = {
        'hw_intf': hw_intf,
        'freq_hz': freq_hz,
        'reward_model': reward_model_unseen,
        'render_mode': 'rgb_array',
        'COLLECT_DATA_FOR_POSTPROCESSING': True,
        'save_dir': logs_dir + 'save/task1',
        'max_env_step': time_lim_eval,
    } | target_kwargs_backwawrd

    env = make_vec_env(env_id=SoftArmEnv, wrapper_class=wrapper_cls, n_envs=n_envs, seed=seed,
                       env_kwargs=env_kwargs_train)
    test_env_forward = make_vec_env(SoftArmEnv, wrapper_class=wrapper_cls_test, seed=seed,
                                    env_kwargs=env_kwargs_forward, n_envs=1)
    test_env_backward = make_vec_env(SoftArmEnv, wrapper_class=wrapper_cls_test, seed=seed,
                                     env_kwargs=env_kwargs_backward, n_envs=1) 

    features = tuple([num_neurons] * hidden_layers)
    uniform_exploration = False
    video_prefix = ""
    if exploration_strategy == 'Mean':
        video_prefix += 'Mean'
        dynamics_model_forward = ActiveLearningHUCRLModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_reach,
            features=features,
            pred_diff=pred_diff,
            beta=0.0,
            seed=seed,
            use_log_uncertainties=True,
            use_al_uncertainties=True,
            deterministic=deterministic,
            lr=lr,
            action_cost=action_cost,
        )

        dynamics_model_backward = ActiveLearningHUCRLModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_unseen,
            features=features,
            pred_diff=pred_diff,
            beta=0.0,
            seed=seed,
            use_log_uncertainties=True,
            use_al_uncertainties=True,
            deterministic=deterministic,
            lr=lr,
            action_cost=action_cost,
        )

    elif exploration_strategy == 'PETS':
        dynamics_model_forward = ActiveLearningPETSModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_reach,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            use_log_uncertainties=True,
            use_al_uncertainties=True,
            deterministic=deterministic,
            lr=lr,
            action_cost=action_cost,
        )
        dynamics_model_backward = ActiveLearningPETSModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_unseen,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            use_log_uncertainties=True,
            use_al_uncertainties=True,
            deterministic=deterministic,
            lr=lr,
            action_cost=action_cost,
        )

        video_prefix += 'PETS'
    elif exploration_strategy == 'SoftAE':
        dynamics_model_forward = ActiveLearningHUCRLModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_reach,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            use_log_uncertainties=True,
            use_al_uncertainties=True,
            deterministic=deterministic,
            lr=lr,
            action_cost=action_cost,
        )

        dynamics_model_backward = ActiveLearningHUCRLModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_unseen,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            use_log_uncertainties=True,
            use_al_uncertainties=True,
            deterministic=deterministic,
            lr=lr,
            action_cost=action_cost,
        )

        video_prefix += 'SoftAE'
    elif exploration_strategy == 'HUCRL':
        dynamics_model_forward = HUCRLModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_reach,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            deterministic=deterministic,
            lr=lr,
        )

        dynamics_model_backward = HUCRLModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_unseen,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            deterministic=deterministic,
            lr=lr,
        )
        video_prefix += 'HUCRL'

    else:
        dynamics_model_forward = ActiveLearningHUCRLModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_reach,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            use_log_uncertainties=True,
            use_al_uncertainties=True,
            deterministic=deterministic,
            lr=lr,
            action_cost=action_cost,
        )

        dynamics_model_backward = ActiveLearningHUCRLModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_model_unseen,
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            use_log_uncertainties=True,
            use_al_uncertainties=True,
            deterministic=deterministic,
            lr=lr,
            action_cost=action_cost,
        )
        video_prefix += 'Random'
        uniform_exploration = True

    dynamics_model = [dynamics_model_forward, dynamics_model_backward]
    test_env = [test_env_forward, test_env_backward]
    optimizer_kwargs = optimizer_config.copy()
    optimizer_kwargs.pop('optimizer_type')
    optimizer_kwargs.pop('horizon')
    optimizer_kwargs.pop('n_particles')
    agent = ModelBasedAgent(
        batch_size=batch_size,
        max_train_steps=max_train_steps,
        num_epochs=num_epochs,
        action_space=env.action_space,
        observation_space=env.observation_space,
        dynamics_model=dynamics_model,
        n_particles=n_particles,
        reset_model=reset_model,
        policy_optimizer_name=optimizer_type,
        optimizer_kwargs=optimizer_kwargs,
        horizon=horizon,
        calibrate_model=calibrate_model,
    )

    trainer = Trainer(
        agent=agent,
        env=env,
        test_env=test_env,
        buffer_size=buffer_size,
        total_train_steps=total_train_steps,
        exploration_steps=exploration_steps,
        use_wandb=use_wandb,
        eval_episodes=eval_episodes,
        eval_freq=eval_freq,
        train_freq=train_freq,
        rollout_steps=rollout_steps,
        normalize=normalize,
        action_normalize=action_normalize,
        learn_deltas=pred_diff,
        validate=validate,
        record_test_video=record_test_video,
        validation_buffer_size=validation_buffer_size,
        validation_batch_size=validation_batch_size,
        seed=seed,
        uniform_exploration=uniform_exploration,
        video_prefix=video_prefix,
        video_folder=logs_dir,
    )
    if resume_path is not None:
        print("Loading training checkpoint from", resume_path)
        trainer.load_agent(resume_path)
        if total_train_steps > trainer.total_train_steps:
            trainer.set_total_trainer_steps(total_train_steps)

    if use_wandb:
        wandb.init(
            dir=logs_dir,
            project=exp_name,
            group=exploration_strategy,
            config=optimizer_config | model_config | trainer_config | env_config,
        )

    trainer.train()

    result_dict = {
    }

    hw_intf.stop()

    wandb.finish()
    return result_dict


def main(args):
    """"""
    from pprint import pprint
    print(args)
    """ generate experiment hash and set up redirect of output streams """
    exp_hash = hash_dict(args.__dict__)
    if args.exp_result_folder is not None:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(args.exp_result_folder, '%s.log ' % exp_hash)
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    pprint(args.__dict__)
    print('\n ------------------------------------ \n')

    """ Experiment core """
    t_start = time.time()
    np.random.seed(args.seed + 5)

    eval_metrics = experiment(
        logs_dir=args.logs_dir,
        use_wandb=bool(args.use_wandb),
        exp_name=args.exp_name,
        time_limit=args.time_limit,
        n_envs=args.n_envs,
        optimizer_type=args.optimizer_type,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_steps=args.num_steps,
        horizon=args.horizon,
        n_particles=args.n_particles,
        reset_model=bool(args.reset_model),
        num_ensembles=args.num_ensembles,
        hidden_layers=args.hidden_layers,
        num_neurons=args.num_neurons,
        beta=args.beta,
        deterministic=bool(args.deterministic),
        max_train_steps=args.max_train_steps,
        pred_diff=bool(args.pred_diff),
        batch_size=args.batch_size,
        eval_freq=args.eval_freq,
        total_train_steps=args.total_train_steps,
        buffer_size=args.buffer_size,
        exploration_steps=args.exploration_steps,
        eval_episodes=args.eval_episodes,
        train_freq=args.train_freq,
        num_epochs=args.num_epochs,
        rollout_steps=args.rollout_steps,
        normalize=bool(args.normalize),
        action_normalize=bool(args.action_normalize),
        validate=bool(args.validate),
        record_test_video=bool(args.record_test_video),
        validation_buffer_size=args.validation_buffer_size,
        validation_batch_size=args.validation_batch_size,
        seed=args.seed,
        exploration_strategy=args.exploration_strategy,
        time_limit_eval=args.time_limit_eval,
        action_cost=args.action_cost,
        action_repeat=args.action_repeat,
        lr=args.lr,
        colored_noise_exponent=args.colored_noise_exponent,
        calibrate_model=bool(args.calibrate_model),
        ctrl_cost=args.ctrl_cost,
        vel_cost=args.vel_cost,
        resume_path=args.resume
    )

    t_end = time.time()

    """ Save experiment results and configuration """
    results_dict = {
        'evals': eval_metrics,
        'params': args.__dict__,
        'duration_total': t_end - t_start
    }

    if args.exp_result_folder is None:
        from pprint import pprint
        pprint(results_dict)
    else:
        exp_result_file = os.path.join(args.exp_result_folder, '%s.json' % exp_hash)
        with open(exp_result_file, 'w') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)
        print('Dumped results to %s' % exp_result_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Active-Exploration-run')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='./')
    parser.add_argument('--use_wandb', type=int, default=0)
    parser.add_argument('--exp_name', type=str, required=True, default='sopra_icem_test')
    # env experiment args
    parser.add_argument('--time_limit', type=int, default=1000)
    parser.add_argument('--n_envs', type=int, default=1)

    # optimizer experiment args
    parser.add_argument('--optimizer_type', type=str, default='iCemTO')
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--num_elites', type=int, default=50)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=20)
    parser.add_argument('--n_particles', type=int, default=10)

    # Model args
    parser.add_argument('--reset_model', type=int, default=1)
    parser.add_argument('--num_ensembles', type=int, default=5)
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--num_neurons', type=int, default=128)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--deterministic', type=int, default=1)
    parser.add_argument('--max_train_steps', type=int, default=5000)
    parser.add_argument('--pred_diff', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=-1)

    # trainer experiment args
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--total_train_steps', type=int, default=2500)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--exploration_steps', type=int, default=0)
    parser.add_argument('--eval_episodes', type=int, default=1)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--rollout_steps', type=int, default=10)
    parser.add_argument('--normalize', type=int, default=1)
    parser.add_argument('--action_normalize', type=int, default=1)
    parser.add_argument('--validate', type=int, default=1)
    parser.add_argument('--record_test_video', type=int, default=0)
    parser.add_argument('--validation_buffer_size', type=int, default=100000)
    parser.add_argument('--validation_batch_size', type=int, default=4096)
    parser.add_argument('--exploration_strategy', type=str, default='Optimistic')
    parser.add_argument('--action_cost', type=float, default=0.0)
    parser.add_argument('--time_limit_eval', type=int, default=10)
    parser.add_argument('--action_repeat', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--colored_noise_exponent', type=float, default=0.25)
    parser.add_argument('--calibrate_model', type=int, default=1)
    parser.add_argument('--ctrl_cost', type=float, default=0.0)
    parser.add_argument('--vel_cost', type=float, default=0.0)

    # general args
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')

    parser.add_argument("--resume", type=str, default=None, help="Path to stored checkpoint from which to resume training")

    args = parser.parse_args()
    main(args)
