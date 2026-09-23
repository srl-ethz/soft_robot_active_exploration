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
        ctrl_cost: float = 0.0, vel_cost: float = 0.0, obstacle_cost: float = 0.0,
        use_delta_action: bool = False, delta_action: int = 1,
        use_action_obs: bool = False, stack_obs_time: int = 1,
        max_rate_of_change_of_activation: float = np.infty,
        optimizer_type: str = 'iCemTO',
        train_mode: bool = True,
        task: Optional[str] = 'reach', damping: Optional[float] = 0.2, obs_state_points: Optional[int] = 4,
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
        obstacle_cost=obstacle_cost,
        use_delta_action=use_delta_action,
        delta_action=delta_action,
        use_action_obs=use_action_obs,
        stack_obs_time=stack_obs_time,
        max_rate_of_change_of_activation=max_rate_of_change_of_activation,
        damping=damping,
        obs_state_points=obs_state_points,
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

    from softae.envs.soft_arm_reach import SoftArmReachOrienEnv as SoftArmEnv
    from softae.models.environment_models.soft_arm import SoftArmRewardModel
    use_obstacle_flags = []
    if task == 'reach_orien':
        target_positions = np.array([
            [0.5, 0.5, 0.5],
            [-0.3, 0.7, -0.3],
        ])
        target_orientations = [
            None,
            np.array([0.5, 0.5, 0.5, 0.5])
        ]
    elif task == 'reach' or task == 'model_eval':
        target_positions = np.array([
            [0.5, 0.5, 0.5],
            [-0.4, 0.1, -0.4],
        ])
        target_orientations = [None for _ in range(len(target_positions))]
    elif task == 'reach_single_rod':
        target_positions = np.array([
            [0.5, 0.5, 0.5],
            [-0.8, 0.15, 0.0],
        ])
        target_orientations = [None for _ in range(len(target_positions))]
        use_obstacle_flags = [True, True]
        n_obstacles_list = [1, 1]
        obstacle_pos_list = [
            np.array([-0.4, 0.4, 0.0]),
            np.array([-0.4, 0.4, 0.0]),
        ]
        obstacle_bounds = [
            0.1,
            0.1
        ]
    elif task == 'reach_multi_rods':
        target_positions = np.array([
            [0.5, 0.5, 0.5],
            [-0.8, 0.15, 0.0],
        ])
        target_orientations = [None for _ in range(len(target_positions))]
        use_obstacle_flags = [True, True]
        n_obstacles_list = [8, 8]
        obstacle_bounds = [
            0.1,
            0.1
        ]
    elif task == 'reach_unseen_obstacle':
        target_positions = np.array([
            [0.5, 0.5, 0.5],
            [-0.8, 0.15, 0.0],
        ])
        target_orientations = [None for _ in range(len(target_positions))]
        use_obstacle_flags = [False, True]
        n_obstacles_list = [0, 1]
        obstacle_pos_list = [
            None,
            np.array([-0.4, 0.4, 0.0]),
        ]
        obstacle_bounds = [
            0.0,
            0.1
        ]
    elif task == 'reach_multi_unseen_obstacles':
        target_positions = np.array([
            [0.5, 0.5, 0],
            [-0.8, 0.15, 0.0],
        ])
        target_orientations = [None for _ in range(len(target_positions))]
        use_obstacle_flags = [False, True]
        n_obstacles_list = [0, 8]
        obstacle_bounds = [
            0.0,
            0.1
        ]
    else:
        raise NotImplementedError("Not a valid task!")

    if len(use_obstacle_flags) == 0:
        use_obstacle_flags = [False for _ in range(len(target_positions))]

    reward_models = []
    for i in range(len(target_positions)):
        if use_obstacle_flags[i]:
            reward_models.append(
                SoftArmRewardModel(target_position=target_positions[i], 
                                target_orientation=target_orientations[i],
                                obs_state_points=obs_state_points, 
                                ctrl_cost=ctrl_cost, 
                                vel_cost=vel_cost,
                                tol=0.05,
                                obstacle_cost_clip_dist=0.2,
                                obstacle_cost=obstacle_cost,
                                obstacle_bound=obstacle_bounds[i],
                                obstacle_pos=obstacle_pos_list[i],
                                )
            )
        else:
            reward_models.append(
                SoftArmRewardModel(target_position=target_positions[i], 
                                    target_orientation=target_orientations[i],
                                    obs_state_points=obs_state_points, 
                                    ctrl_cost=ctrl_cost, 
                                    vel_cost=vel_cost)
                )

    _env_kwargs_base = {
        'damping': damping,
        'obs_state_points': obs_state_points,
        'use_delta_action': use_delta_action,
        'delta_action': delta_action,
        'use_action_obs': use_action_obs,
        'stack_obs_time': stack_obs_time,
        'max_rate_of_change_of_activation': max_rate_of_change_of_activation,
        'game_mode': 1,
        'render_mode': None,
    }

    test_envs = []
    for i in range(2):
        # Generate kwargs for env
        env_kwargs = _env_kwargs_base | {
            'reward_model': reward_models[i],
            'target_position': target_positions[i],
            'target_orientation': target_orientations[i],
        }
        
        if use_obstacle_flags[i]:
            env_kwargs = env_kwargs | {
                'use_obstacle': True,
                'n_obstacles': n_obstacles_list[i],
                'obstacle_pos': obstacle_pos_list[i],
                'obstacle_bound': obstacle_bounds[i],
            }
        # Make vec train env
        if i == 0:
            env = make_vec_env(env_id=SoftArmEnv, wrapper_class=wrapper_cls, n_envs=n_envs, 
                               seed=seed, env_kwargs=env_kwargs)
        # Append vec test env to list
        env_kwargs = env_kwargs | {
            'COLLECT_DATA_FOR_POSTPROCESSING': True,
            'save_dir': os.path.join(logs_dir, f'save/task_{i+1}'),
        }
        test_envs.append(make_vec_env(SoftArmEnv, wrapper_class=wrapper_cls_test, seed=seed, 
                                      env_kwargs=env_kwargs, n_envs=1))

    features = tuple([num_neurons] * hidden_layers)
    uniform_exploration = False
    video_prefix = ""
    
    dynamics_models = []
    if exploration_strategy == 'Mean':
        video_prefix += 'Mean'
        for i in range(len(target_positions)):
            dynamics_models.append(ActiveLearningHUCRLModel(
                action_space=env.action_space,
                observation_space=env.observation_space,
                num_ensemble=num_ensembles,
                reward_model=reward_models[i],
                features=features,
                pred_diff=pred_diff,
                beta=0.0,
                seed=seed,
                use_log_uncertainties=True,
                use_al_uncertainties=True,
                deterministic=deterministic,
                lr=lr,
                action_cost=action_cost,
            ))

    elif exploration_strategy == 'PETS':
        for i in range(len(target_positions)):
            dynamics_models.append(ActiveLearningPETSModel(
                action_space=env.action_space,
                observation_space=env.observation_space,
                num_ensemble=num_ensembles,
                reward_model=reward_models[i],
                features=features,
                pred_diff=pred_diff,
                beta=beta,
                seed=seed,
                use_log_uncertainties=True,
                use_al_uncertainties=True,
                deterministic=deterministic,
                lr=lr,
                action_cost=action_cost,
            ))

        video_prefix += 'PETS'
    elif exploration_strategy == 'OPAX':
        for i in range(len(target_positions)):
            dynamics_models.append(ActiveLearningHUCRLModel(
                action_space=env.action_space,
                observation_space=env.observation_space,
                num_ensemble=num_ensembles,
                reward_model=reward_models[i],
                features=features,
                pred_diff=pred_diff,
                beta=beta,
                seed=seed,
                use_log_uncertainties=True,
                use_al_uncertainties=True,
                deterministic=deterministic,
                lr=lr,
                action_cost=action_cost,
            ))
        
        video_prefix += 'OPAX'
    elif exploration_strategy == 'HUCRL':
        for i in range(len(target_positions)):
            dynamics_models.append(HUCRLModel(
                action_space=env.action_space,
                observation_space=env.observation_space,
                num_ensemble=num_ensembles,
                reward_model=reward_models[i],
                features=features,
                pred_diff=pred_diff,
                beta=beta,
                seed=seed,
                deterministic=deterministic,
                lr=lr,
            ))
        video_prefix += 'HUCRL'

    else:
        for i in range(len(target_positions)):
            dynamics_models.append(ActiveLearningHUCRLModel(
            action_space=env.action_space,
            observation_space=env.observation_space,
            num_ensemble=num_ensembles,
            reward_model=reward_models[i],
            features=features,
            pred_diff=pred_diff,
            beta=beta,
            seed=seed,
            use_log_uncertainties=True,
            use_al_uncertainties=True,
            deterministic=deterministic,
            lr=lr,
            action_cost=action_cost,
        ))
        video_prefix += 'Uniform'
        uniform_exploration = True

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
        dynamics_model=dynamics_models[:2],
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
        test_env=test_envs,
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

        for i in range(len(trainer.agent.dynamics_model_list)):
            trainer.agent.dynamics_model_list[i].reward_model = reward_models[i]

        while len(trainer.agent.dynamics_model_list) < len(reward_models):
            i = len(trainer.agent.dynamics_model_list)
            trainer.agent.dynamics_model_list.append(ActiveLearningHUCRLModel(
                action_space=env.action_space,
                observation_space=env.observation_space,
                num_ensemble=num_ensembles,
                reward_model=reward_models[i],
                features=features,
                pred_diff=pred_diff,
                beta=beta,
                seed=seed,
                use_log_uncertainties=True,
                use_al_uncertainties=True,
                deterministic=deterministic,
                lr=lr,
                action_cost=action_cost,))
        trainer.agent.policy_optimizer._init_fn()

    if use_wandb:
        wandb.init(
            dir=logs_dir,
            project=exp_name,
            group=exploration_strategy,
            config=optimizer_config | model_config | trainer_config | env_config,
        )

    if train_mode:
        trainer.train()
    else:
        ##########################################################################
        # Eval model
        if task == 'model_eval':
            # Validate model
            import cloudpickle
            # Pre-collected validation buffer for dynamics-model MSE evaluation.
            # The default release does not bundle this buffer; collect one via the data-collection
            # launcher and drop the resulting .pkl here to reproduce the reported MSE numbers.
            validation_buffer_path = './validation_buffer.pkl'
            validation_buffer = cloudpickle.load(open(validation_buffer_path, 'rb'))
            
            print(f'Loaded validation buffer from {validation_buffer_path}')
            trainer.validation_buffer = validation_buffer
            trainer.validation_batch_size = trainer.validation_buffer.size
            model_log = trainer.validate_model(rng=trainer.rng, shuffle=False)
            print("Single-step prediction:")
            print(model_log)
            model_log_autoregressive = trainer.validate_model(rng=trainer.rng, shuffle=False, autoregressive=True, start_step=11)
            print("Autoregressive prediction:")
            print(model_log_autoregressive)

            # Save results
            np.save(os.path.join(logs_dir, 'y_true.npy'), trainer.log_y_true)
            np.save(os.path.join(logs_dir, 'mean_pred.npy'), trainer.log_mean_pred)
            np.save(os.path.join(logs_dir, 'norm_mse.npy'), trainer.log_norm_mse)
            np.save(os.path.join(logs_dir, 'y_true_autoregressive.npy'), trainer.log_y_true_autoregressive)
            np.save(os.path.join(logs_dir, 'mean_pred_autoregressive.npy'), trainer.log_mean_pred_autoregressive)
            np.save(os.path.join(logs_dir, 'norm_mse_autoregressive.npy'), trainer.log_norm_mse_autoregressive)

            
            # get parent path of validation_buffer_path
            validation_buffer_parent_path = os.path.dirname(validation_buffer_path)
            gt_arm_data_path = os.path.join(validation_buffer_parent_path, 'save/task_1/traj_arm_data.npz')
            obstacle_npz_path = os.path.join(validation_buffer_parent_path, 'save/task_1/traj_obstacle_data.npz')
            if os.path.exists(gt_arm_data_path):
                print('Plotting single-step video with gt pred overlay...')
                from softae.envs.utils.post_processing import plot_video_with_gt_pred_cylinder_2D
                obstacle_npz_path = None if not os.path.exists(obstacle_npz_path) else obstacle_npz_path

                # plot single-step video with gt pred overlay
                log_mean_pred_path = os.path.join(logs_dir, 'mean_pred.npy')
                output_video_path = os.path.join(logs_dir, 'test_gt_single_pred_overlay.mp4')
                plot_video_with_gt_pred_cylinder_2D(
                    cylinders_history_path=obstacle_npz_path,
                    log_mean_pred_path=log_mean_pred_path,
                    gt_arm_data_path=gt_arm_data_path,
                    video_name=output_video_path,
                    fps=100,
                    step=1,
                    x_limits=(-1.0, 1.0),
                    y_limits=(-0.05, 1.0),
                    pred_color="#c2185b",
                    gt_line_color="#1f77b4",
                    gt_point_color="#1f77b4",
                    pred_marker_size=180,
                    gt_marker_size=70,
                    gt_line_width=2.0,
                    show_legend=False,
                    show_axes=True,
                    xlabel="x",
                    ylabel="y",
                )

                # plot autoregressive video with gt pred overlay
                print('Plotting autoregressive video with gt pred overlay...')  
                log_mean_pred_autoregressive_path = os.path.join(logs_dir, 'mean_pred_autoregressive.npy')
                output_video_path = os.path.join(logs_dir, 'test_gt_autoregressive_pred_overlay.mp4')
                plot_video_with_gt_pred_cylinder_2D(
                    cylinders_history_path=obstacle_npz_path,
                    log_mean_pred_path=log_mean_pred_autoregressive_path,
                    gt_arm_data_path=gt_arm_data_path,
                    video_name=output_video_path,
                    fps=100,
                    step=1,
                    x_limits=(-1.0, 1.0),
                    y_limits=(-0.05, 1.0),
                    pred_color="#c2185b",
                    gt_line_color="#1f77b4",
                    gt_point_color="#1f77b4",
                    pred_marker_size=180,
                    gt_marker_size=70,
                    gt_line_width=2.0,
                    show_legend=False,
                    show_axes=True,
                    xlabel="x",
                    ylabel="y",
                )
            else:
                print('GT arm data or obstacle data not found. Skipping video plotting.')

        ##########################################################################
        # Eval policy
        else:
            print("Starting evaluation")
            trainer.eval_policy(rng=trainer.rng, step=total_train_steps)
            print(f'Evaluation done')
            print(f'Reward task 1: {trainer.eval_rewards[0]}')
            print(f'Reward task 2: {trainer.eval_rewards[1]}')
        ##########################################################################
        
        trainer.save_agent(step=trainer.trainer_steps)

    result_dict = {
    }

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
        obstacle_cost=args.obstacle_cost,
        use_delta_action=bool(args.use_delta_action),
        delta_action=args.delta_action,
        use_action_obs=bool(args.use_action_obs),
        stack_obs_time=args.stack_obs_time,
        max_rate_of_change_of_activation=args.max_rate_of_change_of_activation,
        obs_state_points=args.obs_state_points,
        train_mode=bool(args.train_mode),
        task=args.task,
        damping=args.damping,
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
    parser.add_argument('--exp_name', type=str, required=True, default='neorl_pendulum')
    parser.add_argument('--train_mode', type=int, default=1)
    parser.add_argument('--task', type=str, default='reach')
    parser.add_argument('--exp_result_folder', type=str, default=None)
    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')
    parser.add_argument("--resume", type=str, default=None, help="Path to stored checkpoint from which to resume training")

    # env experiment args
    parser.add_argument('--time_limit', type=int, default=1000)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--damping', type=float, default=0.2)

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
    parser.add_argument('--obstacle_cost', type=float, default=0.0)
    parser.add_argument('--use_delta_action', type=int, default=0)
    parser.add_argument('--delta_action', type=float, default=1.0)
    parser.add_argument('--use_action_obs', type=int, default=0)
    parser.add_argument('--stack_obs_time', type=int, default=1)
    parser.add_argument('--obs_state_points', type=int, default=4)
    parser.add_argument('--max_rate_of_change_of_activation', type=float, default=np.infty)
    
    args = parser.parse_args()
    main(args)
