# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

import jax.random
from softae.utils.replay_buffer import ReplayBuffer, Transition
from softae.trainer.dummy_trainer import DummyTrainer, DummyTrainerState
from softae.agents.model_based.model_based_agent import ModelBasedAgent
import wandb
from jax import random
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class ModelBasedTrainerState(DummyTrainerState):
    validation_buffer: ReplayBuffer
    validation_batch_size: int
    uniform_exploration: bool
    env_steps_done: int
    trainer_steps: int
    model_training_steps: int
    best_performance: float

class ModelBasedTrainer(DummyTrainer):
    def __init__(self,
                 agent: ModelBasedAgent,
                 agent_name: str = "ModelBasedAgent",
                 validation_buffer_size: int = 0,
                 validation_batch_size: int = 1024,
                 uniform_exploration: bool = False,
                 *args,
                 **kwargs,
                 ):
        """

        :param agent: ModelBasedAgent, RL Agent
        :param agent_name: str, name of agent
        :param validation_buffer_size: int, size of validation buffer
        :param validation_batch_size: int, batch size for validation
        :param uniform_exploration: bool, if random/uniform exploration is done
        :param args:
        :param kwargs:
        """

        super(ModelBasedTrainer, self).__init__(agent=agent, agent_name=agent_name, *args, **kwargs)
        assert isinstance(self.agent, ModelBasedAgent), "Only Model based agents are allowed"
        self.validation_buffer = None
        self.validation_batch_size = validation_batch_size
        self.collect_validation_data(validation_buffer_size)
        self.uniform_exploration = uniform_exploration
        self.trainer_steps = 0
        self.model_training_steps = 0
        self.env_steps_done = 0
        self.best_performance = None

    def _get_state(self) -> ModelBasedTrainerState:
        dummy_state = super()._get_state()
        return ModelBasedTrainerState(
            **vars(dummy_state),
            validation_buffer=self.validation_buffer,
            validation_batch_size=self.validation_batch_size,
            uniform_exploration=self.uniform_exploration,
            env_steps_done=self.env_steps_done,
            trainer_steps=self.trainer_steps,
            model_training_steps=self.model_training_steps,
            best_performance=self.best_performance,
        )

    def _set_state(self, state: ModelBasedTrainerState):
        super()._set_state(state)
        self.validation_buffer = state.validation_buffer
        self.validation_batch_size = state.validation_batch_size
        self.uniform_exploration = state.uniform_exploration
        self.env_steps_done = state.env_steps_done
        self.trainer_steps = state.trainer_steps
        self.model_training_steps = state.model_training_steps
        self.best_performance = state.best_performance

    def collect_validation_data(self, validation_buffer_size: int = 0):
        """
        Collect data for validation if validation_buffer_size > 0
        :param validation_buffer_size: int
        :return:
        """
        if validation_buffer_size > 0:
            self.validation_buffer = ReplayBuffer(
                obs_shape=self.buffer.obs_shape,
                action_shape=self.buffer.action_shape,
                normalize=False,
                action_normalize=False,
                learn_deltas=False
            )
            num_points = int(validation_buffer_size * self.num_envs)
            obs_shape = (num_points,) + self.env.observation_space.shape
            action_space = (num_points,) + self.env.action_space.shape
            obs_vec = np.zeros(obs_shape)
            action_vec = np.zeros(action_space)
            reward_vec = np.zeros((num_points,))
            next_obs_vec = np.zeros(obs_shape)
            done_vec = np.zeros((num_points,))
            for step in range(validation_buffer_size):
                obs = self.env.observation_space.sample()
                action = self.env.action_space.sample()
                obs_vec[step * self.num_envs: (step + 1) * self.num_envs] = obs
                action_vec[step * self.num_envs: (step + 1) * self.num_envs] = action

            transitions = Transition(
                obs=obs_vec,
                action=action_vec,
                reward=reward_vec,
                next_obs=next_obs_vec,
                done=done_vec,
            )

            #policy = lambda x, y: np.concatenate(
            #    [self.env.action_space.sample().reshape(1, -1)
            #     for s in range(self.num_envs)], axis=0)
            #self.rng, val_rng = random.split(self.rng, 2)
            #transitions = self.rollout_policy(validation_buffer_size,
            #                                  policy,
            #                                  val_rng
            #                                  )
            self.validation_buffer.add(transitions)

    def validate_model(self, rng: jax.random.PRNGKeyArray, shuffle=True, autoregressive=False, start_step: int = 0) -> dict:
        """
        Validates learned dynamics model
        :param rng: random key for validation
        :return model_log: dict, logs for model validation
        """
        model_log = {}
        if self.validation_buffer is not None:
            if shuffle:
                val_tran = self.validation_buffer.sample(
                    rng=rng,
                    batch_size=self.validation_batch_size,
                )
            else:
                val_tran = self.validation_buffer.sample_time_window(
                    rng=rng,
                    batch_size=self.validation_batch_size,
                )
            if autoregressive:
                mean_pred, std_pred = self.agent.predict_next_state_autoregressive(val_tran, start_step)
            else:
                mean_pred, std_pred = self.agent.predict_next_state(val_tran)
            # print(f'mean_pred: {mean_pred}')
            if self.agent.is_gp:
                eps_uncertainty = std_pred
                al_uncertainty = jnp.ones_like(std_pred) * 1e-3
            else:
                eps_uncertainty = jnp.std(mean_pred, axis=0)
                al_uncertainty = jnp.sqrt(jnp.mean(jnp.square(std_pred), axis=0))
            frac = eps_uncertainty / (al_uncertainty + 1e-6)
            information_gain = jnp.sum(jnp.log(1 + jnp.square(frac)), axis=-1)
            max_info_gain = jnp.max(information_gain)
            mean_info_gain = jnp.mean(information_gain)
            eps_uncertainty = jnp.sum(eps_uncertainty, axis=-1)
            mean_eps_uncertainty = jnp.mean(eps_uncertainty)
            max_eps_uncertainty = jnp.max(eps_uncertainty)
            std_eps_uncertainty = jnp.std(eps_uncertainty)
            std_pred = jnp.mean(jnp.sum(al_uncertainty, axis=-1))
            y_true = val_tran.next_obs
            print(f'y_true: {y_true.shape}, mean_pred: {mean_pred.shape}, std_pred: {std_pred.shape}')
            print(f'mean_mean_pred: {jnp.mean(mean_pred, axis=0).shape}')
            print(f'se: {jnp.square(y_true - jnp.mean(mean_pred, axis=0)).shape}')

            
            # Raw mean squared error
            # state without angular velocity
            mse_wo_w = jnp.mean(jnp.square(y_true[:, :46] - jnp.mean(mean_pred, axis=0)[:, :46]))
            # state without linear and angular velocity
            y_true_sub = np.hstack((y_true[:, 0:15], y_true[:, 30:46]))
            mean_mean_pred_sub = np.hstack((jnp.mean(mean_pred, axis=0)[:, 0:15], jnp.mean(mean_pred, axis=0)[:, 30:46]))
            mse_wo_vw = jnp.mean(jnp.square(y_true_sub - mean_mean_pred_sub))
            # full state
            mse = jnp.mean(jnp.square(y_true - jnp.mean(mean_pred, axis=0)))

            # Normalized mean squared error
            # full state
            # y_true_mean = jnp.mean(y_true, axis=0)
            y_true_std = jnp.std(y_true, axis=0)
            norm_mse = jnp.mean(jnp.square((y_true - jnp.mean(mean_pred, axis=0))/(y_true_std + 1e-6)))
            # state without angular velocity
            norm_mse_wo_w = jnp.mean(jnp.square((y_true[:, :46] - jnp.mean(mean_pred, axis=0)[:, :46])/(y_true_std[:46] + 1e-6)))
            # state without linear and angular velocity
            y_true_sub_std = jnp.std(y_true_sub, axis=0)
            norm_mse_wo_vw = jnp.mean(jnp.square((y_true_sub - mean_mean_pred_sub)/(y_true_sub_std + 1e-6)))

            # save y_true and mean_pred to variable
            if autoregressive:
                self.log_y_true_autoregressive = y_true
                self.log_mean_pred_autoregressive = jnp.mean(mean_pred, axis=0)
                self.log_norm_mse_autoregressive = jnp.square((y_true - jnp.mean(mean_pred, axis=0))/(y_true_std + 1e-6))
            else:
                self.log_y_true = y_true
                self.log_mean_pred = jnp.mean(mean_pred, axis=0)
                self.log_norm_mse = jnp.square((y_true - jnp.mean(mean_pred, axis=0))/(y_true_std + 1e-6))
            
            # original code
            # mse = jnp.mean(jnp.sum(jnp.square(y_true - mean_pred), axis=-1))
            
            model_log = {
                'validation_mse': mse.astype(float).item(),
                'validation_mse_wo_w': mse_wo_w.astype(float).item(),
                'validation_mse_wo_vw': mse_wo_vw.astype(float).item(),
                'validation_norm_mse': norm_mse.astype(float).item(),
                'validation_norm_mse_wo_w': norm_mse_wo_w.astype(float).item(),
                'validation_norm_mse_wo_vw': norm_mse_wo_vw.astype(float).item(),
                'validation_al_std': std_pred.astype(float).item(),
                'validation_eps_std_mean': mean_eps_uncertainty.astype(float).item(),
                'validation_eps_std_max': max_eps_uncertainty.astype(float).item(),
                'validation_eps_std_std': std_eps_uncertainty.astype(float).item(),
                'max_info_gain': max_info_gain.astype(float).item(),
                'mean_info_gain': mean_info_gain.astype(float).item(),
            }
        return model_log

    def validate_model_autoregressive(self, rng: jax.random.PRNGKeyArray) -> dict:
        """
        Validates learned dynamics model
        :param rng: random key for validation
        :return model_log: dict, logs for model validation
        """
        model_log = {}
        if self.validation_buffer is not None:
            val_tran = self.validation_buffer.sample_time_window(
                rng=rng,
                batch_size=self.validation_batch_size,
            )
            mean_pred, std_pred = self.agent.predict_next_state(val_tran)
            # print(f'mean_pred: {mean_pred}')
            if self.agent.is_gp:
                eps_uncertainty = std_pred
                al_uncertainty = jnp.ones_like(std_pred) * 1e-3
            else:
                eps_uncertainty = jnp.std(mean_pred, axis=0)
                al_uncertainty = jnp.sqrt(jnp.mean(jnp.square(std_pred), axis=0))
            frac = eps_uncertainty / (al_uncertainty + 1e-6)
            information_gain = jnp.sum(jnp.log(1 + jnp.square(frac)), axis=-1)
            max_info_gain = jnp.max(information_gain)
            mean_info_gain = jnp.mean(information_gain)
            eps_uncertainty = jnp.sum(eps_uncertainty, axis=-1)
            mean_eps_uncertainty = jnp.mean(eps_uncertainty)
            max_eps_uncertainty = jnp.max(eps_uncertainty)
            std_eps_uncertainty = jnp.std(eps_uncertainty)
            std_pred = jnp.mean(jnp.sum(al_uncertainty, axis=-1))
            y_true = val_tran.next_obs
            print(f'y_true: {y_true.shape}, mean_pred: {mean_pred.shape}, std_pred: {std_pred.shape}')
            print(f'mean_mean_pred: {jnp.mean(mean_pred, axis=0).shape}')
            print(f'se: {jnp.square(y_true - jnp.mean(mean_pred, axis=0)).shape}')

            
            # Raw mean squared error
            # state without angular velocity
            mse_wo_w = jnp.mean(jnp.square(y_true[:, :46] - jnp.mean(mean_pred, axis=0)[:, :46]))
            # state without linear and angular velocity
            y_true_sub = np.hstack((y_true[:, 0:15], y_true[:, 30:46]))
            mean_mean_pred_sub = np.hstack((jnp.mean(mean_pred, axis=0)[:, 0:15], jnp.mean(mean_pred, axis=0)[:, 30:46]))
            mse_wo_vw = jnp.mean(jnp.square(y_true_sub - mean_mean_pred_sub))
            # full state
            mse = jnp.mean(jnp.square(y_true - jnp.mean(mean_pred, axis=0)))

            # Normalized mean squared error
            # full state
            # y_true_mean = jnp.mean(y_true, axis=0)
            y_true_std = jnp.std(y_true, axis=0)
            norm_mse = jnp.mean(jnp.square((y_true - jnp.mean(mean_pred, axis=0))/(y_true_std + 1e-6)))
            # state without angular velocity
            norm_mse_wo_w = jnp.mean(jnp.square((y_true[:, :46] - jnp.mean(mean_pred, axis=0)[:, :46])/(y_true_std[:46] + 1e-6)))
            # state without linear and angular velocity
            y_true_sub_std = jnp.std(y_true_sub, axis=0)
            norm_mse_wo_vw = jnp.mean(jnp.square((y_true_sub - mean_mean_pred_sub)/(y_true_sub_std + 1e-6)))

            # save y_true and mean_pred to variable
            self.log_y_true = y_true
            self.log_mean_pred = jnp.mean(mean_pred, axis=0)
            self.log_norm_mse = jnp.square((y_true - jnp.mean(mean_pred, axis=0))/(y_true_std + 1e-6))
            
            # original code
            # mse = jnp.mean(jnp.sum(jnp.square(y_true - mean_pred), axis=-1))
            
            model_log = {
                'validation_mse': mse.astype(float).item(),
                'validation_mse_wo_w': mse_wo_w.astype(float).item(),
                'validation_mse_wo_vw': mse_wo_vw.astype(float).item(),
                'validation_norm_mse': norm_mse.astype(float).item(),
                'validation_norm_mse_wo_w': norm_mse_wo_w.astype(float).item(),
                'validation_norm_mse_wo_vw': norm_mse_wo_vw.astype(float).item(),
                'validation_al_std': std_pred.astype(float).item(),
                'validation_eps_std_mean': mean_eps_uncertainty.astype(float).item(),
                'validation_eps_std_max': max_eps_uncertainty.astype(float).item(),
                'validation_eps_std_std': std_eps_uncertainty.astype(float).item(),
                'max_info_gain': max_info_gain.astype(float).item(),
                'mean_info_gain': mean_info_gain.astype(float).item(),
            }
        return model_log

    def train(self):
        """
        Train function for model based agent
        """
        if self.use_wandb:
            wandb.define_metric('env_steps')
            wandb.define_metric('learning_step')
            wandb.define_metric("train_steps")
            wandb.define_metric('eval/*', step_metric='learning_step')
        # returns policy used for random exploration
        exploration_policy = lambda x, y: np.concatenate([self.env.action_space.sample().reshape(1, -1) 
                                                          for s in range(self.num_envs)], axis=0)

        self.rng, eval_rng = random.split(self.rng, 2)
        eval_rng, curr_eval = random.split(eval_rng, 2)
        if self.trainer_steps == 0:
            self.agent.set_transforms(
                bias_obs=self.buffer.state_normalizer.mean,
                bias_act=self.buffer.action_normalizer.mean,
                bias_out=self.buffer.next_state_normalizer.mean,
                scale_obs=self.buffer.state_normalizer.std,
                scale_act=self.buffer.action_normalizer.std,
                scale_out=self.buffer.next_state_normalizer.std,
            )
            curr_eval, eval_val_rng = jax.random.split(curr_eval, 2)
            model_log = self.validate_model(eval_val_rng)
            reward_log = self.eval_policy(rng=curr_eval)
            best_performance = reward_log['eval/reward_task_1']
            reward_log['env_steps'] = 0
            reward_log['learning_step'] = 0
            reward_log['train_steps'] = 0
            reward_log['update_steps_per_iter'] = 0
            self.save_agent(0)
            reward_log.update(model_log)
            if self.use_wandb:
                wandb.log(reward_log)

            # collect data with the random policy
            policy = exploration_policy
            self.rng, explore_rng = random.split(self.rng, 2)
            if self.exploration_steps > 0:
                transitions = self.rollout_policy(self.exploration_steps, policy, explore_rng)
                self.buffer.add(transitions)
        rng_keys = random.split(self.rng, self.total_train_steps + 1)
        self.rng = rng_keys[0]
        rng_keys = rng_keys[1:]
        learning_steps = int(self.total_train_steps / (self.rollout_steps * self.num_envs))
        rng_key, reset_rng = random.split(rng_keys[0], 2)
        rng_keys = rng_keys.at[0].set(rng_key)
        reset_seed = random.randint(
            reset_rng,
            (1,),
            minval=0,
            maxval=int(learning_steps * self.rollout_steps)).item()
        # reset env before training starts
        obs, _ = self.env.reset(seed=reset_seed)

        accumulated_reward = 0.0
        num_resets = 0
        for step in tqdm(range(self.trainer_steps, learning_steps)):
            # collect rollouts with policy
            actor_rng, train_rng = random.split(rng_keys[step], 2)
            policy = self.agent.act_in_train if not self.uniform_exploration else exploration_policy
            actor_rng, val_rng = random.split(actor_rng, 2)
            transitions, obs, done = self.step_env(obs, policy, self.rollout_steps, actor_rng)
            self.buffer.add(transitions)
            self.agent.set_transforms(
                bias_obs=self.buffer.state_normalizer.mean,
                bias_act=self.buffer.action_normalizer.mean,
                bias_out=self.buffer.next_state_normalizer.mean,
                scale_obs=self.buffer.state_normalizer.std,
                scale_act=self.buffer.action_normalizer.std,
                scale_out=self.buffer.next_state_normalizer.std,
            )
            accumulated_reward = accumulated_reward + transitions.reward.reshape(
                (self.num_envs, self.rollout_steps, -1)
            ).mean(0).sum().item()
            self.agent.update_posterior(self.buffer)
            num_resets += transitions.done.sum().item()
            reward_log = {}
            train_step_log = {}
            model_log = {}
            self.env_steps_done += self.rollout_steps * self.num_envs
            self.trainer_steps += 1
            env_step_log = {
                'env_steps': self.env_steps_done,
                'learning_step': self.trainer_steps,
            }
            # update agent
            if step % self.train_freq == 0 and (self.buffer.size >= self.agent.batch_size or self.agent.is_gp):
                train_rng, agent_rng = random.split(train_rng, 2)
                total_train_steps = self.agent.train_step(
                    rng=agent_rng,
                    buffer=self.buffer,
                    validate=self.validate,
                    log_results=self.use_wandb,
                )
                self.model_training_steps += total_train_steps
                train_step_log = {
                    'train_steps': self.model_training_steps,
                    'update_steps_per_iter': total_train_steps,
                }
                eval_rng, eval_val_rng = jax.random.split(eval_rng, 2)
                model_log = self.validate_model(eval_val_rng)
            # Evaluate episode
            if self.trainer_steps % self.eval_freq == 0 and self.model_training_steps > 0:
                eval_rng, curr_eval = random.split(eval_rng, 2)
                reward_log = self.eval_policy(rng=curr_eval, step=self.trainer_steps)
                print("\n#########################")
                print(f"Evaluating at trainer_step: {self.trainer_steps}")
                print(reward_log)
                print("#########################")
            if self.trainer_steps % 50 == 0:
                self.save_agent(self.trainer_steps)
            if self.use_wandb:
                train_log = env_step_log
                train_log.update(train_step_log)
                train_log.update(reward_log)
                train_log.update(model_log)
                scalar_dict = {'scale_out': np.mean(self.buffer.next_state_normalizer.std).astype(float).item()}
                train_log.update(scalar_dict)
                wandb.log(train_log)
                print("\n#########################")
                print(train_log)
                print("#########################")
        self.save_agent(self.trainer_steps, agent_name="final_agent")
    

    @property
    def num_reward_models(self) -> int:
        return self.agent.num_dynamics_models
