# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

from jax import random
import wandb
from tqdm import tqdm
from softae.trainer.dummy_trainer import DummyTrainer
from softae.agents.actor_critic.sac import SACAgent
import numpy as np


class OffPolicyTrainer(DummyTrainer):
    def __init__(self,
                 agent_name: str = "OffPolicyAgent",
                 *args,
                 **kwargs
                 ):

        super(OffPolicyTrainer, self).__init__(agent_name=agent_name, *args, **kwargs)
        assert isinstance(self.agent, SACAgent), "Only Off policy agents are allowed"
        self.trainer_steps = 0
        self.model_training_steps = 0
        self.env_steps_done = 0
        self.best_performance = None

    def train(self):
        if self.use_wandb:
            wandb.define_metric('env_steps')
        self.rng, eval_rng = random.split(self.rng, 2)
        eval_rng, curr_eval = random.split(eval_rng, 2)
        reward_log = self.eval_policy(rng=curr_eval)
        best_performance = reward_log['eval/reward_task_0']
        reward_log['env_steps'] = 0
        reward_log['learning_step'] = 0
        reward_log['train_steps'] = 0
        train_steps = 0
        self.save_agent(0)
        if self.use_wandb:
            wandb.define_metric("env_steps")
            wandb.define_metric("train_steps")
            wandb.log(reward_log)
        policy = lambda x, y: np.concatenate([self.env.action_space.sample().reshape(1, -1)
                                              for s in range(self.num_envs)], axis=0)
        transitions = self.rollout_policy(self.exploration_steps, policy, self.rng)
        self.buffer.add(transition=transitions)
        rng_keys = random.split(self.rng, self.total_train_steps + 1)
        self.rng = rng_keys[0]
        rng_keys = rng_keys[1:]
        learning_steps = int(self.total_train_steps/(self.rollout_steps*self.num_envs))
        rng_key, reset_rng = random.split(rng_keys[0], 2)
        rng_keys = rng_keys.at[0].set(rng_key)
        reset_seed = random.randint(
            reset_rng,
            (1,),
            minval=0,
            maxval=int(learning_steps*self.rollout_steps)).item()
        obs, _ = self.env.reset(seed=reset_seed)
        for step in tqdm(range(learning_steps)):
            actor_rng, train_rng = random.split(rng_keys[step], 2)
            policy = self.agent.act_in_train
            transitions, obs, done = self.step_env(obs, policy, self.rollout_steps, actor_rng)

            self.buffer.add(transitions)
            reward_log = {}
            train_step_log = {}
            self.env_steps_done += self.rollout_steps * self.num_envs
            self.trainer_steps += 1
            env_step_log = {
                'env_steps': self.env_steps_done,
                'learning_step': self.trainer_steps,
            }
            if step % self.train_freq == 0:
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
                train_steps += total_train_steps
            # Evaluate episode
            if self.trainer_steps % self.eval_freq == 0:
                eval_rng, curr_eval = random.split(eval_rng, 2)
                reward_log = self.eval_policy(rng=curr_eval, step=self.trainer_steps)
                if self.use_wandb:
                    wandb.log(reward_log)
                if reward_log['eval/reward_task_0'] > best_performance:
                    best_performance = reward_log['eval/reward_task_0']
                    self.save_agent(self.trainer_steps)
            if self.use_wandb:
                train_log = env_step_log
                train_log.update(train_step_log)
                train_log.update(reward_log)
                wandb.log(train_log)

        self.save_agent(step, agent_name="final_agent")

    @property
    def num_reward_models(self) -> int:
        return 2








