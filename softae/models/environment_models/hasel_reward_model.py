# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

import jax
from softae.models.reward_model import RewardModel
from functools import partial

class HASELRewardModel(RewardModel):
    """Get Soft Arm Reward."""
    
    def __init__(self, forward_velocity_weight: float = 1.0):
        super().__init__()
        self.forward_velocity_weight = forward_velocity_weight
       
    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, previous_obs=None, rng=None):
        reward_hop = self.forward_velocity_weight * obs[5]
        reward = reward_hop
        return reward.reshape(-1).squeeze()
