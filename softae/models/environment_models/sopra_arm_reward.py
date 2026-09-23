# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

import jax.numpy as jnp
import jax
from softae.models.reward_model import RewardModel
from functools import partial
from typing import Optional

import numpy as np

class SoftArmRewardModel(RewardModel):
    """Get Soft Arm Reward."""
    
    def __init__(self, ctrl_cost: float = 0.0, vel_cost: float = 0.0, tol: float = 5e-3, 
                 distance_weight: float = 1.0, orientation_weieght: float = 0.5,
                 target_position: np.array = np.array([0.3, 0.95, 0.0]),
                 target_orientation: Optional[np.array] = None,
                 obs_state_points: Optional[int] = 3,):
        super().__init__()
        self.ctrl_cost = ctrl_cost
        self.vel_cost = vel_cost
        self.tol = tol
        self.obs_state_points = obs_state_points
        self.tip_pos_index = 0
        self.tip_orien_index = 6*self.obs_state_points + 4*self.obs_state_points - 4
        self.distance_weight = distance_weight
        self.orientation_weieght = orientation_weieght
        self.target_position = target_position
        self.target_orientation = target_orientation
        self.tol_rew = ToleranceReward(bounds=(0, self.tol), margin=20 * self.tol, value_at_margin=0.1, sigmoid='long_tail') 

    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, next_obs=None, rng=None):
        # Calculate the position distance to the target
        tip_to_target = jnp.array(obs[self.tip_pos_index:self.tip_pos_index+3]) - jnp.array(self.target_position)
        dist = jnp.linalg.norm(tip_to_target)
        
        # Weighted sum to total reward
        reward = self.distance_weight * self.tol_rew(dist) 
        return reward.reshape(-1).squeeze()
    

from typing import Tuple
import chex

class Sigmoids:
    def __init__(self, sigmoid: str, value_at_the_margin: float = 0.1):
        self.sigmoid = sigmoid
        self.value_at_the_margin = value_at_the_margin
    def __call__(self, x, value_at_1):
        if self.sigmoid == 'gaussian”':
            return self._gaussian(x, value_at_1)
        elif self.sigmoid == 'hyperbolic':
            return self._hyperbolic(x, value_at_1)
        elif self.sigmoid == 'long_tail':
            return self._long_tail(x, value_at_1)
        elif self.sigmoid == 'reciprocal':
            return self._reciprocal(x, value_at_1)
        elif self.sigmoid == 'cosine':
            return self._cosine(x, value_at_1)
        elif self.sigmoid == 'linear':
            return self._linear(x, value_at_1)
        elif self.sigmoid == 'quadratic':
            return self._quadratic(x, value_at_1)
        elif self.sigmoid == 'tanh_squared':
            return self._tanh_squared(x, value_at_1)
    @staticmethod
    def _gaussian(x, value_at_1):
        scale = jnp.sqrt(-2 * jnp.log(value_at_1))
        return jnp.exp(-0.5 * (x * scale) ** 2)
    @staticmethod
    def _hyperbolic(x, value_at_1):
        scale = jnp.arccosh(1 / value_at_1)
        return 1 / jnp.cosh(x * scale)
    @staticmethod
    def _long_tail(x, value_at_1):
        scale = jnp.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)
    @staticmethod
    def _reciprocal(x, value_at_1):
        scale = 1 / value_at_1 - 1
        return 1 / (jnp.abs(x) * scale + 1)
    @staticmethod
    def _cosine(x, value_at_1):
        scale = jnp.arccos(2 * value_at_1 - 1) / jnp.pi
        scaled_x = x * scale
        cos_pi_scaled_x = jnp.cos(jnp.pi * scaled_x)
        return jnp.where(jnp.abs(scaled_x) < 1, (1 + cos_pi_scaled_x) / 2, 0.0)
    @staticmethod
    def _linear(x, value_at_1):
        scale = 1 - value_at_1
        scaled_x = x * scale
        return jnp.where(jnp.abs(scaled_x) < 1, 1 - scaled_x, 0.0)
    @staticmethod
    def _quadratic(x, value_at_1):
        scale = jnp.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return jnp.where(jnp.abs(scaled_x) < 1, 1 - scaled_x ** 2, 0.0)
    @staticmethod
    def _tanh_squared(x, value_at_1):
        scale = jnp.arctanh(jnp.sqrt(1 - value_at_1))
        return 1 - jnp.tanh(x * scale) ** 2
    
class ToleranceReward:
    def __init__(self,
                 bounds: Tuple[float, float] = (0.0, 0.0),
                 margin: float = 0.0,
                 sigmoid: str = 'gaussian',
                 value_at_margin: float = 0.1):
        self.bounds = bounds
        self.margin = margin
        self.value_at_margin = value_at_margin
        self._sigmoid = sigmoid
        self.sigmoid = Sigmoids(sigmoid)
        lower, upper = bounds
        self.lower = lower
        self.upper = upper
        if lower > upper:
            raise ValueError('Lower bound must be <= upper bound.')
        if margin < 0:
            raise ValueError('`margin` must be non-negative.')
    def __call__(self, x: chex.Array) -> chex.Array:
        in_bounds = jnp.logical_and(self.lower <= x, x <= self.upper)
        if self.margin == 0:
            return jnp.where(in_bounds, 1.0, 0.0)
        else:
            d = jnp.where(x < self.lower, self.lower - x, x - self.upper) / self.margin
            return jnp.where(in_bounds, 1.0, self.sigmoid(d, self.value_at_margin))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    bound = 5e-3
    reward = ToleranceReward(bounds=(0, bound), margin=20 * bound, value_at_margin=0.1, sigmoid='long_tail')
    x = jnp.linspace(0, 0.2, 1000)
    y = reward(x)
    fig = plt.figure()
    plt.plot(x, y)
    fig.savefig('reward_sopra.png')

    rest = np.array([0.27433328, -0.00670465, 0.03192269])
    target = np.array([0.27784566, 0.03181241, 0.07597183])
    target2 = np.array([0.27524131, 0.0505609, -0.03850071])
    target3 = np.array([0.2497562, -0.09393489, 0.03902382])
    print(np.linalg.norm(rest-target3))
    print(reward(np.linalg.norm(rest-target3)))


    # plt.show()

