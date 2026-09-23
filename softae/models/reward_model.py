# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

import chex
import jax
from typing import Optional, Union

from softae.utils.replay_buffer import Transition


class RewardModel(object):
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, obs: chex.Array, action: chex.Array,
                next_obs: Optional[chex.Array] = None, rng: jax.random.PRNGKeyArray = None) -> chex.Array:
        pass

    def train_step(self, tran: Transition):
        pass

    def set_bounds(self, max_action: Union[chex.Array, float], min_action: Optional[Union[chex.Array, float]] = None):
        pass
