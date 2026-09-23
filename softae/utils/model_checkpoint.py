from softae.utils.replay_buffer import ReplayBuffer, Transition, get_past_values_transition, EpisodicReplayBuffer, Normalizer
from softae.models.bayesian_dynamics_model import BayesianDynamicsModel, ModelProperties
from softae.models.active_learning_model import ActiveLearningHUCRLModel
from softae.utils.type_aliases import ModelProperties
from typing import Union
import pickle
import copy
import numpy as np
import jax.numpy as jnp

class BayesianModelCheckpoint:
    def __init__(self, args, model_params, model_opt_state, alpha, model_props: ModelProperties) -> None:
        self.args = args
        self.model_params = copy.deepcopy(model_params)
        self.model_opt_state = copy.deepcopy(model_opt_state)
        self.alpha = alpha
        self.model_props = copy.deepcopy(model_props)

    def create_model(self, model):
        model.update_model(self.model_params, self.model_opt_state, self.alpha)
        model.set_transforms(
            bias_obs=self.model_props.bias_obs,
            bias_act=self.model_props.bias_act,
            bias_out=self.model_props.bias_out,
            scale_obs=self.model_props.scale_obs,
            scale_act=self.model_props.scale_act,
            scale_out=self.model_props.scale_out
        )
        return model
    
    @staticmethod
    def create_checkpoint(model: BayesianDynamicsModel, args=None):
        return BayesianModelCheckpoint(
            args, model.model_params, model.model_opt_state, model.model_props.alpha,
            model.model_props
        )
    
    def restore_model(self, model: BayesianDynamicsModel):
        model.update_model(self.model_params, self.model_opt_state, self.alpha)
        model.set_transforms(
            bias_obs=self.model_props.bias_obs,
            bias_act=self.model_props.bias_act,
            bias_out=self.model_props.bias_out,
            scale_obs=self.model_props.scale_obs,
            scale_act=self.model_props.scale_act,
            scale_out=self.model_props.scale_out
        )
