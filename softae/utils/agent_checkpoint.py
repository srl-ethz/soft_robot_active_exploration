from softae.agents.actor_critic.sac import SACAgent
from softae.optimizers.sac_based_optimizer import SACOptimizer
from softae.agents.model_based.model_based_agent import ModelBasedAgent
from softae.optimizers.sac_based_optimizer import SacOptimizerState
from softae.utils.model_checkpoint import BayesianModelCheckpoint
from typing import Tuple
import copy

class SACCheckpoint:
    def __init__(self, args, opt_state):
        self.args = args
        self.opt_state = copy.deepcopy(opt_state)

    @staticmethod
    def create_checkpoint(optimizer: SACOptimizer):
        return SACCheckpoint(None, optimizer.optimizer_state)
    
    def restore_checkpoint(self, optimizer: SACOptimizer):
        optimizer.optimizer_state = copy.deepcopy(self.opt_state)

    def create_agent(self) -> Tuple[SACAgent, SacOptimizerState]:
        agent = SACAgent(**self.args)
        return agent, self.opt_state

class ModelBasedAgentCheckpoint:
    def __init__(self, agent: ModelBasedAgent) -> None:
        self.model_check = BayesianModelCheckpoint.create_checkpoint(agent.dynamics_model)
        self.opt_check = None
        if isinstance(agent.policy_optimizer, SACAgent):
            self.opt_check = SACCheckpoint.create_checkpoint(agent.policy_optimizer)
    
    def restore_checkpoint(self, agent: ModelBasedAgent):
        self.model_check.restore_model(agent.dynamics_model)
        if self.opt_check is not None:
            self.opt_check.restore_checkpoint(agent.policy_optimizer)