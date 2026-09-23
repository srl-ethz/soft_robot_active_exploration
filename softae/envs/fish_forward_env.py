# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym.utils import seeding
from flaresim.envs.core_v3 import FlareSimulator
from flaresim.envs.util import convert_observation_to_space, mapping, seconds_2_str
from softae.models.environment_models.soft_fish_reward import SoftFishForwardRewardModel as SoftFishRewardModel
from typing import Union, Dict
import time

from softae.envs.utils.base_renderer import BaseRenderer, BaseElasticaRendererSession


class GymFishEnv(gym.Env):
    """
    Superclass for all Gym-Fish environments.
    """

    def __init__(self,
                #  simulator,
                 env_file: str = None,
                 save_dir: str = './',
                 agent_name: str = "agent_0",
                 max_time: float = 10,
                 ctrl_dt: float = 0.1,
                 seed: int = 0,
                 fps_: int = 30,
                 COLLECT_DATA_FOR_POSTPROCESSING=False,
                 save_vtk_solid=False, save_vtk_fluid=False, save_vtk_coupling=False,
                 save_npy=False, save_vorticity=False, save_velocity_norm=False, save_traj_links=False,
                 reward_model=SoftFishRewardModel(),
                 use_angle_obs: bool = False,
                 ) -> None:
        self.np_random = None
        self.set_seed = seed
        self.seed(seed)
        self.eval_log_step = 0

        # Data related
        self._fps = fps_
        self._fps_count = 0
        self._save_vtk_solid = save_vtk_solid
        self._save_vtk_fluid = save_vtk_fluid
        self._save_vtk_coupling = save_vtk_coupling
        self._save_npy = save_npy
        self._save_vorticity = save_vorticity
        self._save_velocity_norm = save_velocity_norm
        self._save_traj_links = save_traj_links
        self._save_True = save_vtk_solid or save_vtk_fluid or save_vtk_coupling or save_npy or save_vorticity or save_velocity_norm or COLLECT_DATA_FOR_POSTPROCESSING
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING

        self.renderer = None
        self.render_mode = "rgb_array"

        # load the 'self.env_file' and create a CFD simulator 'self.simulator' by "flare simulator platform".
        if env_file is None:
            file_path = os.path.dirname(os.path.abspath(__file__))
            env_file = os.path.join(file_path, 'assets/fish/jsons/config_ib.json')
        self.env_file = env_file
        assert os.path.exists(self.env_file), f"Json file '{self.env_file}' does not exist!"
        self.log_dir = save_dir
        self._save_dir = save_dir
        # When creating this object, the specific simulator has not been instantiated yet!
        self.simulator = FlareSimulator(self.env_file, save_dir)

        # self.env_file = env_file
        # self._save_dir = save_dir
        # self.simulator = simulator

        self._agent_name = agent_name                              # The controlled agent
        self.max_time = max_time
        self.ctrl_dt = ctrl_dt

        # Other attributes you have to use.
        self.reward_model = reward_model
        self.use_angle_obs = use_angle_obs
        self.task = {}
        self.traj_pts = []
        self.traj_links_positions = []

        self.act_list = []

        # Assign 'action_space' and 'observation_space' which are necessary.
        self.time = 0.
        self.step_count = 0
        obs, _ = self.reset(config_file=self.env_file, save_dir=save_dir)
        self.observation_space = self._set_observation_space(obs)
        self.action_space = self._set_action_space()

    def reset(self, seed=0, specific_task=None, agent_name=None, config_file=None, save_dir=None, 
            #   save_vtk_solid=False, save_vtk_fluid=False, save_vtk_coupling=False, 
            #   save_npy=False, save_vorticity=False,
            #   save_velocity_norm=False, save_traj_links=False,
            #   fps_: int = 30, 
              render: bool = False, port: int = 5555, **kwargs):
        if agent_name is not None:
            self._agent_name = agent_name
        # self._fps = fps_
        # self._save_vtk_solid = save_vtk_solid
        # self._save_vtk_fluid = save_vtk_fluid
        # self._save_vtk_coupling = save_vtk_coupling
        # self._save_npy = save_npy
        # self._save_vorticity = save_vorticity
        # self._save_velocity_norm = save_velocity_norm
        # self._save_traj_links = save_traj_links
        self.time = 0.
        self.step_count = 0
        self._fps_count = 0
        # Clear the previous trajectory
        self.traj_pts = []
        self.traj_links_positions = []
        self.last_head_axis_in_world_frame = None
        self.act_list = []

        self.eval_log_step = kwargs.get('step', self.eval_log_step)
        self._save_dir = os.path.join(self.log_dir, 'step' + '-' + str(self.eval_log_step))

        if self._save_True:
            os.makedirs(self._save_dir, exist_ok=True)
        
        self._reset_all(config_file=config_file, save_dir=self._save_dir, specific_task=specific_task,
                        render=render, port=port)
        obs_ = self._get_obs(self.task)
        return obs_, {}

    # methods which have to override or modify in subclasses:
    # -------------------------------------------------------
    def _reset_all(self, config_file=None, save_dir=None, specific_task=None,
                   render: int = False, port: int = 5555) -> None:
        """
        This method runs only once in supclass method 'self.reset()', you can put anything you want to reset.
        """
        # Simulator
        self.simulator.reset(render=render, config_file=config_file, save_dir=self._save_dir, port=port)

    def _get_obs(self, task: dict) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        This method wrap the simulator status and task to 'observation' of the env.
        """
        # Update local matrix
        x_axis = self.simulator.agents[f"{self._agent_name}"].GetBaseLinkFwd()
        z_axis = self.simulator.agents[f"{self._agent_name}"].GetBaseLinkRight()

        head_axis_in_world_frame = x_axis

        # Calculate angel(Radian) between the head axis Vector from last frame to this frame
        if self.last_head_axis_in_world_frame is None:
            head_axis_angel_diff_in_world_frame = 0
        else:
            head_axis_angel_diff_in_world_frame = np.arccos(np.dot(head_axis_in_world_frame, self.last_head_axis_in_world_frame))
            if np.dot(self.last_head_axis_in_world_frame, z_axis) < 0:
                head_axis_angel_diff_in_world_frame = -head_axis_angel_diff_in_world_frame
        self.last_head_axis_in_world_frame = head_axis_in_world_frame
        self.last_fish_pos = self.simulator.agents[f"{self._agent_name}"].body_position

        if self.use_angle_obs:
            obs_ = np.concatenate((
                # Relative position of the target point
                # self.simulator.agents[f"{self._agent_name}"].body_position,
                # Velocity
                self.simulator.agents[f"{self._agent_name}"].body_velocity,
                # Head orientation
                head_axis_in_world_frame,
                # Head angular velocity
                np.array([head_axis_angel_diff_in_world_frame], dtype=np.float32),
                # Joint positions. (general coordinate)
                self.simulator.agents[f"{self._agent_name}"].joints_position() / 0.52,
                # swb: Joint velocities (general coordinate).
                self.simulator.agents[f"{self._agent_name}"].joints_velocity() / 10
            ), axis=0)
        else:
            obs_ = np.concatenate((
                # Relative position of the target point
                # self.simulator.agents[f"{self._agent_name}"].body_position,
                # Velocity
                self.simulator.agents[f"{self._agent_name}"].body_velocity,
                # Joint positions. (general coordinate)
                self.simulator.agents[f"{self._agent_name}"].joints_position() / 0.52,
                # swb: Joint velocities (general coordinate).
                self.simulator.agents[f"{self._agent_name}"].joints_velocity() / 10
            ), axis=0)

        return obs_

    # methods inherited from 'gym.Env' class, do not modify this if not necessary:
    # ----------------------------------------------------------------------------
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray):
        # do 'self.ctrl_dt' sec times iteration in simulator
        self.step_count += 1
        step_succeed = True
        time_before_env_step = time.time()
        while self.time < self.ctrl_dt*self.step_count:
            # do 'self.simulator.dt' sec times iteration in simulator
            step_succeed = self.simulator.step({f"{self._agent_name}": action})
            self._save_data_for_visual()
            self.time += self.simulator.dt
            # In some cases, we need to determine if the task has failed. If it has, we should break.
            if step_succeed is False:
                # If the execution fails, just exit directly!
                break
        time_after_env_step = time.time()
        # print(f"env step consumes: {time_after_env_step-time_before_env_step} seconds")

        self.traj_pts.append(self.simulator.agents[self._agent_name].body_position)
        self.act_list.append(action)
        # Get the observations
        terminate = False
        truncate = False
        state = self._get_obs(self.task)
        # Map the action to [-1,1] originally [-2, 2]
        normalized_action = np.abs(mapping(action, (self.action_space.low, self.action_space.high)))  
        # self.reward_model.target_position = self.task[f"{self._agent_name}"]['body_position']
        reward = self.reward_model.predict(obs=state, action=normalized_action, next_obs=state)
        reward = reward.astype(float).item()
        return_info = {
            "step_succeed": step_succeed,
            "reach_time_limit": (self.simulator.time > self.max_time),
        }
        return state, reward, terminate, truncate, return_info
    

    @staticmethod
    def _set_observation_space(observation):
        observation_space = convert_observation_to_space(observation)
        return observation_space

    def _set_action_space(self):
        low, high = np.float32(self.simulator.agents[self._agent_name].force_limits)  # <------------- here
        # If you change 'np.float32' above, you may get warning 'WARN:
        # Box bound precision lowered by casting to float32'.
        action_space = spaces.Box(low=low, high=high)
        action_space.seed(self.set_seed)
        return action_space

    def render(self):
        if self.render_mode == 'rgb_array':
            maxwidth = 800
            aspect_ratio = 3 / 4
            if self.renderer is None:
                from gym_softrobot.utils.render.matplotlib_renderer import Session_2D

                assert issubclass(
                    Session_2D, BaseRenderer
                ), "Rendering module is not properly subclassed"
                assert issubclass(
                    Session_2D, BaseElasticaRendererSession
                ), "Rendering module is not properly subclassed"
                self.renderer = Session_2D(width=maxwidth, height=int(maxwidth * aspect_ratio))
                fish_radius = 0.5
                self.renderer.add_point(self.last_fish_pos.tolist(), fish_radius)

            state_image = self.renderer.render(axis_range=[(-5.0, 5.0), (-5.0, 5.0)])
            self.renderer.close()
            self.renderer = None
            return state_image
        else:
            raise NotImplementedError("Rendering mode is not implemented")
        
    def post_processing(self, video_dir=None, filename_video=None, SAVE_DATA=True):
        self.save_traj()
        self.save_fbx()
        self.save_act()
        return

    # def close(self):
    #     print("##############################")
    #     print("##############################")
    #     print("##############################")
    #     print("##############################")
    #     print("##############################")
    #     print("##############################")
    #     self.simulator.close()
    #     del self.simulator

    # swb: 20*(old_dist - now_dist) -
    def _get_rew(self, task: dict, action: np.ndarray) -> float:
        body_position = self.simulator.agents[self._agent_name].body_position
        goal_position = task[self._agent_name]['body_position']

        self.old_distance = self.distance
        self.distance = np.linalg.norm(goal_position - body_position)

        dist_rew = (self.old_distance - self.distance) * 20.0
        ori_reward = 0

        act_reward = - np.sum(np.abs(mapping(action, (self.action_space.low, self.action_space.high))) ** 0.5) * 0.5

        total_rew = dist_rew + ori_reward + act_reward

        return total_rew

    # When it comes to termination, consider these scenarios: [a. collision b. timeout c. reaching the destination]
    # If any of them occurs, just terminate directly!
    def _get_done(self, task: dict) -> bool:
        done_ = False
        done_ = done_ or self.simulator.time > self.max_time
        done_ = done_ or self.simulator.is_failed
        return done_

    def _save_data_for_visual(self):
        if self._fps is not None:
            # swb:然后这边加一块保存图片的代码啊！！
            if self.simulator.time >= self._fps_count * (1 / self._fps):
                self._fps_count += 1
                self.simulator.save_vtk(self._fps_count, coupled_data=self._save_vtk_coupling,
                                        fluid_data=self._save_vtk_fluid, solid_data=self._save_vtk_solid)
                if self._save_npy:
                    self.simulator.save_npy_data(suffix=self._fps_count, slice_ratio=0.5)
                if self._save_vorticity:
                    self.simulator.save_vorticity(suffix=self._fps_count, v_min=-0.2, v_max=0.2, dpi=300, slice_ratio=0.5)
                if self._save_velocity_norm:
                    self.simulator.save_velocity_norm(suffix=self._fps_count, scale=2.0, dpi=300, slice_ratio=0.5)
                if self._save_traj_links:
                    # pyqtgraph is recommended http://vispy.org http://pyqtgraph.org/
                    # Open3D offers comprehensive functionality, well-encapsulated, clear interfaces, multiple
                    # algorithms, its own GUI system, and limited compatibility with Qt. Learning is required if you
                    # want to use it.
                    # open3d, vispy, pyqtgraph, which one is better?
                    self.traj_links_positions.append(self.simulator.get_links_positions(self._agent_name))
                    self.simulator.save_links_positions(prefix=self._fps_count,
                                                        title=seconds_2_str(self.simulator.time, 3),
                                                        target=[0, 0, 0])

    # save related:
    # -------------
    def save_traj(self, suffix: str = "result", dpi=300):
        # remove exist directories and create new ones to save some outputs of gymfish
        if os.path.exists(os.path.join(self._save_dir, "traj/")):
            shutil.rmtree(os.path.join(self._save_dir, "traj/"))
        os.makedirs(os.path.join(self._save_dir, "traj/"))

        domain = self.simulator.domain_size
        # goal_pos = self.task[self._agent_name]['body_position']
        traj_pts = self.traj_pts

        # build figure
        plt.figure("Gym-Fish")

        plt.xlim(0 - domain[0], 0 + domain[0])
        plt.ylim(0 - domain[2], 0 + domain[2])

        # plt.xticks(np.arange(0.5 - domain[0] / 2, domain[0] / 2, 0.5))
        # plt.yticks(np.arange(0.5 - domain[2] / 2, domain[2] / 2, 0.5))

        plt.grid()
        plt.gca().set_aspect(1)
        # plt.gca().invert_yaxis()

        # draw goal point
        # plt.plot([goal_pos[0]], [goal_pos[2]], color="red", marker=".")
        # plt.plot([goal_pos[2]], [goal_pos[0]], color="red", marker=".")

        plt.title("traj-map")
        # draw trajectory points
        if traj_pts is not None:
            # plt.scatter([p[0] for p in traj_pts], [p[2] for p in traj_pts], c=range(len(traj_pts)), marker=".",
            #             cmap='viridis')
            plt.scatter([p[2] for p in traj_pts], [p[0] for p in traj_pts], c=range(len(traj_pts)), marker=".",
                        cmap='viridis')
        plt.xlabel("z-axis")
        plt.ylabel("x-axis")
        # save figure and show
        plt.savefig(os.path.join(os.path.join(self._save_dir, "traj/"), f"{suffix}.png"), bbox_inches="tight", dpi=dpi)
        plt.close()  # was a bug: have to close figure otherwise the time consumption will increase.
        np.save(os.path.join(os.path.join(self._save_dir, "traj/"), f"{suffix}.npy"), np.array(traj_pts))

    def save_yaml_data(self, content: dict):
        self.simulator.save_yaml_data(content=content)

    def save_fbx(self):
        self.simulator.save_fbx()
    
    def save_act(self):
        os.makedirs(os.path.join(self._save_dir, "act/"), exist_ok=True)
        np.save(os.path.join(os.path.join(self._save_dir, "act/"), "action_history.npy"), np.array(self.act_list))


if __name__ == '__main__':
    import multiprocessing as mp
    # If this is not added here, the program will report the error.
    mp.set_start_method("spawn")
    file_path = os.path.dirname(os.path.abspath(__file__))
    env_file_path = os.path.join(file_path, 'assets/fish/jsons/config_ib.json')
    env = GymFishEnv(env_file=env_file_path,
                     save_dir='./',
                     seed=0,
                     max_time=15,
                     ctrl_dt=0.1,
                     save_vtk_solid=True, save_vtk_fluid=True, save_vtk_coupling=True,
                     fps_=30,
                     use_angle_obs=True,
                     )

    obs, _ = env.reset()

    for step_count in range(100):
        action = env.action_space.sample()
        next_obs, reward, terminate, truncate, info = env.step(action)
        obs = next_obs
        # One episode is finished
        if np.linalg.norm(env.simulator.agents[f"agent_0"].body_position -
                            task[f"agent_0"]['body_position']) < 0.2:
            print("Reach a goal point!")
            task = dict()
            task[f"agent_0"] = dict()
            goal_pos = np.array([env.np_random.uniform(-0.5, 2.5),
                                    0.0,
                                    env.np_random.uniform(-1.5, 1.5)])
            task[f"agent_0"]['body_position'] = goal_pos
            env.reset_task(task)
            print(f"Switch the task: {goal_pos}")