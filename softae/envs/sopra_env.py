# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

import time
import numpy as np
from typing import Optional
from gym import core
from gym import spaces
import os.path
import matplotlib.pyplot as plt
from softae.envs.utils.rater import Rate
from softae.models.environment_models.sopra_arm_reward import SoftArmRewardModel
from softae.envs.utils.hardware_interfaces.sopra_hardware_interface import SoPrAHWInterface
from softae.envs.utils.convert_frame_state import FramesStateReader
from copy import deepcopy

class SopraTrackingEnv(core.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}
    def __init__(
        self,
        hw_intf: SoPrAHWInterface,
        freq_hz: float = 1.0e1,
        COLLECT_DATA_FOR_POSTPROCESSING: bool = False,
        reward_model: SoftArmRewardModel = SoftArmRewardModel(),
        save_dir: str = './', 
        **kwargs
    ):
        
        # create folder for saving data
        self.log_dir = save_dir
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.action_repeat = 10
        self.sleep_rate = Rate(1.0 / freq_hz / self.action_repeat)

        self.eval_log_step = 0

        self.max_env_step = kwargs.get("max_env_step", 500)

        self.hw_intf = hw_intf
        self.frame_reader = FramesStateReader()
        self.prev_poses = None
        self.prev_timestamp = None

        self.action_delta_pressure_scale = 100.0
        self.max_pressure_safe = np.array([500, 500, 500, 250, 250, 250]) 
        self.previous_pressure = np.zeros_like(self.max_pressure_safe)
        assert (
                len(self.max_pressure_safe) == self.hw_intf.num_chambers
            ), "Number of air chambers does not match dimension of the max pressure limit"

        self.action_space = spaces.Box(
            low=0,
            high=1.0,
            shape=(self.hw_intf.num_chambers,),
            dtype=np.float64,
        )
        
        self.num_obs_dim = (self.hw_intf.num_mocap_frames - 1) * (7 + 6) + 6   # pos, quat, lin vel, and ang vel + valve_pressure
        # -1 because base frame is removed since we transform into base frame before 

        # Hacky way to get the observation space for now
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(self.num_obs_dim,),
            dtype=np.float64,
        )

        # target position
        self.target_position = kwargs.get(
            "target_position", np.zeros(3)  # default end effector target position
        )

        self.reward_model = reward_model

        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING
        self.render_mode = kwargs.get("render_mode", "rgb_array")

        self.reset()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
        **kwargs
    ):
        super().reset(seed=seed)
        print("Resetting the environment...")

        self.eval_log_step = kwargs.get('step', self.eval_log_step)
        self._save_dir = os.path.join(self.log_dir, 'step' + '-' + str(self.eval_log_step))
        os.makedirs(self._save_dir, exist_ok=True)

        self.reset_sopra()
        self.action = np.zeros(self.action_space.shape)

        # get state
        state = self.get_state()
        reward = self.reward_model.predict(obs=state, action=self.action, next_obs=state)

        # reset current_step
        self.current_step = 0
        # reset time_tracker
        self.time_tracker = np.float64(0.0)
        self.prev_poses = None
        self.prev_timestamp = None
        # reset tip_history and frame_history
        self.tip_history = np.zeros((self.max_env_step, 3))
        self.state_history = np.zeros((self.max_env_step, self.num_obs_dim))
        self.action_history = np.zeros((self.max_env_step, self.action_space.shape[0]))
        self.pressure_history = np.zeros((self.max_env_step, self.action_space.shape[0]))

        self._target = self.target_position
        time.sleep(1)

        return state, {}
    
    def reset_sopra(self):
        pressures = np.zeros(self.action_space.shape)
        self.set_pressures(pressures)
        time.sleep(2)
        self.previous_action = np.zeros(self.action_space.shape)
        self.previous_pressure = np.zeros(self.action_space.shape)
        self.previous_commanded_pressure_norm = np.zeros(self.action_space.shape)


    def get_state(self):
        """
        Returns current state of the system to the controller.

        Returns
        -------
        numpy.ndarray
            1D (number_of_states) array containing data with 'float' type.
            Size of the states depends on the problem.
        """
        
        ################################################
        # Directly getting pose and twist from mocap thread
        ################################################
        # state = self.hw_intf.get_mocap_data()
        # while state is None and self.hw_intf.keep_running.is_set():
        #     time.sleep(0.1)
        #     print(f"Waiting for valid mocap data...")
        #     state = self.hw_intf.get_mocap_data()
        ################################################

        ################################################
        # Calculate twist at 10Hz
        ################################################
        poses_mocap, timestamp, mocap_segment_poses = self.hw_intf.get_mocap_data()
        # pressure_data = np.array(self.hw_intf.get_pressure_data())

        while poses_mocap is None and self.hw_intf.keep_running.is_set():
            time.sleep(0.1)
            print(f"Waiting for valid mocap data...")
            poses_mocap, timestamp, mocap_segment_poses = self.hw_intf.get_mocap_data()

        twists_mocap = self.hw_intf.compute_twists(mocap_segment_poses, timestamp,
                                                   self.prev_poses, self.prev_timestamp)
        self.prev_poses = deepcopy(mocap_segment_poses)
        self.prev_timestamp = timestamp
        state = np.concatenate([poses_mocap, twists_mocap, self.previous_commanded_pressure_norm.copy()])
        ################################################
        return np.array(state).flatten()

    def set_action(self, action):
        """
        Set the action to the system.

        Parameters
        ----------
        action : numpy.ndarray
            1D (number_of_actions) array containing data with 'float' type.
            Size of the actions depends on the problem.
        """
        # action is the delta of the air chamber pressure but normalized to [0, 1]
        delta_pressures = (action - 0.5) * self.action_delta_pressure_scale
        prev_commanded_pressures = self.previous_commanded_pressure_norm.copy() * self.max_pressure_safe
        clipped_pressures = np.clip(delta_pressures + prev_commanded_pressures, np.zeros_like(delta_pressures), self.max_pressure_safe)
        int_pressures = clipped_pressures.astype(int)
        # set pressure targets
        self.hw_intf.set_pressure_targets(int_pressures)
        return int_pressures
    
    def set_delta_pressure(self, delta_pressures):
        """
        Set the action to the system.

        Parameters
        ----------
        action : numpy.ndarray
            1D (number_of_actions) array containing data with 'float' type.
            Size of the actions depends on the problem.
        """
        # action is the delta of the air chamber pressure but normalized to [0, 1]
        prev_commanded_pressures = self.previous_commanded_pressure_norm.copy() * self.max_pressure_safe
        clipped_pressures = np.clip(delta_pressures + prev_commanded_pressures, np.zeros_like(delta_pressures), self.max_pressure_safe)
        int_pressures = clipped_pressures.astype(int)
        # set pressure targets
        self.hw_intf.set_pressure_targets(int_pressures)
        return int_pressures

    def set_pressures(self, pressures):
        """
        Set the action to the system.

        Parameters
        ----------
        action : numpy.ndarray
            1D (number_of_actions) array containing data with 'float' type.
            Size of the actions depends on the problem.
        """
        # change pressure to int type for valve input
        int_pressures = (np.rint(pressures)).astype(int)
        # set pressure targets
        self.hw_intf.set_pressure_targets(int_pressures)

    def step(self, action):
        for i in range(self.action_repeat):
            delta_pressure = (action - 0.5) * self.action_delta_pressure_scale / self.action_repeat

            pressures = self.set_delta_pressure(delta_pressure)
            self.previous_commanded_pressure_norm = pressures / self.max_pressure_safe
            self.sleep_rate.sleep()
        new_state = self.get_state()

        terminate = False
        truncate = False

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            self.tip_history[self.current_step] = new_state[:3]
            self.state_history[self.current_step] = new_state
            self.action_history[self.current_step] = action
            self.pressure_history[self.current_step] = pressures
        self.current_step += 1

        reward = self.reward_model.predict(obs=new_state, action=action, next_obs=new_state)

        reward = reward.astype(float).item()
        self.previous_action = action
        self.previous_pressure = new_state[26:]
        self._target = self.target_position

        return new_state, reward, terminate, truncate, {"ctime": self.time_tracker}

    def close(self):
        # self.stop()
        self.hw_intf.stop()
        super().close()

    # TODO: add render function
    def render(self):
        pass

    def plot_tip_history(self, img_name, dpi=300):
        fig, axs = plt.subplots(3, 1, sharex=True)
        histories = np.array(self.tip_history)
        y_labels = ['x[m]', 'y[m]', 'z[m]']
        for i in range(3):
            history = histories[:,i]
            axs[i].plot(range(len(history)), history)
            axs[i].set_ylabel(y_labels[i])

        for ax in axs.flat:
            ax.set(xlabel='Env Steps')     
        for ax in fig.get_axes():
            ax.label_outer()

        plt.tight_layout(w_pad=-10)

        # save figure and show
        plt.savefig(os.path.join(self._save_dir, img_name), bbox_inches="tight", dpi=dpi)
        plt.close() 

    def plot_frames_history(self, img_name='frames_history', dpi=300):
        # matplot marker position and target
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        frame1_history = np.array(self.state_history)[:,7:10]
        frame2_history = np.array(self.state_history)[:,:3]
        ax.plot(frame1_history[:,0], frame1_history[:,1], frame1_history[:,2], color = 'blue', alpha=0.3)
        ax.plot(frame2_history[:,0], frame2_history[:,1], frame2_history[:,2], color = 'blue', alpha=0.3)
        ax.scatter(frame1_history[-1,0], frame1_history[-1,1], frame1_history[-1,2], color = 'blue')
        ax.scatter(frame2_history[-1,0], frame2_history[-1,1], frame2_history[-1,2], color = 'blue')
        ax.scatter(self.target_position[0], self.target_position[1], self.target_position[2], color = 'red')
        ax.set_box_aspect([1,1,1])
        plt.savefig(os.path.join(self._save_dir, img_name), bbox_inches="tight", dpi=dpi)
        plt.close() 
        return 
        
    def post_processing(self, render_video=False, video_dir=None, filename_video=None, **kwargs):
        np.save(os.path.join(self._save_dir, 'target_position.npy'), np.array(self.target_position))
        np.save(os.path.join(self._save_dir, 'tip_history.npy'), np.array(self.tip_history))
        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            np.save(os.path.join(self._save_dir, 'state_history.npy'), np.array(self.state_history))
            np.save(os.path.join(self._save_dir, 'action_history.npy'), np.array(self.action_history))
            np.save(os.path.join(self._save_dir, 'pressure_history.npy'), np.array(self.pressure_history))
