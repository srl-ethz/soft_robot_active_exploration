# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

import os
import matplotlib.pyplot as plt

import numpy as np
import scipy

from typing import Optional
from gym import core, spaces
from gym.utils import seeding
from softae.models.environment_models.hasel_reward_model import HASELRewardModel

from gym.envs.mujoco import mujoco_env

import imageio
import mujoco
import yaml
from jinja2 import Environment, FileSystemLoader
import shutil


class HASELLegSimEnv(mujoco_env.MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        'render_fps': 500,
    }

    def __init__(
        self,
        # config: HelicoidArmConfig,
        seed: int = 0,
        COLLECT_DATA_FOR_POSTPROCESSING: bool = False,
        reward_model: HASELRewardModel = HASELRewardModel(),
        save_dir: str = './tmp',
        **kwargs
    ):
        
        # set seed
        self.np_random = None
        self.set_seed = seed
        self.seed(seed)
        
        # create folder for saving data
        self.log_dir = save_dir
        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # simulation parameters
        self.sim_dt = kwargs.get("integration_freq", 2.0e-3)
        self.time_step = self.sim_dt
        self.RL_update_interval = kwargs.get("sim_interval", 1.0e-2)
        self.num_steps_per_update = np.rint(
            self.RL_update_interval / self.sim_dt
        ).astype(int)
        self.frame_skip = self.num_steps_per_update
        self.eval_log_step = 0

        # Number of DC motors
        self.num_motors = 1
        # Number of HASEL units
        self.num_hasels = 4
        self.hasel_unit_force_rescale = np.array([1.0, 1.0, 1.0, 1.0]) * 40.0
        # Volatage scaling for the action
        self.volatage_scale_factor = kwargs.get("volatage_scale_factor", 6)  # action [0, 1] -> voltage [0, 6]kV 

        # Observation space
        num_obs_dim = 11
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(num_obs_dim,),
            dtype=np.float64,
        )

        # Load parameters
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.dir_path = dir_path
        params_load = scipy.io.loadmat("%s/assets/hasel/pele_leg/fittedParams.mat" % dir_path)
        self.params = params_load['fittedParams'][0]

        # reward function
        self.reward_model = reward_model

        # Target Control Mode
        self.control_mode = kwargs.get("control_mode", 'position_control')
        self.randomise_target = False
        self.randomise_velocity = False
        self.randomise_sine_traj = False
        self.target_velocity = 0.0
        self.target_position = 0.0
        self.qpos_target_start = None
        # target joint angle control
        if self.control_mode == 'position_control':
            if 'target_position' in kwargs:
                self.target_position = kwargs.get('target_position')
            else:
                self.randomise_target = True
                self.target_position = self.randomise_target_position()

        # target joint velocity control
        elif self.control_mode == 'velocity_control':
            if 'target_velocity' in kwargs:
                self.target_velocity = kwargs.get('target_velocity')
            else:
                self.randomise_velocity = True
                self.target_velocity = self.randomise_target_velocity()

        elif self.control_mode == 'trajectory_control':
            self.randomise_sine_traj = True
            self.target_position, self.target_velocity = self.randomise_target_sine_traj()

        else:
            raise NotImplementedError("Specified control mode is not yet implemented!")

        # initialize data collection and render
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING
        self.render_mode = kwargs.get("render_mode", 'rgb_array')
        self.camera_name = 'main_camera'
        self.frames = []

        # domain randomization
        self.domain_randomize = kwargs.get("domain_randomize", False)
        if self.domain_randomize:
            self.template_env = Environment(
                autoescape=False,
                loader=FileSystemLoader("%s/assets/hasel/pele_leg/" % dir_path),
                trim_blocks=False)
        else:
            self.init_env(dir_path)
        self.reset()


    def init_env(self, dir_path):
        if self.domain_randomize:
            scene_xml_path, _ = self.create_dr_xmls(dir_path)
            mujoco_env.MujocoEnv.__init__(
                self,
                scene_xml_path,
                1,
                self.observation_space,
                self.render_mode,
                camera_name=self.camera_name,
            )

        else:
            mujoco_env.MujocoEnv.__init__(
                self,
                "%s/assets/hasel/pele_leg/scene.xml" % dir_path,
                1,
                self.observation_space,
                self.render_mode,
                camera_name=self.camera_name,
            )

        # Action space (needs to be set after env init)
        min_action = 0.0
        max_action = 1.0
        self.action_space = spaces.Box(
            min_action, max_action, shape=(self.num_motors + self.num_hasels,), dtype=np.float64
        )


    def create_dr_xmls(self, dir_path):
        # xml paths in experiment dir
        scene_xml_path = os.path.join(dir_path, 'scene.xml')
        setup_xml_path = os.path.join(dir_path, 'leg_setup.xml')
        template_path = 'leg_setup_template.xml'
        # copy scene and leg_setup_template xml to the experiment dir
        shutil.copyfile("%s/assets/hasel/pele_leg/scene.xml" % self.dir_path,  # original scene xml file in env dir
                        scene_xml_path)
        # randomize domain params
        params = self.randomize_env_params()
        context = {'params': params}
        with open(setup_xml_path, 'w') as f:
            xml = self.template_env.get_template(template_path).render(context)
            f.write(xml)
        return scene_xml_path, setup_xml_path


    def randomize_env_params(self):  #TODO
        params = yaml.safe_load(open("%s/assets/hasel/pele_leg/leg_setup_params.yaml" % self.dir_path, 'r'))
        # randomize damping coefficients for knees
        damping = self.np_random.uniform(float(params['damping_range']['lb']), float(params['damping_range']['ub']), 2)
        params['damping']['knee1'], params['damping']['knee2'] = damping
        # randomize tendon slacks
        tendon_length0_delta = self.np_random.uniform(float(-params['tendon_len_range']['sub']), float(params['tendon_len_range']['add']), 2)
        params['tendon_len']['tendon1_min'], params['tendon_len']['tendon2_min'] = ((params['tendon_len']['tendon1_min'], 
                                                                                     params['tendon_len']['tendon2_min']) 
                                                                                     + tendon_length0_delta)
        # randomize mass
        mass_deltas = self.np_random.uniform(float(-params['mass_range']['sub']), float(params['mass_range']['add']), 5)
        for key, mass_delta in zip(params['mass'], mass_deltas):
            params['mass'][key] += mass_delta
        return params

    
    def reset_model(self):
        HARD_CODED_INIT_QPOS = np.array([0.0, 0.01432521, -0.34, -0.01536157, 0.03270195])
        HARD_CODED_INIT_QVEL = np.zeros(5)
        self.set_state(HARD_CODED_INIT_QPOS, HARD_CODED_INIT_QVEL)
        return None


    def reset(
        self,
        *,
        seed: Optional[int] = None,
        **kwargs
    ):
        self.seed(seed)

        # core.Env.reset(self, seed=seed)
        self.eval_log_step = kwargs.get('step', self.eval_log_step)
        self._save_dir = os.path.join(self.log_dir, 'step' + '-' + str(self.eval_log_step) + '-' + str(seed))
        os.makedirs(self._save_dir, exist_ok=True)

        # domain randomization
        if self.domain_randomize:
            self.init_env(self._save_dir)
            # if not collecting data for postprocessing, remove xml files
            if not self.COLLECT_DATA_FOR_POSTPROCESSING:
                try:
                    shutil.rmtree(self._save_dir)
                except OSError:
                    pass
        
        super().reset(seed=seed)

        # reset model to initial boom angle
        self.reset_model()


        # reset muscle model
        self.action = np.zeros(self.action_space.shape)

        # reset history list
        self.angle_history = []
        self.target_history = []
        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            self.state_history = []
            self.action_history = []
            self.ctrl_history = []
            self.tendon_history = []
            self.actuator_force_history = []
            self.frames = []

        # reset current_step
        self.current_step = 0
        # reset time_tracker
        self.time_tracker = np.float64(0.0)
        # reset previous_action
        self.previous_action = self.action

        # Get random target
        if self.randomise_target:
            self.target_position = self.randomise_target_position()
        if self.randomise_velocity:
            self.target_velocity = self.randomise_target_velocity()

        self._target_position, self._target_velocity = self.update_target_posvel(self.current_step, previous_obs=np.concatenate([self.data.qpos, self.data.qvel]))
        # self.set_render_target_position(np.array([self._target_position[0], -0.0675, self._target_position[1]]))

        # set state
        state = self._get_obs()
        self.previous_obs = state

        self.angle_history.append(state[0])
        self.target_history.append([self._target_position, self._target_velocity]) # TODO fix this hacky way to add initial target since target is always looking one step into the future
        self.target_history.append([self._target_position, self._target_velocity])
        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            self.frames.append(self.render())
            self.state_history.append(state)
            self.actuator_force_history.append(self.data.actuator_force.copy())

        return state, {}
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def randomise_target_position(self):
        target_range_low = 0
        target_range_high = 2 * np.pi
        target_position = self.np_random.uniform(target_range_low, target_range_high)
        return target_position
    
    def randomise_target_velocity(self):
        target_range_low = 0.5
        target_range_high = 1.5
        target_velocity = self.np_random.uniform(target_range_low, target_range_high)
        return target_velocity


    def update_target_position(self, time):
        return self.target_position

    def update_target_velocity(self, time):
        return self.target_velocity


    def update_target_posvel(self, time, previous_obs=None):
        return self.update_target_position(time), self.update_target_velocity(time)
    
    def generate_signal(self, signal_params, t, frequency, max_voltage, sampling_rate):
        signal_type = signal_params['type']
        period = 1 / frequency
        signal = np.zeros_like(t)
        if signal_type == 'sine':
            delay_in_fraction = signal_params['delay'] / 360
            for i in range(len(t)):
                if t[i] > period * delay_in_fraction:
                    sine_value = np.sin(2 * np.pi * frequency * (t[i] - period * delay_in_fraction))
                    signal[i] = max_voltage * sine_value if signal_params['reverse_polarity'] else max_voltage * np.abs(sine_value)
                else:
                    signal[i] = 0
        elif signal_type == 'step':
            duty_cycle = signal_params['duty_ratio']
            for i in range(len(t)):
                current_time = t[i] % period
                if current_time < period * duty_cycle:
                    if signal_params['reverse_polarity']:
                        signal[i] = max_voltage if (i // int(period * sampling_rate)) % 2 == 0 else -max_voltage
                    else:
                        signal[i] = max_voltage
                else:
                    signal[i] = 0
        return signal

    def get_knee_pos(self):
        return self.data.geom('knee_sphere').xpos.copy()

    def get_knee_height(self):
        return self.data.geom('knee_sphere').xpos.copy()[-1] 
    
    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos,  # yaw, pitch, roll, joint_hip, joint_knee
                self.data.qvel,
                np.array([self.get_knee_height()]),
                # self.data.actuator_force,
                # np.array([self._target_position, self._target_velocity]),
                # self.previous_action,
            ]
        )

    def muscle_force_model(self, V, L, volt):
        """
        Predict muscle force based on the model.
        
        Parameters:
        V (float): Velocity of the muscle.
        L (float): Length of the muscle.
        volt (float): Voltage applied to the muscle.
        
        Returns:
        float: Predicted muscle force.
        """
        # Unpack the parameters
        a1, a2, a3, a4, a5, a6, F_P = self.params  # Use self.params
        # print(f'volt: {volt}')

        # Protection against division by zero for Volt
        volt = max(volt, np.array([np.finfo(float).eps]))
        # print(f'volt: {volt}')

        # Inline computation of force components for V
        FV = max(volt * np.exp(-a1 * (V / volt) ** a2), 0)
        FV = np.array([0.0]) if not np.isfinite(FV) else FV
        # print(f'FV: {FV}')

        # Inline computation of force components for L
        FL = max(volt * np.exp(-a3 * (L / volt) ** a4), 0)
        FL = np.array([0.0]) if not np.isfinite(FL) else FL
        # print(f'FL: {FL}')

        # Calculate alpha(Volt)
        alpha_volt = a5 * volt ** a6
        # print(f'alpha_volt: {alpha_volt}')

        # Calculate and ensure non-negative, real predicted force
        F_pred = max(np.array([0.0]), np.real(alpha_volt * (FV * FL) + F_P))
        # print(f'F_pred: {F_pred}')

        return F_pred

    def control_logic(self, voltage_action):
        """
        Custom control logic for the muscle simulation.
        """
        # Volatage action
        voltage = voltage_action
        tendon_ctrl_force = np.zeros_like(voltage)

        for i in range(self.num_hasels):        
            # Calculate tendon displacement and velocity
            tendon_displacement = abs(self.model.tendon_length0[i] - self.data.ten_length[i])
            tendon_velocity = abs(self.data.ten_velocity[i])
            
            # Predict the muscle force
            force_predict = self.muscle_force_model(tendon_velocity, tendon_displacement, voltage[i])
            # Apply the predicted force to the control signal
            tendon_ctrl_force[i] = -force_predict * self.hasel_unit_force_rescale[i]

        return tendon_ctrl_force

    def set_action(self, action):
        """
        Set the action to the system.

        Parameters
        ----------
        action : numpy.ndarray
            1D (number_of_actions) array containing data with 'float' type.
            Size of the actions depends on the problem.
        """
        ctrl = np.zeros_like(action)
        self.motor_action = action[0] * 2 - 1
        ctrl[0] = self.motor_action
        # ctrl[0] = -0.34 # TODO hard-coded for motor control to keep roughly the same roll of the broom (-0.34 posvel)
        self.voltage_action = action[1:] * self.volatage_scale_factor
        ctrl[1:] = self.control_logic(self.voltage_action)
        # ctrl[1:] = np.zeros(self.num_hasels)
        return ctrl

    
    def set_render_target_position(self, position):
        """
        Set the 3D Cartesian position of the free joint, keeping the current orientation.
        
        Parameters:
        position (list or numpy.ndarray): 3D position (x, y, z) for the free joint.
        """
        if self.qpos_target_start is None:
            free_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "free_joint")
            if free_joint_id == -1:
                raise ValueError("Free joint not found in the model.")
            # Get the start index for this joint in qpos
            self.qpos_target_start = self.model.jnt_qposadr[free_joint_id]
        
        # Set the position for the free joint
        self.data.qpos[self.qpos_target_start:self.qpos_target_start+3] = position
        self.set_state(self.data.qpos, self.data.qvel)


    def step(self, action):
        self.action = action
        ctrl = self.set_action(self.action)
        # print(f"ctrl: {ctrl}")
        # print(f"actuator_force: {self.data.actuator_force}")
        
        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            self.action_history.append(self.action)
            self.ctrl_history.append(ctrl)
            self.tendon_history.append(np.concatenate([self.model.tendon_length0 - self.data.ten_length, self.data.ten_velocity]))

        # mujoco.mj_kinematics(self.model, self.data)
        for _ in range(self.num_steps_per_update):
            self.do_simulation(ctrl, 1)
        # self.set_render_target_position(np.array([self._target_position[0], -0.0675, self._target_position[1]]))

        self.current_step += 1

        # observe current state: current as sensed signal
        state = self._get_obs()

        terminate = False
        truncate = False

        self.angle_history.append(state[0])
        self.target_history.append([self._target_position, self._target_velocity])
        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            self.frames.append(self.render())
            self.state_history.append(state)
            self.actuator_force_history.append(self.data.actuator_force.copy())

        reward = self.reward_model.predict(obs=state, action=action, previous_obs=self.previous_obs)
        reward = reward.astype(float).item()

        self.previous_action = action
        self.previous_obs = state
        self._target_position, self._target_velocity = self.update_target_posvel(self.current_step)
        # self.set_render_target_position(np.array([self._target_position[0], -0.0675, self._target_position[1]]))
        # mujoco.mj_kinematics(self.model, self.data)

        return state, reward, terminate, truncate, {"ctime": self.time_tracker}

    def plot_history(self, img_name, dpi=300, num_subplot=10):
        fig, axs = plt.subplots(num_subplot, 1, sharex=True)
        histories = np.concatenate((np.array(self.state_history)[:-1, :5], np.array(self.action_history)), axis=1)

        # target trajectory
        axs[0].plot(range(len(histories[:,0])), np.array(self.target_history)[:-2, 0], color='r')

        y_labels = ['Yaw[rad]', 'Pitch[rad]', 'Roll[rad]', 'Joint Hip Angle[rad]', 'Joint Knee Angle[rad]', 'Action 0', 'Action 1', 'Action 2', 'Action 3', 'Action 4']
        for i in range(num_subplot):
            history = histories[:,i]
            axs[i].plot(range(len(history)), history)
            axs[i].set_ylabel(y_labels[i])

        for ax in axs.flat:
            ax.set(xlabel='Action Steps (100Hz)')     
        for ax in fig.get_axes():
            ax.label_outer()

        plt.tight_layout(w_pad=10)

        # save figure and show
        plt.savefig(os.path.join(self._save_dir, img_name), bbox_inches="tight", dpi=dpi)
        plt.close() 

    def plot_state_history(self, img_name, dpi=300):
        y_labels = ['Yaw[rad]', 'Pitch[rad]', 'Roll[rad]', 'Joint Hip Angle[rad]', 'Joint Knee Angle[rad]', 'Yaw Vel[rad/s]', 'Knee Height[m]']
        fig, axs = plt.subplots(len(y_labels), 1, sharex=True)
        histories = np.array(self.state_history)[:-1, :6]
        knee_height_histories = np.array(self.state_history)[:-1, 10].reshape(-1, 1)
        histories = np.concatenate((histories, knee_height_histories), axis=1)

        # target trajectory
        axs[0].plot(range(len(histories[:,0])), np.array(self.target_history)[:-2, 0], color='r')
        axs[5].plot(range(len(histories[:,0])), np.array(self.target_history)[:-2, 1], color='r')

        for i in range(len(y_labels)):
            history = histories[:,i]
            axs[i].plot(range(len(history)), history)
            axs[i].set_ylabel(y_labels[i])

        for ax in axs.flat:
            ax.set(xlabel='Action Steps (100Hz)')     
        for ax in fig.get_axes():
            ax.label_outer()

        plt.tight_layout(w_pad=10)

        # save figure and show
        plt.savefig(os.path.join(self._save_dir, img_name), bbox_inches="tight", dpi=dpi)
        plt.close() 

    
    def plot_all_tendon_history(self, list_tendon=None):
        histories = np.concatenate((np.array(self.tendon_history), np.array(self.action_history)[:,1:], np.array(self.ctrl_history)[:,1:], np.array(self.actuator_force_history)[1:, 1:]), axis=1)
        y_labels = ['Tendon Displacement[m]', 'Tendon Velocity[m/s]', 'Action', 'Commanded Force[N]', 'Measured Force[N]']
        
        if list_tendon is None:
            list_tendon = range(self.num_hasels)
        for tendon_id in list_tendon:
            self.plot_tendon_history(histories, y_labels, f'tendon_history{tendon_id}.png', tendon_id, num_tendons=self.num_hasels)


    def plot_tendon_history(self, histories, y_labels, img_name, tendon_id, num_tendons=2, dpi=300):
        fig, axs = plt.subplots(len(y_labels), 1, sharex=True)
        
        for i in range(len(y_labels)):
            history = histories[:,num_tendons*i+tendon_id]
            axs[i].plot(range(len(history)), history)
            axs[i].set_ylabel(y_labels[i])

        for ax in axs.flat:
            ax.set(xlabel='Action Steps (100Hz)')     
        for ax in fig.get_axes():
            ax.label_outer()

        plt.tight_layout(w_pad=10)
        # save figure and show
        plt.savefig(os.path.join(self._save_dir, img_name), bbox_inches="tight", dpi=dpi)
        plt.close() 


    def post_processing(self, **kwargs):
        np.save(os.path.join(self._save_dir, 'target.npy'), np.array(self.target_history))
        np.save(os.path.join(self._save_dir, 'angle_history.npy'), np.array(self.angle_history))
        # self.plot_angle_history('angle_history.png')
        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            np.save(os.path.join(self._save_dir, 'state_history.npy'), np.array(self.state_history))
            np.save(os.path.join(self._save_dir, 'action_history.npy'), np.array(self.action_history))
            np.save(os.path.join(self._save_dir, 'ctrl_history.npy'), np.array(self.ctrl_history))
            np.save(os.path.join(self._save_dir, 'tendon_history.npy'), np.array(self.tendon_history))
            np.save(os.path.join(self._save_dir, 'actuator_force_history.npy'), np.array(self.actuator_force_history))
            self.plot_history('history.png')
            self.plot_state_history('state_history.png')
            self.plot_all_tendon_history()
            video_name = kwargs.get('video_name', 'video.mp4')
            imageio.mimwrite(os.path.join(self._save_dir, video_name), self.frames, fps=np.rint(1/self.RL_update_interval))


    
    def close(self):
        super().close()
        
if __name__ == "__main__":
    from tqdm import trange
    target_kwargs_train = {
        'forward_velocity_weight': 1.0,
    }
    reward_model_forward = HASELRewardModel(forward_velocity_weight = target_kwargs_train['forward_velocity_weight'],
                                            ctrl_cost=0.0, vel_cost=0.0)
    print(reward_model_forward)
    
    env = HASELLegSimEnv(COLLECT_DATA_FOR_POSTPROCESSING=True, control_mode='velocity_control', 
                         domain_randomize=False,
                         save_dir = './tmp',
                         reward_model=reward_model_forward,
                         #  , target_position = np.array([0.2, -0.7])
                         )
    state, _ = env.reset()
    print(state)
    frames = []
    frame_rate = 100
    
    # General Parameters
    duration = 5  # Total duration in seconds
    frequency = 2  # Hz (for both signals)
    max_voltage = 6000  # in volts (6 kV)
    sampling_rate = 100  # samples per second (for smoothness)
    # Amplifier Configuration
    amplifier_config = {
        'AO0': {
            'amplifier': 'Trek 610E',
            'gain': 1000,
            'actuator': 'hip flexor'
        },
        'AO1': {
            'amplifier': 'Trek 20/20',
            'gain': 1000,
            'actuator': 'knee extensor'
        }
    }
    # Time and Voltage Units
    voltage_unit = 'kV'
    # Signal-Specific Parameters for AO0 and AO1
    ao_signals = {
        'AO0': {
            'type': 'sine',
            'delay': 60,
            'duty_ratio': None,
            'reverse_polarity': True
        },
        'AO1': {
            'type': 'step',
            'delay': None,
            'duty_ratio': 0.4,
            'reverse_polarity': True
        }
    }

    t = np.linspace(0, duration, int(sampling_rate * duration))
    signal_AO0 = env.generate_signal(ao_signals['AO0'], t, frequency, max_voltage, sampling_rate)
    signal_AO1 = env.generate_signal(ao_signals['AO1'], t, frequency, max_voltage, sampling_rate)
    signal_AO0_scaled = signal_AO0 / amplifier_config['AO0']['gain']
    signal_AO1_scaled = signal_AO1 / amplifier_config['AO1']['gain']

    total_reward = 0

    for i in trange(len(t)):
    # for i in trange(500):
        action = np.array([1.0 * env.volatage_scale_factor, signal_AO0_scaled[i], signal_AO1_scaled[i], signal_AO1_scaled[i], signal_AO0_scaled[i]]) / env.volatage_scale_factor  # F E E F
        # action = np.array([0, 0, 0, 0, 0])
        state, reward, terminate, truncate, time_dict = env.step(action)
        total_reward += reward
        image = env.render()
        frames.append(image)
        # joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "Boom3_joint")
        # boom3_qpos_adr = env.model.jnt_qposadr[joint_id]
        # env.data.qpos[boom3_qpos_adr] = -0.34
        # print(env.data.qpos)
        # print(env.data.qvel)
        # env.data.qpos = np.array([ 0.0,  0.01432521, -0.34,       -0.01536157,  0.03270195])
        # env.data.qvel = np.zeros(env.data.qvel.shape)
        # env.set_state(env.data.qpos, env.data.qvel)

    env.post_processing(video_name='100x_1.0_video.mp4')
    env.close()

    print("Total reward: {}".format(total_reward))
