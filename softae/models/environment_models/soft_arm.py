# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

import jax.numpy as jnp
import jax
from softae.models.reward_model import RewardModel
from functools import partial
from typing import Optional
import matplotlib.pyplot as plt

import numpy as np
import numba
from numba import njit
from elastica.external_forces import NoForces
from scipy.interpolate import make_interp_spline
from collections import defaultdict

class SoftArmRewardModel(RewardModel):
    """Get Soft Arm Reward."""
    
    def __init__(self, ctrl_cost: float = 0.0, vel_cost: float = 0.0, obstacle_cost: float = 0.0,
                 tol: float = 5e-2, 
                 distance_weight: float = 1.0, orientation_weieght: float = 0.5,
                 target_position: np.array = np.array([0.3, 0.95, 0.0]),
                 target_orientation: Optional[np.array] = None,
                 obstacle_pos: Optional[np.array] = None,
                 obstacle_bound: Optional[float] = 0.1,
                 obstacle_cost_clip_dist: Optional[float] = 1.0,
                 obs_state_points: Optional[int] = 4,):
        super().__init__()
        self.ctrl_cost = ctrl_cost
        self.vel_cost = vel_cost
        self.obstacle_cost = obstacle_cost
        self.obstacle_cost_clip_dist = obstacle_cost_clip_dist
        self.tol = tol
        self.obs_state_points = obs_state_points
        self.tip_pos_index = 3*self.obs_state_points
        self.tip_orien_index = 6*(self.obs_state_points+1) + 4*self.obs_state_points - 4
        self.distance_weight = distance_weight
        self.orientation_weieght = orientation_weieght
        self.target_position = target_position
        self.target_orientation = target_orientation
        self.obstacle_pos = obstacle_pos
        self.tol_rew = ToleranceReward(bounds=(0, self.tol), margin=20 * self.tol, value_at_margin=0.1, sigmoid='long_tail') 
        self.tol_cost = ToleranceReward(bounds=(0, obstacle_bound), margin=3 * obstacle_bound, value_at_margin=0.1, sigmoid='long_tail') 

    @partial(jax.jit, static_argnums=0)
    def tol_obstacle_cutoff(self, coll_dist, dist):
        return self.tol_cost(coll_dist)  # * (dist < self.obstacle_cost_clip_dist)

    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, next_obs=None, rng=None):
        # Calculate the position distance to the target
        tip_to_target = jnp.array(obs[self.tip_pos_index:self.tip_pos_index+3]) - jnp.array(self.target_position)
        dist = jnp.linalg.norm(tip_to_target)
        # Calculate the orientation distance to the target (if oriented)
        if self.target_orientation is None:
            orien_rew = 0
        else:
            orientation_dist = (
            1.0 - jnp.dot(jnp.array(obs[self.tip_orien_index:self.tip_orien_index+4]), jnp.array(self.target_orientation)) ** 2
            )
            orien_rew = 1-((orientation_dist) ** 2)
        # Calculate the control and velocity
        ctrl = jnp.sum(jnp.square(action), axis=-1)
        # 
        velocity = jnp.array(obs[3*(self.obs_state_points+1):3*(self.obs_state_points+1)+6*(self.obs_state_points+1)]) 
        vel = jnp.sum(jnp.square(velocity), axis=-1)
        obstacle = 0
        # Obstacle penalty
        if self.obstacle_pos is not None:
            # get tangent vector to the obstacle
            dx = obs[self.tip_pos_index] - self.obstacle_pos[0]
            dy = obs[self.tip_pos_index + 1] - self.obstacle_pos[1]
            r2 = dx ** 2 + dy ** 2 + 1e-6  # avoid div by 0
            vx_tangent = dy / r2
            vy_tangent = -dx / r2
            tangent_vector = jnp.array([vx_tangent, vy_tangent])
            tangent_vector = tangent_vector / (jnp.linalg.norm(tangent_vector) + 1e-6)  # normalize
            # get tip velocity vector
            tip_vel = velocity[-3:-1]
            tip_vel_norm = tip_vel / (jnp.linalg.norm(tip_vel) + 1e-6)
            tangent_alignment = jnp.dot(tip_vel_norm, tangent_vector)
            # Optional: Use sigmoid or just positive part
            tangent_reward = (1.0 + tangent_alignment) / 2.0  # normalized to [0, 1]

            tip_to_obstacle = jnp.array(obs[self.tip_pos_index:self.tip_pos_index+2] - self.obstacle_pos[:2])
            coll_dist = jnp.linalg.norm(tip_to_obstacle)
            # obstacle = (tip_to_obstacle[2]>= -0.1) * (tip_to_obstacle[0] <= 0.1) * 100
            # obstacle = self.tol_obstacle_cutoff(coll_dist, dist) - 2 * tangent_reward
            obstacle = - tangent_reward
            # obstacle = (coll_dist < 0.1) * obstacle

        # Weighted sum to total reward
        reward = (self.distance_weight * self.tol_rew(dist) 
                  + self.orientation_weieght * orien_rew 
                #   - self.ctrl_cost * ctrl 
                  - self.tol_rew(dist) * self.vel_cost * vel
                  - self.obstacle_cost * obstacle)
        # reward = - self.obstacle_cost * obstacle
        # reward = - dist
        return reward.reshape(-1).squeeze()
    
    def plot_reward_field_2D(
        self,
        img_name="reward_field.png",
        cylinder_position=[-0.4, 0.7, 0.0],
        cylinder_radius=0.1,
        sphere_position=[-0.7, 0.3, 0.0],
        sphere_radius=0.05,
        vis2D=True,
        **kwargs,
    ):
        plt.rcParams.update({"font.size": 20})

        # 2d case <always 2d case for now>
        import matplotlib.animation as animation
        from matplotlib.patches import Circle

        # plot settings
        dpi = kwargs.get("dpi", 300)
        xlim = kwargs.get("x_limits", (-1.0, 1.0))
        zlim = kwargs.get("z_limits", (-0.05, 1.0))
        difference = lambda x: x[1] - x[0]
        max_axis_length = max(difference(xlim), difference(zlim))
        # The scaling factor from physical space to matplotlib space
        scaling_factor = (2 * 0.1) / max_axis_length 
        scaling_factor *= 2.6e3  # Along one-axis

        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)

        # generate 2 2d grids for the x & y bounds
        x, z = np.meshgrid(np.linspace(xlim[0], xlim[1], 200), np.linspace(zlim[0], zlim[1], 200))
        dist = np.sqrt((x - self.target_position[0]) ** 2 + (z - self.target_position[1]) ** 2 + self.target_position[2] ** 2)
        dist_obstacle = np.sqrt((x - self.obstacle_pos[0]) ** 2 + (z - self.obstacle_pos[1]) ** 2)
        reward = self.distance_weight * self.tol_rew(dist) - self.obstacle_cost * self.tol_obstacle_cutoff(dist_obstacle, dist) 
    
        reward_min = -max(abs(reward.min()), abs(reward.max()))
        reward_max = max(abs(reward.min()), abs(reward.max()))

        # Compute grid spacing
        dx = x[0, 1] - x[0, 0]  # Uniform spacing along rows
        dz = z[1, 0] - z[0, 0]  # Uniform spacing along columns

        # Compute gradients
        try:
            grad_z, grad_x = np.gradient(reward, dz, dx)
        except Exception as e:
            print("Error in gradient computation:", e)
        grad_z = np.clip(grad_z, -1e6, 1e6)
        grad_x = np.clip(grad_x, -1e6, 1e6) 

        # Compute gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_z**2)

        grad_magnitude = np.clip(grad_magnitude, 1e-6, 1e6)

        # Avoid NaN by handling zero magnitude explicitly
        grad_x = np.where(grad_magnitude > 0, grad_x / grad_magnitude, 0)
        grad_z = np.where(grad_magnitude > 0, grad_z / grad_magnitude, 0)

        ax.set_aspect("equal")

        # Format axis ticks with one decimal digit
        from matplotlib.ticker import FormatStrFormatter
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # plot heatmap of reward
        c = ax.pcolormesh(x, z, reward, cmap='RdBu_r', vmin=reward_min, vmax=reward_max)
        ax.set_title('Reward Field with Obstacle Weight {}'.format(self.obstacle_cost))
        # set the limits of the plot to the limits of the data
        ax.axis([x.min(), x.max(), z.min(), z.max()])
        plt.colorbar(c, cax=cax)

        # plot cicles for target and obstacle
        cylinder_artist = Circle(
            (cylinder_position[0], cylinder_position[1]),
            cylinder_radius,
            color='black',
        )
        ax.add_artist(cylinder_artist)
        cylinder_artist.center = (cylinder_position[0], cylinder_position[1])

        sphere_artist = Circle(
            (sphere_position[0], sphere_position[1]),
            sphere_radius,
            color='black',
        )
        ax.add_artist(sphere_artist)
        sphere_artist.center = (sphere_position[0], sphere_position[1])

        # Plot gradient arrows
        step = 10  # Subsample arrows for clarity
        ax.quiver(x[::step, ::step], z[::step, ::step], grad_x[::step, ::step], grad_z[::step, ::step],
          color='black', scale=50)

        fig.savefig(img_name)
    

class SoftArmSparseRewardModel(RewardModel):
    """Get Soft Arm Reward."""
    
    def __init__(self, tol: float = 5e-2, 
                 target_position: np.array = np.array([0.3, 0.95, 0.0]),
                 obs_state_points: Optional[int] = 4,
                 **kwargs):
        super().__init__()
        self.tol = tol
        self.obs_state_points = obs_state_points
        self.tip_pos_index = 3*self.obs_state_points
        self.target_position = target_position
    
    # sparse reward
    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, next_obs=None, rng=None):
        # Calculate the position distance to the target
        tip_to_target = jnp.array(obs[self.tip_pos_index:self.tip_pos_index+3]) - jnp.array(self.target_position)
        dist = jnp.linalg.norm(tip_to_target)
        reward = (dist < self.tol) * 1
        return reward.reshape(-1).squeeze()

# class SoftArmObsTargetRewardModel(RewardModel):
#     """Get Soft Arm Reward."""
    
#     def __init__(self, ctrl_cost: float = 0.0, vel_cost: float = 0.0,
#                  obs_state_points: Optional[int] = 4,):
#         super().__init__()
#         self.ctrl_cost = ctrl_cost
#         self.vel_cost = vel_cost
#         self.obs_state_points = obs_state_points
#         self.tip_pos_index = 3*self.obs_state_points
#         self.tip_orien_index = 6*(self.obs_state_points+1) + 4*self.obs_state_points - 4
#         self.target_index = 6*(self.obs_state_points+1) + 7*self.obs_state_points

#     @partial(jax.jit, static_argnums=0)
#     def predict(self, obs, action, next_obs=None, rng=None):
#         # Calculate the position distance to the target
#         tip_to_target = jnp.array(obs[self.tip_pos_index:self.tip_pos_index+3]) - jnp.array(obs[self.target_index:self.target_index+3])
#         dist = jnp.linalg.norm(tip_to_target)
#         reward = - dist
#         return reward.reshape(-1).squeeze()


# class SoftArmTrajRewardModel(RewardModel):
#     """Get Soft Arm Reward."""
    
#     def __init__(self, ctrl_cost: float = 0.0, vel_cost: float = 0.0,
#                  obs_state_points: Optional[int] = 4,
#                  target_position: Optional[jnp.ndarray] = None):
#         super().__init__()
#         self.ctrl_cost = ctrl_cost
#         self.vel_cost = vel_cost
#         self.obs_state_points = obs_state_points
#         self.tip_pos_index = 3*self.obs_state_points
#         self.tip_orien_index = 6*(self.obs_state_points+1) + 4*self.obs_state_points - 4
#         self.target_position = target_position

#     def update_target():
#         pass

#     # @partial(jax.jit, static_argnums=0)
#     def predict(self, obs, action, next_obs=None, rng=None, target=None):
#         # Calculate the position distance to the target
#         if target is None:
#             target = self.target_position.copy()
#             print("target none")
#             print(target)
#         else:
#             self.target_position = target.copy()
#             print("target not None")
#             print(target)
#         tip_to_target = jnp.array(obs[self.tip_pos_index:self.tip_pos_index+3]) - jnp.array(target)
#         dist = jnp.linalg.norm(tip_to_target)
#         reward = - dist
#         return reward.reshape(-1).squeeze()

class SoftArm2DTrajRewardModel(RewardModel):
    """Get Soft Arm Reward."""
    
    def __init__(self, ctrl_cost: float = 0.0, vel_cost: float = 0.0,
                 target_position: np.array = np.array([0.3, 0.95, 0.0]),
                 obs_state_points: Optional[int] = 4,
                 **kwargs):
        super().__init__()
        self.ctrl_cost = ctrl_cost
        self.vel_cost = vel_cost
        self.obs_state_points = obs_state_points
        self.tip_pos_index = 3*self.obs_state_points
        self.tip_orien_index = 6*(self.obs_state_points+1) + 4*self.obs_state_points - 4
        self.target_position = target_position

    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, next_obs=None, rng=None):
        # Calculate the position distance to the 2D target
        tip_to_target = jnp.array([(obs[self.tip_pos_index] + obs[self.tip_pos_index+2])/jnp.sqrt(2), obs[self.tip_pos_index+1]]) - jnp.array(self.target_position[:2])
        dist = jnp.linalg.norm(tip_to_target)
        reward = - dist
        return reward.reshape(-1).squeeze()


class SoftArm3DTrajRewardModel(RewardModel):
    """Get Soft Arm Reward."""
    
    def __init__(self, ctrl_cost: float = 0.0, vel_cost: float = 0.0,
                 target_position: np.array = np.array([0.3, 0.95, 0.0]),
                 obs_state_points: Optional[int] = 4,
                 **kwargs):
        super().__init__()
        self.ctrl_cost = ctrl_cost
        self.vel_cost = vel_cost
        self.obs_state_points = obs_state_points
        self.tip_pos_index = 3*self.obs_state_points
        self.tip_orien_index = 6*(self.obs_state_points+1) + 4*self.obs_state_points - 4
        self.target_position = target_position

    @partial(jax.jit, static_argnums=0)
    def predict(self, obs, action, next_obs=None, rng=None):
        # Calculate the position distance to the 2D target
        tip_to_target = jnp.array(obs[self.tip_pos_index:self.tip_pos_index+2]) - jnp.array(self.target_position[:2])
        dist = jnp.linalg.norm(tip_to_target)
        velocity = jnp.array(obs[3*(self.obs_state_points+1):3*(self.obs_state_points+1)+6*(self.obs_state_points+1)]) 
        vel = jnp.sum(jnp.square(velocity), axis=-1)
        act = jnp.sum(jnp.square(action), axis=-1)
        reward = - dist - vel * self.vel_cost - act * self.ctrl_cost
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


class MuscleTorquesWithVaryingBetaSplines(NoForces):
    """
    This class compute the muscle torques using Beta spline.
    Points of beta spline can be changed through out the simulation, and
    every time it changes a new spline generated. Control algorithm has to
    select the spline control points. Location of control points on the arm
    is fixed and they are equidistant points.
    Attributes
    ----------
    direction : str
        Depending on the user input direction, computed torques are applied in the direction of d1, d2, or d3.
    points_array : numpy.ndarray or callable object
        This variable is a reference to points_func_array variable which can be a numpy.ndarray or callable object.
    base_length : float
        Initial length of the arm.
    muscle_torque_scale : float
        Scaling factor for beta spline muscle torques. Beta spline is non-dimensional and muscle_torque_scale scales it.
    torque_profile_recorder : defaultdict(list)
        This is a dictionary to store time-history of muscle torques and beta-spline.
    step_skip : int
        Determines the data collection step.
    counter : int
        Used to determine the current call step of this object.
    number_of_control_points : int
        Number of control points used in beta spline. Note that these are the control points in the middle and there
        are two more control points at the start and end of the rod, which are 0.
    points_cached : numpy.ndarray
        2D (2, number_of_control_points+2) array containing data with 'float' type.
        This array stores the location of control points in first row and in the second row it stores the values of
        control points selected at previous step. If control points are changed, points_cached updated.
    max_rate_of_change_of_control_points : float
        This limits the maximum change that can happen for control points in between two calls of this object.
    my_spline : object
        Stores the beta spline object generated by control points.
    """

    def __init__(
        self,
        base_length,
        number_of_control_points,
        points_func_array,
        muscle_torque_scale,
        direction,
        step_skip,
        max_rate_of_change_of_activation=0.01,
        **kwargs,
    ):
        """
        Parameters
        ----------
        base_length : float
            Initial length of the arm.
        number_of_control_points : int
            Number of control points used in beta spline. Note that these are the control points in the middle and there
            are two more control points at the start and end of the rod, which are 0.
        points_func_array  : numpy.ndarray
            2D (2, number_of_control_points+2) array containing data with 'float' type.
            This array stores the location of control points in first row and in the second row it stores the values of
            control points selected at previous step. If control points are changed, points_cached updated.
        muscle_torque_scale : float
            Scaling factor for beta spline muscle torques. Beta spline is non-dimensional and muscle_torque_scale
            scales it.
        direction  : str
            Depending on the user input direction, computed torques are applied in the "normal", "binormal", "tangent".
        step_skip  : int
            Determines the data collection step.
        max_rate_of_change_of_control_points : float
            This limits the maximum change that can happen for control points in between two calls of this object.
        **kwargs
            Arbitrary keyword arguments.
        """
        super(MuscleTorquesWithVaryingBetaSplines, self).__init__()

        if direction == str("normal"):
            self.direction = int(0)
        elif direction == str("binormal"):
            self.direction = int(1)
        elif direction == str("tangent"):
            self.direction = int(2)
        else:
            raise NameError(
                "Please type normal, binormal or tangent as muscle torque direction. Input should be string."
            )

        self.points_array = (
            points_func_array
            if hasattr(points_func_array, "__call__")
            else lambda time_v: points_func_array
        )

        self.base_length = base_length
        self.muscle_torque_scale = muscle_torque_scale

        self.torque_profile_recorder = kwargs.get("torque_profile_recorder", None)
        self.step_skip = step_skip
        self.counter = 0  # for recording data from the muscles
        self.number_of_control_points = number_of_control_points
        self.points_cached = np.zeros(
            (2, self.number_of_control_points + 2)
        )  # This caches the control points. Note that first and last control points are zero.
        self.points_cached[0, :] = np.linspace(
            0, self.base_length, self.number_of_control_points + 2
        )  # position of control points along the rod.
        self.points_cached[1, 1:-1] = np.zeros(
            self.number_of_control_points
        )  # initalize at a value that RL can not match

        # Max rate of change of activation determines, maximum change in activation
        # signal in one time-step.
        self.max_rate_of_change_of_activation = max_rate_of_change_of_activation

        # Purpose of this flag is to just generate spline even the control points are zero
        # so that code wont crash.
        self.initial_call_flag = 0

    def apply_torques(self, system, time: float = 0.0):

        # Check if RL algorithm changed the points we fit the spline at this time step
        # if points_array changed create a new spline. Using this approach we don't create a
        # spline every time step.
        # Make sure that first and last point y values are zero. Because we cannot generate a
        # torque at first and last nodes.
        # print('torque',self.max_rate_of_change_of_activation)

        if (
            not np.array_equal(self.points_cached[1, 1:-1], self.points_array(time))
            or self.initial_call_flag == 0
        ):
            self.initial_call_flag = 1

            # Apply filter to the activation signal, to prevent drastic changes in activation signal.
            self.filter_activation(
                self.points_cached[1, 1:-1],
                np.array((self.points_array(time))),
                self.max_rate_of_change_of_activation,
            )

            self.my_spline = make_interp_spline(
                self.points_cached[0], self.points_cached[1]
            )
            cumulative_lengths = np.cumsum(system.lengths)

            # Compute the muscle torque magnitude from the beta spline.
            self.torque_magnitude_cache = self.muscle_torque_scale * self.my_spline(
                cumulative_lengths
            )

        self.compute_muscle_torques(
            self.torque_magnitude_cache, self.direction, system.external_torques,
        )

        if self.counter % self.step_skip == 0:
            if self.torque_profile_recorder is not None:
                self.torque_profile_recorder["time"].append(time)

                self.torque_profile_recorder["torque_mag"].append(
                    self.torque_magnitude_cache.copy()
                )
                self.torque_profile_recorder["torque"].append(
                    system.external_torques.copy()
                )
                self.torque_profile_recorder["element_position"].append(
                    np.cumsum(system.lengths)
                )

        self.counter += 1

    @staticmethod
    @njit(cache=True)
    def compute_muscle_torques(torque_magnitude, direction, external_torques):
        """
        This Numba function updates external torques, it is used here because it updates faster than numpy version.
        Parameters
        ----------
        torque_magnitude : numpy.ndarray
            1D (n_elem,) array containing data with 'float' type.
            Computed muscle torque values.
        direction : int
            Determines which component of torque vector updated.
        external_torques : numpy.ndarray
            2D (3, n_elem) array containing data with 'float' type.
        Returns
        -------
        """

        blocksize = torque_magnitude.shape[0]
        for k in range(blocksize):
            external_torques[direction, k] += torque_magnitude[k]

    @staticmethod
    @numba.njit()
    def filter_activation(signal, input_signal, max_signal_rate_of_change):
        """
        Filters the input signal. If change in new signal (input signal) greater than
        previous signal (signal) then, increase for signal is max_signal_rate_of_change amount.
        Parameters
        ----------
        signal : numpy.ndarray
            1D (number_of_control_points,) array containing data with 'float' type.
        input_signal : numpy.ndarray
            1D (number_of_control_points,) array containing data with 'float' type.
        max_signal_rate_of_change : float
            This limits the maximum change that can happen between signal and input signal.
        Returns
        -------
        """
        signal_difference = input_signal - signal
        signal += np.sign(signal_difference) * np.minimum(
            max_signal_rate_of_change, np.abs(signal_difference)
        )

if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # bound = 0.1
    # reward = ToleranceReward(bounds=(0, bound), margin=3 * bound, value_at_margin=0.1, sigmoid='long_tail')
    # x = jnp.linspace(0, 1.0, 1000)
    # y = reward(x)
    # fig = plt.figure()
    # plt.plot(x, y)
    # fig.savefig('reward_sopra.png')
    # # plt.show()

    target_position = np.array([-0.8, 0.3, 0.0])
    obstacle_pos = np.array([-0.35, 0.55, 0.0])
    reward_model_unseen = SoftArmRewardModel(target_position=target_position,
                                             tol=0.05,
                                             obstacle_bound=0.1,
                                             obstacle_cost=1.0,
                                             obstacle_pos=obstacle_pos,
                                             obstacle_cost_clip_dist=0.2,)
    reward_model_unseen.plot_reward_field_2D(img_name="reward_field.png", 
                                             cylinder_position=obstacle_pos,
                                             sphere_position=target_position,)

