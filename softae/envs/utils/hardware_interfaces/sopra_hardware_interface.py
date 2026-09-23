import time
import numpy as np
from copy import deepcopy
import threading

from pydrake.math import RigidTransform
from pydrake.common.eigen_geometry import AngleAxis

from softae.envs.utils.rater import Rate

from mobilerack_pybind_module import ValveController, QualisysClient

class SoPrAHWInterface():
    def __init__(
            self,
            max_pressure: int = 500,
            num_mocap_frames: int = 3,
            **kwargs,
            ):
        
        print("Start initializing hardware interface")
        # valves = [1, 2, 3, 4, 5, 6]
        valves = [1, 5, 3, 7, 6, 4]
        cameras = [0,1,3,4,5,6,7,8,9]
        self.num_chambers = len(valves)
        self.num_mocap_frames = num_mocap_frames
        self.vc = ValveController("192.168.0.100", valves, max_pressure)
        self.qc = QualisysClient(self.num_mocap_frames, cameras, "6D")

        time.sleep(1)  # hacky way to wait until data from qtm is received
        _, timestamp = self.qc.getData6D()
        self.vc.syncTimeStamp(timestamp//1000)  # sync the time to be that of QTM

        self.mocap_prev_poses = None
        self.mocap_data_lock = threading.RLock()        
        self.mocap_measurements = None
        self.mocap_prev_timestamp_us = None
        
        self.valve_targets_lock = threading.RLock()
        self.pressure_targets = None

        self.valve_data_lock = threading.RLock()
        self.pressure_measurements = None

        # invert the frame order, tip first -> tip position = state[:3]
        self.inverse_frames = kwargs.get("inverse_frames", True)

        self.keep_running = threading.Event()
        self.keep_running.set()

        # set rate for motion capture reading and valve pressure setting and reading
        self.mocap_read_rate = Rate(period_sec=1.0/200.0, identifier="mocap") # 200 Hz
        self.valve_set_rate = Rate(period_sec=1.0/200.0, identifier="valve") # 200 Hz
        self.valve_read_rate = Rate(period_sec=1.0/200.0, identifier="valve") # 200 Hz

        self.update_valve_data_thread = threading.Thread(target=self.update_valve_command)
        self.update_valve_data_thread.start()

        self.update_valve_read_thread = threading.Thread(target=self.update_valve_measurement)
        self.update_valve_read_thread.start()

        self.update_mocap_data_thread = None
        if self.qc is not None:
            self.update_mocap_data_thread = threading.Thread(target=self.update_mocap_observation)
            self.update_mocap_data_thread.start()
        
    def update_valve_measurement(self):
        while self.keep_running.is_set():
            with self.valve_data_lock:
                self.pressure_measurements = self.vc.getPressures()
            self.valve_read_rate.sleep()
    
    def update_valve_command(self):
        while self.keep_running.is_set():
            with self.valve_targets_lock:
                pressure_targets = self.pressure_targets.copy() if self.pressure_targets is not None else None
            if pressure_targets is not None:   # and (self.pressure_targets != self.pressure_prev_targets).any()
                self.vc.setPressures(pressure_targets)
                # self.pressure_prev_targets = pressure_targets.copy()
            self.valve_set_rate.sleep()

    def update_mocap_observation(self):
        while self.keep_running.is_set():
            # Read mocap data
            frames, timestamp_us = self.qc.getData6D()
            if timestamp_us != self.mocap_prev_timestamp_us:
                self.mocap_prev_timestamp_us = timestamp_us
                timestamp = timestamp_us / 1e6
                assert (
                    len(frames) == self.num_mocap_frames
                ), "Number of mocap frames does not match"

                mocap_segment_poses = []
                X_W_armBase = RigidTransform(frames[0])
                for segment_idx in range(self.num_mocap_frames):
                    X_W_currentFrame = RigidTransform(np.array(frames[segment_idx]))
                    X_armBase_currentFrame = X_W_armBase.inverse() @ X_W_currentFrame            

                    if segment_idx == 0:
                        assert np.linalg.norm(X_armBase_currentFrame.translation()) < 1e-3, "Arm base frame is not at origin"
                        assert np.allclose(X_armBase_currentFrame.rotation().matrix(), np.eye(3)), "Arm base frame is not identity"
                        continue  # skip arm base frame since it should be identity

                    if self.inverse_frames:
                        mocap_segment_poses.insert(0, X_armBase_currentFrame)
                    else:
                        mocap_segment_poses.append(X_armBase_currentFrame)

                if self.mocap_prev_poses is None:
                    self.mocap_prev_poses = deepcopy(mocap_segment_poses)
                    self.mocap_prev_timestamp = timestamp

                poses_mocap = [np.concatenate([pose.translation(), pose.rotation().ToQuaternion().wxyz()]) for pose in mocap_segment_poses]
                poses_mocap = np.concatenate(poses_mocap)

                # twists_mocap = self.compute_twists(mocap_segment_poses, timestamp, self.mocap_prev_poses, self.mocap_prev_timestamp)
                # self.mocap_prev_poses = deepcopy(mocap_segment_poses)
                # self.mocap_prev_timestamp = timestamp

                with self.mocap_data_lock:
                        # self.mocap_measurements = np.concatenate([poses_mocap, [timestamp]])
                        self.mocap_measurements = [poses_mocap, timestamp, mocap_segment_poses]

            self.mocap_read_rate.sleep()
    
    
    def update_mocap_observation_full(self):
        while self.keep_running.is_set():
            # Read mocap data
            frames, timestamp_us = self.qc.getData6D()
            if timestamp_us != self.mocap_prev_timestamp_us:
                self.mocap_prev_timestamp_us = timestamp_us
                timestamp = timestamp_us / 1e6
                assert (
                    len(frames) == self.num_mocap_frames
                ), "Number of mocap frames does not match"

                mocap_segment_poses = []
                X_W_armBase = RigidTransform(frames[0])
                for segment_idx in range(self.num_mocap_frames):
                    X_W_currentFrame = RigidTransform(np.array(frames[segment_idx]))
                    X_armBase_currentFrame = X_W_armBase.inverse() @ X_W_currentFrame            

                    if segment_idx == 0:
                        assert np.linalg.norm(X_armBase_currentFrame.translation()) < 1e-3, "Arm base frame is not at origin"
                        assert np.allclose(X_armBase_currentFrame.rotation().matrix(), np.eye(3)), "Arm base frame is not identity"
                        continue  # skip arm base frame since it should be identity

                    if self.inverse_frames:
                        mocap_segment_poses.insert(0, X_armBase_currentFrame)
                    else:
                        mocap_segment_poses.append(X_armBase_currentFrame)

                if self.mocap_prev_poses is None:
                    self.mocap_prev_poses = deepcopy(mocap_segment_poses)
                    self.mocap_prev_timestamp = timestamp

                poses_mocap = [np.concatenate([pose.translation(), pose.rotation().ToQuaternion().wxyz()]) for pose in mocap_segment_poses]
                poses_mocap = np.concatenate(poses_mocap)

                twists_mocap = self.compute_twists(mocap_segment_poses, timestamp, self.mocap_prev_poses, self.mocap_prev_timestamp)
                self.mocap_prev_poses = deepcopy(mocap_segment_poses)
                self.mocap_prev_timestamp = timestamp

                with self.mocap_data_lock:
                    self.mocap_measurements = np.concatenate([poses_mocap, twists_mocap])

            self.mocap_read_rate.sleep()
    
    def get_mocap_data(self):
        with self.mocap_data_lock:
            return self.mocap_measurements.copy()

    def set_pressure_targets(self, pressure_targets):
        with self.valve_targets_lock:
            self.pressure_targets = pressure_targets.copy()
        
    def get_pressure_data(self):
        with self.valve_data_lock:
            return self.pressure_measurements.copy()

    def compute_twists(self, current_poses, curr_timestamp, prev_poses, prev_timestamp):
        twists = []

        if prev_poses is None:
            prev_poses = current_poses
            prev_timestamp = curr_timestamp

        dt = curr_timestamp - prev_timestamp
        for current_pose, prev_pose in zip(current_poses, prev_poses):
            motion_xyz = current_pose.translation() - prev_pose.translation()
            quat_delta = (
                current_pose.rotation()
                .ToQuaternion()
                .multiply(prev_pose.rotation().ToQuaternion().conjugate())
            )
            angle_axis = AngleAxis(quat_delta)
            motion_rpy = angle_axis.axis() * angle_axis.angle()
            twists.append(np.concatenate([motion_xyz, motion_rpy]))
        
        twists = np.concatenate(twists)
        if dt > 0:
            twists/=dt
        else:
            twists = np.zeros_like(twists)
        return twists
    
    def stop(self):
        self.keep_running.clear()
        if self.update_mocap_data_thread is not None:
            self.update_mocap_data_thread.join()


if __name__ == "__main__":
    hw_intf = SoPrAHWInterface()

    keep_running = True
    while keep_running:
        try:
            # seg_idx = input("Enter segment number: ")
            # pressure_tgt = np.array([  0, 396, 451, 243, 248, 198], dtype=int)
            pressure_tgt = np.array([  0, 0,0, 450, 0, 0], dtype=int)
            # pressure_tgt[int(seg_idx)] = 500
            hw_intf.set_pressure_targets(pressure_tgt.tolist())
            print(hw_intf.get_pressure_data())
            time.sleep(1)
        except KeyboardInterrupt:
            keep_running = False

    input("Press enter to stop...")
    hw_intf.stop()