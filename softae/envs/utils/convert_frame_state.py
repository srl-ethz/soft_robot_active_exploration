import numpy as np
from copy import deepcopy
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr

class FramesStateReader():
    def __init__(
        self, dt=0.1, base_frame_idx=0, inverse_frames=True, local_coord=True):
        """Initializes a new reader."""
        self.prev_poses = None
        self.dt = dt
        self.inverse_frames = inverse_frames
        self.local_coord = local_coord  # convert all frames relative to the base frame
        self.base_frame_idx = base_frame_idx

    def compute_twists(self, current_poses, prev_poses, dt=None, curr_timestamp=None, prev_timestamp=None):
        twists = []
        if prev_poses is None:
            prev_poses = current_poses
            prev_timestamp = curr_timestamp
        for current_pose, prev_pose in zip(current_poses, prev_poses):
            motion_xyz = current_pose[:3] - prev_pose[:3]
            motion_rpy = pr.quaternion_gradient([prev_pose[3:], current_pose[3:]], dt=1)
            twists.append(np.concatenate([motion_xyz, motion_rpy[1]]))
        twists = np.concatenate(twists)

        if dt is None:
            if curr_timestamp is not None:
                dt = curr_timestamp - prev_timestamp
            else:
                return twists

        if dt > 0:
            twists/=dt
        else:
            twists = np.zeros_like(twists)
                
        return twists

    def compute_state_from_frames(self, frames, dt_real=None):
        poses = []
        
        X_W_armBase = np.array(frames[self.base_frame_idx])

        for frame_idx in [x for x in range(len(frames)) if x != self.base_frame_idx]:
            X_W_currentFrame = np.array(frames[frame_idx])

            if self.local_coord:
                # X_armBase_currentFrame = X_W_armBase.inverse() @ X_W_currentFrame
                X_armBase_currentFrame = pt.concat(pt.invert_transform(X_W_armBase), X_W_currentFrame)
            else:
                X_armBase_currentFrame = X_W_currentFrame

            pose = pt.pq_from_transform(X_armBase_currentFrame)

            if self.inverse_frames:
                poses.insert(0, pose)
            else:
                poses.append(pose)
                
        if not self.local_coord:
            poses.insert(self.base_frame_idx, pt.pq_from_transform(X_W_armBase))

        if self.prev_poses is None:
            self.prev_poses = deepcopy(poses)

        poses_concat = np.concatenate(poses)
        twists_concat = self.compute_twists(poses, self.prev_poses, dt=dt_real)
        self.prev_poses = deepcopy(poses)
        return np.concatenate([poses_concat, twists_concat])
