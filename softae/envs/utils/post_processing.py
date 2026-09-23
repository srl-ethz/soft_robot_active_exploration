# Copyright (c) 2025 Hehui Zheng, ETH Zurich.
# Licensed under the MIT License. See LICENSE file for details.

import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps
from tqdm import tqdm
from typing import Dict, Sequence


def _to_numpy(x):
    if hasattr(x, "block_until_ready"):
        x = x.block_until_ready()
    return np.asarray(x)


def _load_gt_arm_positions(gt_arm_data_path: str):
    if not os.path.exists(gt_arm_data_path):
        raise FileNotFoundError(f"Ground-truth file not found: {gt_arm_data_path}")

    data = np.load(gt_arm_data_path, allow_pickle=False)

    if "position_rod_observed" in data:
        gt_positions = data["position_rod_observed"]  # [T, 3, N]
        print("Loaded GT from key: position_rod_observed")
    elif "position_rod" in data:
        gt_positions = data["position_rod"]  # [T, 3, N]
        print("Loaded GT from key: position_rod")
    else:
        raise KeyError(
            f"{gt_arm_data_path} must contain 'position_rod_observed' or 'position_rod'. "
            f"Found keys: {list(data.keys())}"
        )

    if gt_positions.ndim != 3 or gt_positions.shape[1] != 3:
        raise ValueError(
            f"GT rod positions must have shape [T, 3, N], got {gt_positions.shape}"
        )

    return gt_positions


def _extract_predicted_points(log_mean_pred, n_points=5):
    """
    log_mean_pred: [T, 58]
    first 15 dims = xyz of 5 points
    returns: [T, 5, 3]
    """
    pred = _to_numpy(log_mean_pred)
    if pred.ndim != 2 or pred.shape[1] < 3 * n_points:
        raise ValueError(
            f"log_mean_pred must have shape [T, >= {3*n_points}], got {pred.shape}"
        )
    return pred[:, : 3 * n_points].reshape(pred.shape[0], n_points, 3)


def _load_log_mean_pred(log_mean_pred_path: str):
    if not os.path.exists(log_mean_pred_path):
        raise FileNotFoundError(f"log_mean_pred file not found: {log_mean_pred_path}")

    if log_mean_pred_path.endswith(".npz"):
        data = np.load(log_mean_pred_path, allow_pickle=False)
        if "log_mean_pred" in data:
            arr = data["log_mean_pred"]
            print("Loaded prediction from key: log_mean_pred")
        else:
            print(f"Keys in npz: {list(data.keys())}")
            raise KeyError(
                "NPZ does not contain key 'log_mean_pred'. "
                "Either rename the key or load the correct file."
            )
    else:
        arr = np.load(log_mean_pred_path, allow_pickle=False)
        print("Loaded prediction from raw npy file")

    return arr


def _build_cylinders_history_from_npz(obstacle_npz_path: str):
    """
    Expect obstacle_data.npz saved with:
        np.savez(path, obstacle_history=self.obstacle_histories, allow_pickle=True)

    Returns cylinders_history in the same style as your plotting code:
        list of dicts with keys including:
            - com
            - radius
            - height
            - direction
    """
    if obstacle_npz_path is None:
        return []

    if not os.path.exists(obstacle_npz_path):
        raise FileNotFoundError(f"Obstacle file not found: {obstacle_npz_path}")

    data = np.load(obstacle_npz_path, allow_pickle=True)
    if "obstacle_history" not in data:
        raise KeyError(
            f"{obstacle_npz_path} must contain key 'obstacle_history'. "
            f"Found keys: {list(data.keys())}"
        )

    obstacle_history = data["obstacle_history"]

    # Usually saved as an object array containing list[defaultdict]
    if isinstance(obstacle_history, np.ndarray) and obstacle_history.dtype == object:
        if obstacle_history.shape == ():
            obstacle_history = obstacle_history.item()
        else:
            obstacle_history = obstacle_history.tolist()

    cylinders_history = []
    for item in obstacle_history:
        cylinders_history.append(dict(item))

    print(f"Loaded {len(cylinders_history)} obstacle(s)")
    return cylinders_history


def plot_video_with_sphere(
    rods_history: Sequence[Dict],
    sphere_history: Sequence[Dict],
    video_name="video.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    import matplotlib.animation as animation

    sim_time = np.array(sphere_history[0]["time"])

    n_visualized_spheres = len(sphere_history)
    sphere_history_unpacker = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["position"][t_idx],
        sphere_history[sph_idx]["radius"][t_idx],
    )
    sphere_cmap = colormaps.get_cmap("Spectral")

    n_visualized_rods = len(rods_history)
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][t_idx]

    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.0, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    scaling_factor = (2 * 0.1) / max_axis_length
    scaling_factor *= 2.6e3

    fig = plt.figure(figsize=(8, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    time_idx = 0
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])

        _ = com_history_unpacker(rod_idx, time_idx)

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[2],
            inst_position[0],
            inst_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )

    sphere_artists = [None for _ in range(n_visualized_spheres)]
    for sphere_idx in range(n_visualized_spheres):
        sphere_position, sphere_radius = sphere_history_unpacker(sphere_idx, time_idx)
        sphere_artists[sphere_idx] = ax.scatter(
            sphere_position[2],
            sphere_position[0],
            sphere_position[1],
            s=np.pi * (scaling_factor * sphere_radius) ** 2,
            color=sphere_cmap(sphere_idx),
        )
        ax.add_artist(sphere_artists[sphere_idx])

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-v0_8-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):
                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    _ = com_history_unpacker(rod_idx, time_idx)

                    rod_scatters[rod_idx]._offsets3d = (
                        inst_position[2],
                        inst_position[0],
                        inst_position[1],
                    )

                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2 * 0.1
                    )

                for sphere_idx in range(n_visualized_spheres):
                    sphere_position, _ = sphere_history_unpacker(sphere_idx, time_idx)
                    sphere_artists[sphere_idx]._offsets3d = (
                        sphere_position[2],
                        sphere_position[0],
                        sphere_position[1],
                    )

                writer.grab_frame()

    plt.close(plt.gcf())


def plot_video_with_sphere_2D(
    rods_history: Sequence[Dict],
    sphere_history: Sequence[Dict],
    video_name="video_2D.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    import matplotlib.animation as animation
    from matplotlib.patches import Circle

    sim_time = np.array(sphere_history[0]["time"])

    n_visualized_spheres = len(sphere_history)
    sphere_history_unpacker = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["position"][t_idx],
        sphere_history[sph_idx]["radius"][t_idx],
    )
    sphere_cmap = plt.get_cmap("Spectral")

    n_visualized_rods = len(rods_history)
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][t_idx]

    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.05, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(ylim), difference(zlim))
    scaling_factor = (2 * 0.1) / max_axis_length
    scaling_factor *= 2.6e3

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(*ylim)
    ax.set_ylim(*zlim)

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1], "r", lw=0.5)[0]
        _ = com_history_unpacker(rod_idx, time_idx)

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[0],
            inst_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )

    sphere_artists = [None for _ in range(n_visualized_spheres)]
    for sphere_idx in range(n_visualized_spheres):
        sphere_position, sphere_radius = sphere_history_unpacker(sphere_idx, time_idx)
        sphere_artists[sphere_idx] = Circle(
            (sphere_position[0], sphere_position[1]),
            sphere_radius,
            color=sphere_cmap(sphere_idx),
        )
        ax.add_artist(sphere_artists[sphere_idx])

    ax.set_aspect("equal")

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-v0_8-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):
                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[1])

                    _ = com_history_unpacker(rod_idx, time_idx)

                    rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2
                    )

                for sphere_idx in range(n_visualized_spheres):
                    sphere_position, _ = sphere_history_unpacker(sphere_idx, time_idx)
                    sphere_artists[sphere_idx].center = (
                        sphere_position[0],
                        sphere_position[1],
                    )

                writer.grab_frame()

    plt.close(plt.gcf())


def plot_video_with_sphere_cylinder(
    rods_history: Sequence[Dict],
    cylinders_history: Sequence[Dict],
    sphere_history: Sequence[Dict],
    video_name="video.mp4",
    fps=60,
    step=1,
    **kwargs,
):
    import matplotlib.animation as animation

    plt.rcParams.update({"font.size": 22})

    n_visualized_cylinders = len(cylinders_history)
    cylinder_heights = [x["height"] for x in cylinders_history]
    cylinder_radii = [x["radius"] for x in cylinders_history]
    sim_time = np.array(rods_history[0]["time"])

    cylinder_cmap = plt.get_cmap("Spectral")

    n_visualized_rods = len(rods_history)

    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][t_idx]
    cylinder_history_unpacker = lambda cyl_idx, t_idx: (
        cylinders_history[cyl_idx]["com"][t_idx]
        - 0.5 * cylinder_heights[cyl_idx] * cylinders_history[cyl_idx]["direction"].reshape(3),
        cylinder_radii[cyl_idx],
        cylinder_heights[cyl_idx],
    )

    n_visualized_spheres = len(sphere_history)
    sphere_history_unpacker = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["position"][t_idx],
        sphere_history[sph_idx]["radius"][t_idx],
    )
    sphere_cmap = plt.get_cmap("Spectral")

    print("Plotting videos!")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    def make_data_for_cylinder_along_z(cstart, cradius, cheight):
        center_x, center_y = cstart[0], cstart[1]
        z = np.linspace(0, cheight, 3)
        theta = np.linspace(0, 2 * np.pi, 25)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = cradius * np.cos(theta_grid) + center_x
        y_grid = cradius * np.sin(theta_grid) + center_y
        z_grid += cstart[2]
        return [x_grid, y_grid, z_grid]

    xlim = kwargs.get("x_limits", (1.0, -1.0))
    ylim = kwargs.get("y_limits", (1.0, -1.0))
    zlim = kwargs.get("z_limits", (-0.05, 1.0))

    fig = plt.figure(1, figsize=(10, 8), frameon=True, dpi=dpi)
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.view_init(elev=20, azim=135)

    time_idx = 0
    cylinder_surfs = [None for _ in range(n_visualized_cylinders)]

    for cylinder_idx in range(n_visualized_cylinders):
        XC, YC, ZC = make_data_for_cylinder_along_z(
            *cylinder_history_unpacker(cylinder_idx, time_idx)
        )
        cylinder_surfs[cylinder_idx] = ax.plot_surface(
            ZC, XC, YC, color=cylinder_cmap(cylinder_idx), alpha=1.0
        )

    rod_scatters = [None for _ in range(n_visualized_rods)]
    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_scatters[rod_idx] = ax.scatter(
            inst_position[2],
            inst_position[0],
            inst_position[1],
            s=np.pi * inst_radius ** 2 * 1e4,
        )

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    scaling_factor = (2 * 0.1) / max_axis_length
    scaling_factor *= 2.6e3

    sphere_artists = [None for _ in range(n_visualized_spheres)]
    for sphere_idx in range(n_visualized_spheres):
        sphere_position, sphere_radius = sphere_history_unpacker(sphere_idx, time_idx)
        sphere_artists[sphere_idx] = ax.scatter(
            sphere_position[2],
            sphere_position[0],
            sphere_position[1],
            s=np.pi * (scaling_factor * sphere_radius) ** 2,
            color=sphere_cmap(sphere_idx),
        )
        ax.add_artist(sphere_artists[sphere_idx])

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-v0_8-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):
                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )
                    rod_scatters[rod_idx]._offsets3d = (
                        inst_position[2],
                        inst_position[0],
                        inst_position[1],
                    )
                    rod_scatters[rod_idx].set_sizes(np.pi * inst_radius ** 2 * 1e4)

                for cylinder_idx in range(n_visualized_cylinders):
                    XC, YC, ZC = make_data_for_cylinder_along_z(
                        *cylinder_history_unpacker(cylinder_idx, time_idx)
                    )
                    cylinder_surfs[cylinder_idx].remove()
                    cylinder_surfs[cylinder_idx] = ax.plot_surface(
                        ZC, XC, YC, color=cylinder_cmap(cylinder_idx), alpha=1.0
                    )

                for sphere_idx in range(n_visualized_spheres):
                    sphere_position, _ = sphere_history_unpacker(sphere_idx, time_idx)
                    sphere_artists[sphere_idx]._offsets3d = (
                        sphere_position[2],
                        sphere_position[0],
                        sphere_position[1],
                    )

                writer.grab_frame()

    plt.close(plt.gcf())


def plot_video_with_sphere_cylinder_2D(
    rods_history: Sequence[Dict],
    cylinders_history: Sequence[Dict],
    sphere_history: Sequence[Dict],
    video_name="video_2D.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    import matplotlib.animation as animation
    from matplotlib.patches import Circle

    sim_time = np.array(sphere_history[0]["time"])

    n_visualized_cylinders = len(cylinders_history)
    cylinder_heights = [x["height"] for x in cylinders_history]
    cylinder_radii = [x["radius"] for x in cylinders_history]
    cylinder_history_unpacker = lambda cyl_idx, t_idx: (
        cylinders_history[cyl_idx]["com"][t_idx],
        cylinder_radii[cyl_idx],
        1.0,
    )
    cylinder_cmap = plt.get_cmap("Spectral")

    def make_data_for_cylinder_along_y(cstart, cradius, cheight):
        x_grid = cstart[0]
        y_grid = cstart[1]
        z_grid = cstart[2]
        return [x_grid, y_grid, z_grid]

    n_visualized_spheres = len(sphere_history)
    sphere_history_unpacker = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["position"][t_idx],
        sphere_history[sph_idx]["radius"][t_idx],
    )
    sphere_cmap = plt.get_cmap("Spectral")

    n_visualized_rods = len(rods_history)
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][t_idx]

    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.05, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(zlim))
    scaling_factor = (2 * 0.1) / max_axis_length
    scaling_factor *= 2.6e3

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1], "r", lw=0.5)[0]
        _ = com_history_unpacker(rod_idx, time_idx)

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[0],
            inst_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )

    cylinder_artists = [None for _ in range(n_visualized_cylinders)]
    for cylinder_idx in range(n_visualized_cylinders):
        cylinder_position, cylinder_radius, _ = cylinder_history_unpacker(cylinder_idx, time_idx)
        cylinder_artists[cylinder_idx] = Circle(
            (cylinder_position[0], cylinder_position[1]),
            cylinder_radius,
            color=cylinder_cmap(cylinder_idx),
        )
        ax.add_artist(cylinder_artists[cylinder_idx])

    sphere_artists = [None for _ in range(n_visualized_spheres)]
    for sphere_idx in range(n_visualized_spheres):
        sphere_position, sphere_radius = sphere_history_unpacker(sphere_idx, time_idx)
        sphere_artists[sphere_idx] = Circle(
            (sphere_position[0], sphere_position[1]),
            sphere_radius,
            color=sphere_cmap(sphere_idx),
        )
        ax.add_artist(sphere_artists[sphere_idx])

    ax.set_aspect("equal")

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-v0_8-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):
                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[1])

                    _ = com_history_unpacker(rod_idx, time_idx)

                    rod_scatters[rod_idx].set_offsets(inst_position[[0, 1]].T)
                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2
                    )

                for cylinder_idx in range(n_visualized_cylinders):
                    XC, YC, _ = make_data_for_cylinder_along_y(
                        *cylinder_history_unpacker(cylinder_idx, time_idx)
                    )
                    cylinder_artists[cylinder_idx].center = (XC, YC)

                for sphere_idx in range(n_visualized_spheres):
                    sphere_position, _ = sphere_history_unpacker(sphere_idx, time_idx)
                    sphere_artists[sphere_idx].center = (
                        sphere_position[0],
                        sphere_position[1],
                    )

                writer.grab_frame()

    plt.close(plt.gcf())


def plot_video_with_gt_pred_cylinder_2D(
    cylinders_history_path: str = None,
    log_mean_pred_path: str = None,
    gt_arm_data_path: str = None,
    video_name="video_2D_gt_pred.mp4",
    fps=60,
    step=1,
    **kwargs,
):
    """
    Plot 2D video with:
      - ground-truth arm from gt_arm_data_path
      - predicted 5 points from log_mean_pred[:, :15]
      - optional cylinder obstacles
      - NO target sphere

    Alignment
    ---------
    The prediction is assumed to be a next-state prediction:
        pred_points_raw[t] corresponds to gt frame (t + 1)

    Therefore the displayed prediction is aligned as:
        frame 0 -> use sampled gt[0] as prediction
        frame 1 -> use pred_points_raw[0] vs gt[1]
        frame 2 -> use pred_points_raw[1] vs gt[2]
        ...

    Notes
    -----
    - Plots x-y plane.
    - GT rod is drawn as a line + small GT points.
    - Prediction is drawn as larger points.
    - If cylinders_history_path is None, cylinders are not plotted.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    from tqdm import tqdm

    log_mean_pred = _load_log_mean_pred(log_mean_pred_path)

    # Optional cylinder loading
    if cylinders_history_path is not None:
        cylinders_history = _build_cylinders_history_from_npz(cylinders_history_path)
    else:
        cylinders_history = []

    plt.rcParams.update({"font.size": 22})

    pred_color = kwargs.get("pred_color", "#d81b60")
    gt_line_color = kwargs.get("gt_line_color", "#1f77b4")
    gt_point_color = kwargs.get("gt_point_color", "#1f77b4")
    gt_line_width = kwargs.get("gt_line_width", 2.0)
    pred_marker_size = kwargs.get("pred_marker_size", 140)
    gt_marker_size = kwargs.get("gt_marker_size", 55)
    obstacle_alpha = kwargs.get("obstacle_alpha", 0.95)
    show_legend = kwargs.get("show_legend", False)

    gt_positions = _load_gt_arm_positions(gt_arm_data_path)       # [T_gt, 3, N]
    pred_points_raw = _extract_predicted_points(log_mean_pred, 5) # [T_pred, 5, 3]

    # -------------------------------------------------------------------------
    # Build 5-point GT samples so prediction and GT are aligned frame-by-frame
    # -------------------------------------------------------------------------
    gt_sample_idx = np.linspace(0, gt_positions.shape[-1] - 1, 5).round().astype(int)
    gt_points_5 = np.stack(
        [gt_positions[t][:, gt_sample_idx].T for t in range(gt_positions.shape[0])],
        axis=0,
    )  # [T_gt, 5, 3]

    # pred_points_raw[t] corresponds to gt_points_5[t+1]
    # So displayed pred_points should be:
    #   pred_points[0] = gt_points_5[0]
    #   pred_points[1] = pred_points_raw[0]
    #   pred_points[2] = pred_points_raw[1]
    #   ...
    if pred_points_raw.shape[0] >= gt_points_5.shape[0] - 1:
        pred_points = np.concatenate(
            [gt_points_5[0:1], pred_points_raw[: gt_points_5.shape[0] - 1]],
            axis=0,
        )
    else:
        # If prediction is shorter, only keep the frames we can align
        pred_points = np.concatenate(
            [gt_points_5[0:1], pred_points_raw],
            axis=0,
        )

    n_visualized_cylinders = len(cylinders_history)
    cylinder_cmap = plt.get_cmap("Spectral")

    if n_visualized_cylinders > 0:
        cylinder_radii = [x["radius"] for x in cylinders_history]

        cylinder_history_unpacker = lambda cyl_idx, t_idx: (
            cylinders_history[cyl_idx]["com"][t_idx],
            cylinder_radii[cyl_idx],
            1.0,
        )
    else:
        cylinder_history_unpacker = None

    def make_data_for_cylinder_along_y(cstart, cradius, cheight):
        x_grid = cstart[0]
        y_grid = cstart[1]
        z_grid = cstart[2]
        return [x_grid, y_grid, z_grid]

    # Number of renderable frames after alignment
    n_frames = min(
        gt_positions.shape[0],
        pred_points.shape[0],
        *(
            [len(cylinders_history[i]["com"]) for i in range(n_visualized_cylinders)]
            if n_visualized_cylinders > 0
            else [10**12]
        ),
    )

    if n_frames <= 0:
        raise ValueError("No frames available to render.")

    print("plot GT + aligned next-state prediction + optional obstacle 2D video")

    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="GT vs Prediction", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-0.05, 1.0))

    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")

    if kwargs.get("show_axes", True):
        ax.set_xlabel(kwargs.get("xlabel", "x"))
        ax.set_ylabel(kwargs.get("ylabel", "y"))
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    time_idx = 0
    gt_inst = gt_positions[time_idx]     # [3, N]
    gt_sampled = gt_points_5[time_idx]   # [5, 3]
    pred_inst = pred_points[time_idx]    # [5, 3]

    gt_line = ax.plot(
        gt_inst[0],
        gt_inst[1],
        color=gt_line_color,
        lw=gt_line_width,
        label="ground truth",
        zorder=2,
    )[0]

    gt_scatter = ax.scatter(
        gt_sampled[:, 0],
        gt_sampled[:, 1],
        s=gt_marker_size,
        color=gt_point_color,
        alpha=0.9,
        zorder=3,
    )

    pred_scatter = ax.scatter(
        pred_inst[:, 0],
        pred_inst[:, 1],
        s=pred_marker_size,
        color=pred_color,
        alpha=0.95,
        label="prediction",
        zorder=4,
    )

    cylinder_artists = []
    if n_visualized_cylinders > 0:
        for cylinder_idx in range(n_visualized_cylinders):
            cylinder_position, cylinder_radius, _ = cylinder_history_unpacker(cylinder_idx, time_idx)
            circle = Circle(
                (cylinder_position[0], cylinder_position[1]),
                cylinder_radius,
                color=cylinder_cmap(cylinder_idx),
                alpha=obstacle_alpha,
                zorder=1,
            )
            cylinder_artists.append(circle)
            ax.add_artist(circle)

    if show_legend:
        ax.legend(loc="best")

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-v0_8-whitegrid"):
            for time_idx in tqdm(range(0, n_frames, int(step))):
                gt_inst = gt_positions[time_idx]
                gt_sampled = gt_points_5[time_idx]
                pred_inst = pred_points[time_idx]

                gt_line.set_xdata(gt_inst[0])
                gt_line.set_ydata(gt_inst[1])

                gt_scatter.set_offsets(gt_sampled[:, :2])
                pred_scatter.set_offsets(pred_inst[:, :2])

                if n_visualized_cylinders > 0:
                    for cylinder_idx in range(n_visualized_cylinders):
                        XC, YC, _ = make_data_for_cylinder_along_y(
                            *cylinder_history_unpacker(cylinder_idx, time_idx)
                        )
                        cylinder_artists[cylinder_idx].center = (XC, YC)

                writer.grab_frame()

    plt.close(fig)


def plot_video_with_reward_field_2D(
    rods_history: Sequence[Dict],
    cylinders_history: Sequence[Dict],
    sphere_history: Sequence[Dict],
    video_name="video_reward_2D.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    import matplotlib.animation as animation
    from matplotlib.patches import Circle

    sim_time = np.array(sphere_history[0]["time"])

    n_visualized_cylinders = len(cylinders_history)
    cylinder_radii = [x["radius"] for x in cylinders_history]
    cylinder_history_unpacker = lambda cyl_idx, t_idx: (
        cylinders_history[cyl_idx]["com"][t_idx],
        cylinder_radii[cyl_idx],
        1.0,
    )
    cylinder_cmap = plt.get_cmap("Spectral")

    def make_data_for_cylinder_along_y(cstart, cradius, cheight):
        x_grid = cstart[0]
        y_grid = cstart[1]
        z_grid = cstart[2]
        return [x_grid, y_grid, z_grid]

    n_visualized_spheres = len(sphere_history)
    sphere_history_unpacker = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["position"][t_idx],
        sphere_history[sph_idx]["radius"][t_idx],
    )

    n_visualized_rods = len(rods_history)
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][t_idx]

    print("plot reward field 2D visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.05, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(zlim))
    scaling_factor = (2 * 0.1) / max_axis_length
    scaling_factor *= 2.6e3

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)

    reward_model = kwargs.get("reward_model", None)
    if reward_model is None:
        raise NotImplementedError("Reward model is not provided")

    x, z = np.meshgrid(np.linspace(xlim[0], xlim[1], 200), np.linspace(zlim[0], zlim[1], 200))
    dist = np.sqrt(
        (x - reward_model.target_position[0]) ** 2
        + (z - reward_model.target_position[1]) ** 2
        + reward_model.target_position[2] ** 2
    )
    dist_obstacle = np.sqrt(
        (x - reward_model.obstacle_pos[0]) ** 2
        + (z - reward_model.obstacle_pos[1]) ** 2
    )
    reward = (
        reward_model.distance_weight * reward_model.tol_rew(dist)
        - reward_model.obstacle_cost * reward_model.tol_obstacle_cutoff(dist_obstacle, dist)
    )

    reward_min = -max(abs(reward.min()), abs(reward.max()))
    reward_max = max(abs(reward.min()), abs(reward.max()))

    dx = x[0, 1] - x[0, 0]
    dz = z[1, 0] - z[0, 0]

    grad_z, grad_x = np.gradient(reward, dz, dx)
    grad_z = np.clip(grad_z, -1e6, 1e6)
    grad_x = np.clip(grad_x, -1e6, 1e6)

    grad_magnitude = np.sqrt(grad_x**2 + grad_z**2)
    grad_x = np.where(grad_magnitude > 0, grad_x / grad_magnitude, 0)
    grad_z = np.where(grad_magnitude > 0, grad_z / grad_magnitude, 0)

    ax.set_aspect("equal")

    from matplotlib.ticker import FormatStrFormatter
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    c = ax.pcolormesh(x, z, reward, cmap="RdBu_r", vmin=reward_min, vmax=reward_max)
    ax.set_title(f"Reward Field with Obstacle Weight {reward_model.obstacle_cost}")
    ax.axis([x.min(), x.max(), z.min(), z.max()])
    plt.colorbar(c, cax=cax)

    arrow_step = 10
    ax.quiver(
        x[::arrow_step, ::arrow_step],
        z[::arrow_step, ::arrow_step],
        grad_x[::arrow_step, ::arrow_step],
        grad_z[::arrow_step, ::arrow_step],
        color="black",
        scale=50,
    )

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1], "r", lw=0.5)[0]
        _ = com_history_unpacker(rod_idx, time_idx)
        rod_scatters[rod_idx] = ax.scatter(
            inst_position[0],
            inst_position[1],
            s=np.pi * (scaling_factor * inst_radius[-1]) ** 2,
        )

    cylinder_artists = [None for _ in range(n_visualized_cylinders)]
    for cylinder_idx in range(n_visualized_cylinders):
        cylinder_position, cylinder_radius, _ = cylinder_history_unpacker(cylinder_idx, time_idx)
        cylinder_artists[cylinder_idx] = Circle(
            (cylinder_position[0], cylinder_position[1]),
            cylinder_radius,
            color="black",
        )
        ax.add_artist(cylinder_artists[cylinder_idx])

    sphere_artists = [None for _ in range(n_visualized_spheres)]
    for sphere_idx in range(n_visualized_spheres):
        sphere_position, sphere_radius = sphere_history_unpacker(sphere_idx, time_idx)
        sphere_artists[sphere_idx] = Circle(
            (sphere_position[0], sphere_position[1]),
            sphere_radius,
            color="red",
        )
        ax.add_artist(sphere_artists[sphere_idx])

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-v0_8-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):
                for cylinder_idx in range(n_visualized_cylinders):
                    XC, YC, _ = make_data_for_cylinder_along_y(
                        *cylinder_history_unpacker(cylinder_idx, time_idx)
                    )
                    cylinder_artists[cylinder_idx].center = (XC, YC)

                for sphere_idx in range(n_visualized_spheres):
                    sphere_position, _ = sphere_history_unpacker(sphere_idx, time_idx)
                    sphere_artists[sphere_idx].center = (
                        sphere_position[0],
                        sphere_position[1],
                    )

                for rod_idx in range(n_visualized_rods):
                    inst_position, _ = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[1])

                    _ = com_history_unpacker(rod_idx, time_idx)

                    rod_scatters[rod_idx].set_offsets(inst_position[[0, 1]].T)

                writer.grab_frame()

    plt.close(plt.gcf())