# Learning soft robotic dynamics with active exploration

Code accompanying the manuscript *Learning soft robotic dynamics with active exploration*.

This code is released under the MIT License (see LICENSE file). 

---

## 1. System Requirements

### Software dependencies
- **Python:** Default 3.10 for the soft-arm, HASEL, and SoPrA experiments. 3.11 for the soft-fish experiments (depends on FishGym package version).
- **JAX:** 0.4.13 (CUDA 11 wheel for GPU)
- **Core packages:** `flax==0.7.0`, `chex==0.1.82`, `distrax~=0.1.2`, `optax==0.1.7`, `orbax-checkpoint==0.5.20`, `gym>=0.26.0`, `dm_control`, `argparse-dataclass>=0.2.1`, `trajax`, `jaxkern`, `moviepy==1.0.3`, `pandas`, `wandb`, `pyglet==2.0.15`, `vapory==0.1.2`, `scipy==1.12.0`, `numpy==1.24`, `pyelastica==0.3.2`
- **Optional:** [FishGym](https://github.com/fish-gym/gym-fish) (only required for the articulated soft-fish experiments, please contact [Manifold Dynamics Research](https://manifolddynamics.com/) for access)

### Operating systems tested
- Ubuntu 20.04 LTS
- Ubuntu 22.04 LTS

### Hardware
- **Recommended:** NVIDIA GPU with CUDA 11 support and ≥8 GB VRAM. Training and evaluation use JAX with CUDA acceleration.
- No other non-standard hardware required for simulation experiments.

---

## 2. Installation Guide

We provide two install options: a **[local install with conda](#option-a--local-install-with-conda)** for users who want to install and modify the environment directly on their host, and a **[pre-built Docker image](#option-b--docker-recommended-for-reviewers)** for easy, reproducible deployment. The two paths are independent, choose whichever fits your setup. 

### Option A — Local install with conda

```bash
cd soft_robot_active_exploration
conda create -n softae-310 python=3.10
conda activate softae-310
conda env config vars set PYTHONPATH=$(pwd)
conda deactivate && conda activate softae-310    # reload so PYTHONPATH takes effect
pip install -e .
```

`conda env config vars set PYTHONPATH=$(pwd)` is required because the launchers (e.g. `experiments/soft_arm_exp/launcher.py`) import `experiments.util` as a top-level package. Setting `PYTHONPATH` to the project root makes this import resolve from any working directory.

Install the JAX wheel for your platform. The default `jaxlib` from PyPI is CPU-only; install the CUDA wheel for GPU acceleration:
```bash
pip install jax==0.4.13
# GPU (Linux + CUDA 11):
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.13+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl
```

Install the remaining packages ([PyElastica](https://docs.cosseratrods.org/en/latest/) for the soft-arm physics simulation, plus pinned `scipy` / `numpy`):
```bash
pip install scipy==1.12.0 numpy==1.24 pyelastica==0.3.2
```

For the soft-fish experiments, additionally install [FishGym](https://github.com/fish-gym/gym-fish).

**Typical install time:** on a standard desktop with a working CUDA 11 driver and a reasonable internet connection, end-to-end installation takes under **10 minutes**.

### Option B — Docker (recommended for reviewers)

We ship a pre-built Docker image (`softae:cuda11`) that carries CUDA 11.8 + cuDNN 8.6 + Python 3.10. 

**Host prerequisites:** Docker, Docker Compose, and the NVIDIA Container Toolkit (for GPU passthrough).

**Step 1 — Obtain the image.** Choose either of the two paths below; both leave a local `softae:cuda11` image ready for Step 2.

*Option B1 — Pull the pre-built image from Docker Hub (recommended).* The image is published at [`hehuizhengethz/softae-image:1.0`](https://hub.docker.com/r/hehuizhengethz/softae-image):
```bash
docker pull hehuizhengethz/softae-image:1.0
docker tag hehuizhengethz/softae-image:1.0 softae:cuda11
```
The pull takes a few minutes depending on your connection (the image is ~14 GB). The `docker tag` step aliases it to the name the Compose file expects.

*Option B2 — Build from source.* Rebuild the image locally from the bundled `Dockerfile`. Takes 5–25 min:
```bash
cd soft_robot_active_exploration
docker compose build softae-demo
```

**Step 2 — Drop into an interactive shell inside the image.** The shell has the entire project bind-mounted at `/workspace`, so host edits show up immediately (no rebuild needed) and any files the launcher writes appear on the host under `./` automatically.
```bash
docker compose run --rm softae-shell
```
Now you're at a prompt inside the container — you can run any launcher (see § 3 Demo and § 4 Instructions for Use). Type `exit` to leave the container.

The Docker image only covers the soft-arm experiments. For the soft-fish experiments, install [FishGym](https://github.com/fish-gym/gym-fish) directly on the host.

---

## 3. Evaluation Demo

Three pretrained agents for the soft-arm reach task accompany this release. They are archived alongside the code snapshot on Zenodo at [10.5281/zenodo.22918959](https://doi.org/10.5281/zenodo.22918959). Download them and place them under `pretrained/` in the project root:

| File | Method |
|---|---|
| `pretrained/softae_softarm.pkl` | SoftAE (our proposed method) |
| `pretrained/hucrl_softarm.pkl` | H-UCRL baseline |
| `pretrained/random_softarm.pkl` | Random (uniform sampled action) baseline |

### Instructions to run evaluation
```bash
cd experiments/soft_arm_exp
python launcher_eval.py
```

By default this loads `pretrained/softae_softarm.pkl` and runs evaluation rollouts on the `reach` task (two reach targets in the soft-arm workspace). To evaluate a different method, edit the `PRETRAINED_CHECKPOINT` variable at the top of `launcher_eval.py` to point at `hucrl_softarm.pkl` or `random_softarm.pkl` — the launcher auto-detects the matching exploration strategy from the filename prefix.

### Expected output
The demo writes to a per-run log directory `experiments/soft_arm_exp/results/SoftAE-Eval-Reach/<seed><Method><timestamp>/` and prints to stdout:
- `Starting evaluation`
- `Evaluation done`
- `Reward task 1: <float>` — episode return for the first (near) reach target (e.g., ~195 for checkpoint `pretrained/softae_softarm.pkl`)
- `Reward task 2: <float>` — episode return for the second (far) reach target (e.g., ~176.05 for checkpoint `pretrained/softae_softarm.pkl`) 

Per-target rollout outputs (animated MP4, state/action/reward histories) land under `results/SoftAE-Eval-Reach/<seed><Method><timestamp>/save/task_1/` and `.../save/task_2/`.

### Expected run time
On a single NVIDIA GPU, the demo finishes in roughly **10 minutes** end-to-end .

---

## 4. Instructions for Use

### Running on your own task / robot
Each robotic system has a dedicated launcher under `experiments/`:

| Robot | Launcher | Task switch |
|---|---|---|
| Soft arm (Cosserat-rod sim) | `experiments/soft_arm_exp/launcher.py` | `TASK = 'reach' / 'reach_orien' / 'reach_unseen_obstacle' / 'reach_multi_unseen_obstacles' / 'reach_single_rod' / 'reach_multi_rods'` |
| Soft fish (FishGym fluid sim) | `experiments/soft_fish_exp/launcher.py` | edit hyperparams in `hyperparams.yaml` |
| HASEL leg | `experiments/hasel_exp/launcher_leg_icemto.py` | edit launcher constants |
| SoPrA arm | `experiments/sopra_exp/launcher_icemto.py` | edit launcher constants |

To run training from scratch for the simulated soft arm:
```bash
cd experiments/soft_arm_exp
python launcher.py
```
You can change `PROJECT_NAME`, `TASK`, and `OPTIMIZER_TYPE` (`iCemTO` or `SacOpt`) at the top of the launcher file. Hyperparameters are defined as nested dictionaries in the same file and are swept via `dict_permutations`. Training sweeps all five exploration strategies (`SoftAE`, `PETS`, `Mean`, `HUCRL`, `Random`) per seed.

To evaluate a trained agent on a new task, point `PRETRAINED_CHECKPOINT` in `launcher_eval.py` at your `.pkl` checkpoint and change `TASK` accordingly. The eval launcher auto-selects the matching exploration-strategy config based on the checkpoint filename prefix (`softae_…`, `hucrl_…`, or `random_…`).

### POV-Ray high-quality rendering for the simulated soft arm (optional)

The default soft-arm demo writes an MP4 via matplotlib. For higher-quality, photorealistic videos, we provide a POV-Ray pipeline (adapted from [Elastica+RL](https://www.cosseratrods.org/Elastica+RL/)) that ray-traces each frame of the soft arm, obstacles, and reach targets and stitches the PNGs into an MP4.

**Host prerequisites:** the `povray` binary and GNU `parallel` (`apt install povray parallel`). The bundled Docker image already includes both.

Generate the per-frame `.pov` files from a saved trajectory (e.g. `traj_arm_data.npz` written by the demo under `save/task_1/`):
```bash
cd softae/envs/utils/softarm_povray_render
python generate_POVray_files.py \
    --data_dir <output_dir> \
    --arm_and_target_data_path <path/to/traj_arm_data.npz> \
    [--obstacle_data_path <path/to/traj_obstacle_data.npz>]
```

Render the frames in parallel and stitch them into an MP4:
```bash
bash render_video.sh <output_dir> <num_frames> <fps> [num_workers]
```

The final video lands at `<output_dir>/output_video.mp4`.

---

## 5. Citation and license

This code is released under the MIT License (see `LICENSE`). 

Repository: https://github.com/srl-ethz/soft_robot_active_exploration

Archived snapshot: https://doi.org/10.5281/zenodo.22918959