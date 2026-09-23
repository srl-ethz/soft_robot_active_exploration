# Reproducible image for the soft-arm reach demo.
# Base image provides CUDA 11.8 + cuDNN 8.6, matching the bundled
# jaxlib-0.4.13+cuda11.cudnn86 wheel. We use the `devel` variant (not
# `runtime`) because JAX's CUDA backend invokes `ptxas` / `nvlink` at runtime
# for JIT compilation; those binaries are only present in the devel image.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

# System dependencies:
# - povray: required by vapory (3D rendering of soft-arm rollouts)
# - ffmpeg: matplotlib animation backend writes MP4 video files via ffmpeg
# - libgl1 / libglib / libxext / libsm / libxrender: pyglet headless rendering
# - git: trajax dependency is fetched from GitHub
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        git \
        ca-certificates \
        povray \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libxext6 \
        libsm6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10 (matches the cp310 wheel used for jaxlib below).
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-distutils \
        python3.10-venv \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

WORKDIR /workspace

COPY . /workspace

# trajax is declared in setup.py with a git+ssh URL. Inside the container we
# fetch it over HTTPS instead so the build does not need SSH keys.
RUN sed -i 's|git+ssh://git@github.com/|git+https://github.com/|g' setup.py \
    && pip install -e . \
    && pip install jax==0.4.13 \
    && pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.13+cuda11.cudnn86-cp310-cp310-manylinux2014_x86_64.whl \
    && pip install \
        scipy==1.12.0 \
        numpy==1.24 \
        optax==0.1.7 \
        jaxkern \
        moviepy \
        pandas \
        wandb \
        pyelastica==0.3.2 \
        pyglet==2.0.15 \
        vapory==0.1.2

# Default to a shell so users can run launcher_eval.py or launcher.py interactively.
# docker-compose.yml overrides this to run the demo non-interactively.
CMD ["bash"]
