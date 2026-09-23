#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'flax==0.7.0',
    'chex==0.1.82',
    'distrax~=0.1.2',
    'gym>=0.26.0',
    'argparse-dataclass>=0.2.1',
    'dm_control',
    'trajax @ git+https://github.com/lenarttreven/trajax.git',
    'optax==0.1.7',
    'orbax-checkpoint==0.5.20',
    'jaxkern',
    'moviepy==1.0.3',
    'pandas',
    'wandb',
    'pyglet==2.0.15',
    'vapory==0.1.2'
]

extras = {'dev': ['seaborn', 'control>=0.9.2']}
setup(
    name='softae',
    version='0.0.1',
    packages=find_packages(),
    python_requires='>=3.8',
    include_package_data=True,
    install_requires=required,
    extras_require=extras
    )
