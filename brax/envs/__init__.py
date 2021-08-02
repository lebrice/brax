# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Some example environments to help get started quickly with brax."""

import functools
from typing import Callable, overload, Union

import gym
import brax
from brax.envs import ant
from brax.envs import env
from brax.envs import fetch
from brax.envs import grasp
from brax.envs import halfcheetah
from brax.envs import humanoid
from brax.envs import reacher
from brax.envs import reacherangle
from brax.envs import ur5e
from brax.envs import wrappers
from brax.envs.wrappers import GymWrapper, VecGymWrapper

_envs = {
    "fetch": fetch.Fetch,
    "ant": ant.Ant,
    "grasp": grasp.Grasp,
    "halfcheetah": halfcheetah.Halfcheetah,
    "humanoid": humanoid.Humanoid,
    "ur5e": ur5e.Ur5e,
    "reacher": reacher.Reacher,
    "reacherangle": reacherangle.ReacherAngle,
}
State = env.State
Env = env.Env


def create(env_name: str, **kwargs) -> Env:
    """Creates an Env with a specified brax system."""
    return _envs[env_name](**kwargs)


def create_fn(env_name: str, **kwargs) -> Callable[..., Env]:
    """Returns a function that when called, creates an Env."""
    return functools.partial(create, env_name, **kwargs)


@overload
def create_env(env_name: str, **kwargs) -> GymWrapper:
    ...


@overload
def create_env(env_name: str, batch_size: int, **kwargs) -> VecGymWrapper:
    ...


def create_env(
    env_name: str, batch_size: int = None, **kwargs
) -> Union[GymWrapper, VecGymWrapper]:
    if batch_size is None:
        return create_gym_env(env_name=env_name, **kwargs)
    else:
        return create_gym_vector_env(env_name=env_name, batch_size=batch_size, **kwargs)


def create_gym_env(
    env_name: str, backend: str = "cpu", seed: int = 0, **kwargs
) -> gym.Env:
    """Creates a Gym Env with a specified brax system."""
    return GymWrapper(create(env_name, **kwargs), backend=backend, seed=seed)


def create_gym_vector_env(
    env_name: str, batch_size: int, backend: str = "cpu", seed: int = 0, **kwargs
) -> gym.vector.VectorEnv:
    """Creates a Gym Vector Env with a specified brax system."""
    return VecGymWrapper(
        create(env_name, batch_size=batch_size, **kwargs), backend=backend, seed=seed
    )


def create_baselines_vec_env(
    env_name: str, backend: str = "cpu", seed: int = 0, **kwargs
):
    """Creates a StableBaselines3 VecEnv with a specified brax system."""
    try:
        return wrappers.VecEnvWrapper(
            create(env_name, **kwargs), backend=backend, seed=seed
        )
    except ImportError:
        raise ImportError("StableBaselines3 is not available.")
