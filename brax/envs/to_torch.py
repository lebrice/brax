import datetime
import time
from collections import abc
from functools import partial, singledispatch
from typing import Any, Mapping, Dict, TypeVar, Union, Optional

import numpy as np
import gym
import matplotlib.pyplot as plt
import torch
from jax._src import dlpack as jax_dlpack
from jaxlib.xla_extension import DeviceArray
from torch import Tensor
from torch.utils import dlpack as torch_dlpack
from brax.envs.wrappers import GymWrapper, VecGymWrapper


class JaxToTorchWrapper(gym.Wrapper):
    """Wrapper that converts Jax tensors to PyTorch tensors."""

    def __init__(self, env: Union[GymWrapper, VecGymWrapper], device: torch.device = None):
        """Creates a Wrapper around a `GymWrapper` or `VecGymWrapper` so it outputs
        PyTorch tensors.

        Parameters
        ----------
        env : Union[GymWrapper, VecGymWrapper]
            A `GymWrapper` or `VecGymWrapper` to wrap.
        device : torch.device, optional
            device on which to move the Tensors. Defaults to `None`, in which case the
            tensors will be on the same devices as their Jax equivalents.
        """
        if not hasattr(env, "reward_range"):
            env.reward_range = (-np.inf, np.inf)
        super().__init__(env)
        self.device: Optional[torch.device] = device

    def observation(self, observation):
        return jax_to_torch(observation, device=self.device)

    def action(self, action):
        return torch_to_jax(action)

    def reward(self, reward):
        return jax_to_torch(reward, device=self.device)

    def done(self, done: DeviceArray) -> Tensor:
        return jax_to_torch(done, device=self.device)

    def info(self, info: Dict) -> Dict:
        return jax_to_torch(info, device=self.device)

    def reset(self):
        obs = super().reset()
        return self.observation(obs)

    def step(self, action):
        action = self.action(action)
        obs, rewards, done, info = super().step(action)
        obs = self.observation(obs)
        rewards = self.reward(rewards)
        done = self.done(done)
        info = self.info(info)
        return obs, rewards, done, info


@singledispatch
def torch_to_jax(v: Any) -> Any:
    """Converts values to JAX tensors."""
    # Don't do anything by default, and when a handler is registered for this type of
    # value, it gets used to convert it to a torch tensor.
    # NOTE: The alternative would be to raise an error when an unsupported value is
    # encountered:
    # raise NotImplementedError(f"Don't know how to convert {v} to a Jax tensor")
    return v


@torch_to_jax.register(Tensor)
def _tensor_to_jax(v: Tensor) -> DeviceArray:
    tensor = torch_dlpack.to_dlpack(v)
    tensor = jax_dlpack.from_dlpack(tensor)
    return tensor


@torch_to_jax.register(abc.Mapping)
def _torch_dict_to_jax(v: Mapping) -> Mapping:
    return type(v)(**{k: torch_to_jax(value) for k, value in v.items()})


@singledispatch
def jax_to_torch(v: Any, device: torch.device = None) -> Any:
    """Convert JAX values to PyTorch Tensors."""
    raise NotImplementedError(f"Don't know how to convert {v} to a Torch tensor")


@jax_to_torch.register(DeviceArray)
def _devicearray_to_tensor(v: DeviceArray, device: torch.device = None) -> Tensor:
    dpack = jax_dlpack.to_dlpack(v)
    tensor = torch_dlpack.from_dlpack(dpack)
    if device:
        return tensor.to(device=device)
    return tensor


@jax_to_torch.register(abc.Mapping)
def _jax_dict_to_torch(
    v: Dict[str, Union[DeviceArray, Any]], device: torch.device = None
) -> Dict[str, Union[Tensor, Any]]:
    return type(v)(**{k: jax_to_torch(value, device=device) for k, value in v.items()})


@singledispatch
def torch_to_jax(v: Any) -> Any:
    """Converts values to JAX tensors."""
    # raise NotImplementedError(f"Don't know how to convert {v} to a Jax tensor")
    # NOTE: Could also not do anything as the default, and when a new handler is
    # registered, it gets used.
    return v


@torch_to_jax.register(Tensor)
def _tensor_to_jax(v: Tensor) -> DeviceArray:
    tensor = torch_dlpack.to_dlpack(v)
    tensor = jax_dlpack.from_dlpack(tensor)
    return tensor


@torch_to_jax.register(abc.Mapping)
def _torch_dict_to_jax(v: Mapping) -> Mapping:
    return type(v)(**{k: torch_to_jax(value) for k, value in v.items()})


@singledispatch
def jax_to_torch(v: Any, device: torch.device = None) -> Any:
    """Convert JAX values to PyTorch Tensors."""
    raise NotImplementedError(f"Don't know how to convert {v} to a Torch tensor")


@jax_to_torch.register(DeviceArray)
def _devicearray_to_tensor(v: DeviceArray, device: torch.device = None) -> Tensor:
    dpack = jax_dlpack.to_dlpack(v)
    tensor = torch_dlpack.from_dlpack(dpack)
    if device:
        return tensor.to(device=device)
    return tensor


@jax_to_torch.register(abc.Mapping)
def _jax_dict_to_torch(
    v: Dict[str, Union[DeviceArray, Any]], device: torch.device = None
) -> Dict[str, Union[Tensor, Any]]:
    return type(v)(**{k: jax_to_torch(value, device=device) for k, value in v.items()})

