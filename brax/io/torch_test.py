from .torch import jax_to_torch, torch_to_jax
import jax
import jax.numpy as jnp
import torch
from torch import Tensor
import pytest
import numpy as np
from typing import Callable

class Quadratic:
    def __init__(self, a: float, b: float, c: float):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        return self.a * x**2 + self.b * x + self.c
    
    def grad(self, x):
        return 2 * self.a * x + self.b


@pytest.mark.parametrize("x", [0., 0.5, 1.0, 2.0])
@pytest.mark.parametrize("foo", [
    Quadratic(1,1,1),
    lambda v: v ** 2 + 123 * v ** v,
])
def test_wrapping_jax_function_with_grads(x: float, foo: Callable):
    x_np = np.array(x)
    
    # Recreate the same function, but ensure that it only work on Torch tensors:    
    def foo_torch(x: Tensor) -> Tensor:
        assert isinstance(x, Tensor)
        return foo(x)

    # Recreate the same function, but ensure that it only work on jax arrays:
    def foo_jax(x: jnp.ndarray) -> jnp.ndarray:
        assert isinstance(x, jnp.ndarray)
        return foo(x)
    
    # NOTE: Only works with float32 values!
    x_torch = torch.as_tensor(x_np, dtype=torch.float32).requires_grad_(True)
    y_torch = foo_torch(x_torch)
    y_torch.backward()

    grad_torch = x_torch.grad
    # Check That the torch gradient is what we'd expect, when we have the grad fn:
    if hasattr(foo, "grad"):
        assert grad_torch == foo.grad(x_torch)

    # Check that the pure jax gradient works:
    x_jax = jnp.array(x_np)
    y_jax = foo_jax(x_jax) 
    assert np.allclose(y_jax.tolist(), y_torch.tolist())
    grad_jax = jax.jacobian(foo_jax)(x_jax)
    assert np.allclose(grad_jax.tolist(), grad_torch.tolist())

    # Wrapping that function so it can work as part of a PyTorch "graph":
    wrapped_foo_jax = jax_to_torch(foo_jax)
    # Check that it behaves exactly the same as the PyTorch version above:
    wrapped_x = torch.as_tensor(x_np, dtype=torch.float32).requires_grad_(True)
    y_wrapped = wrapped_foo_jax(wrapped_x)
    assert np.allclose(y_wrapped.tolist(), y_torch.tolist())
    y_wrapped.backward()
    assert np.allclose(wrapped_x.grad.tolist(), x_torch.grad.tolist())
