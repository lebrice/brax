from .torch import jax_to_torch, torch_to_jax
import jax
import jax.numpy as jnp
import torch
from torch import Tensor
import pytest
import numpy as np

@pytest.mark.parametrize("x", [0., 0.5, 1.0, 2.0])
def test_wrapping_jax_function_with_grads(x: float):
    x_np = np.array(x)
    a = 1
    b = 1
    c = 1

    def foo_torch(x: Tensor) -> Tensor:
        assert isinstance(x, Tensor)
        return a * x ** 2 + b * x + c

    
    # NOTE: Only works with float32 values!
    x_torch = torch.as_tensor(x_np, dtype=torch.float32).requires_grad_(True)
    y_torch = foo_torch(x_torch)
    y_torch.backward()

    # Check That the torch gradient is what we'd expect:
    grad_torch = x_torch.grad
    assert grad_torch == a * 2 * x + b


    # Recreate the same quadratic function, but have it only work on jax arrays:
    def foo_jax(x: jnp.ndarray) -> jnp.ndarray:
        assert isinstance(x, jnp.ndarray)
        return a * x ** 2 + b * x + c

    # Check that the pure jax gradient works:
    x_jax = jnp.array(x_np)
    y_jax = foo_jax(x_jax) 
    assert y_jax.tolist() == y_torch.tolist()
    grad_jax = jax.jacobian(foo_jax)(x_jax)
    assert grad_jax.tolist() == grad_torch.tolist()

    # Wrapping that function so it can work as part of a PyTorch "graph":
    wrapped_foo_jax = jax_to_torch(foo_jax)
    # Check that it behaves exactly the same as the PyTorch version above:
    wrapped_x = torch.as_tensor(x_np, dtype=torch.float32).requires_grad_(True)
    y_wrapped = wrapped_foo_jax(wrapped_x)
    assert y_wrapped == y_torch
    y_wrapped.backward()
    assert wrapped_x.grad == x_torch.grad
