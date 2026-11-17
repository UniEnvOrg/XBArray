# Write backend-agnostic code

This guide builds one function that runs on NumPy, PyTorch and JAX, and lists the
rules that keep it that way.

## The pattern

Take the backend as the first argument and call everything through it:

```python
from typing import Optional, Tuple
from xbarray import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType


def augment(
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    points: BArrayType,                    # (..., N, 3)
    rng: BRNGType,
    noise_std: float = 0.01,
    max_shift: float = 0.1,
) -> Tuple[BRNGType, BArrayType]:
    """Add per-point noise and one random shift per point cloud."""
    device = backend.device(points)
    rng, noise = backend.random.random_normal(
        points.shape, rng=rng, std=noise_std, dtype=points.dtype, device=device
    )
    shift_shape = (*points.shape[:-2], 1, 3)
    rng, shift = backend.random.random_uniform(
        shift_shape, rng=rng, low=-max_shift, high=max_shift, dtype=points.dtype, device=device
    )
    return rng, points + noise + shift
```

Run it on each backend:

=== "NumPy"

    ```python
    import numpy as np
    from xbarray.backends.numpy import NumpyComputeBackend as backend

    points = np.zeros((2, 100, 3), dtype=np.float32)
    rng = backend.random.random_number_generator(0)
    rng, out = augment(backend, points, rng)
    print(out.shape, out.dtype)  # (2, 100, 3) float32
    ```

=== "PyTorch"

    ```python
    import torch
    from xbarray.backends.pytorch import PyTorchComputeBackend as backend

    points = torch.zeros((2, 100, 3))
    rng = backend.random.random_number_generator(0)
    rng, out = augment(backend, points, rng)
    print(out.shape, out.dtype)  # torch.Size([2, 100, 3]) torch.float32
    ```

=== "JAX"

    ```python
    import jax
    import jax.numpy as jnp
    from xbarray.backends.jax import JaxComputeBackend as backend

    points = jnp.zeros((2, 100, 3))
    rng = backend.random.random_number_generator(0)
    rng, out = jax.jit(lambda p, k: augment(backend, p, k))(points, rng)
    print(out.shape, out.dtype)  # (2, 100, 3) float32
    ```

The JAX version runs under `jax.jit`. The backend is an ordinary Python object, so it
can be closed over. The array and the key are traced arguments.

## Binding the backend

If callers always use one backend, bind it with `functools.partial`. This is how
`xbarray.transformations.pointcloud.numpy` and its siblings are built from the generic
functions in `...pointcloud.base`:

```python
from functools import partial
from xbarray.backends.numpy import NumpyComputeBackend

augment_numpy = partial(augment, NumpyComputeBackend)
```

To pick the backend from the input instead, use `get_backend_from_tensor`:

```python
from xbarray import get_backend_from_tensor


def augment_any(points, rng, **kwargs):
    return augment(get_backend_from_tensor(points), points, rng, **kwargs)
```

## Rules

**Call functions on the backend, not methods on the array.** Array methods differ
between libraries: `x.astype(dtype)` exists on NumPy and JAX arrays but not on
PyTorch tensors. `backend.astype(x, dtype)` works everywhere. Operators such as `+`,
`*`, `@`, `<` and basic indexing like `x[..., :3]` are fine.

**Do not write into arrays.** `x[i] = v` and `x += y` on a slice fail on JAX. Build a
new array with `backend.where`, `backend.concat` or `backend.stack`, or use
`backend.at(x, i).set(v)` and keep the result.

**Take dtypes and devices from the inputs.** Create new arrays with
`dtype=x.dtype, device=backend.device(x)`. Otherwise a NumPy default (`float64`) and a
PyTorch default (`float32`) give different results, and PyTorch fails when arrays sit
on different devices.

**Use `default_index_dtype` for indices.** `take_along_axis` on PyTorch only accepts
`int64` indices, and `default_integer_dtype` is `int32` on PyTorch.

**Thread the generator.** Accept `rng`, use the one returned by each draw, and return
it. See [Random number generation](../concepts/random.md).

**Use only Array API names.** Library-specific functions such as `np.einsum` or
`torch.cdist` are not on the backend. Check the
[Array API specification](https://data-apis.org/array-api/latest/API_specification/index.html)
and [array-api-extra](https://data-apis.org/array-api-extra/) for what is.

**Keep host round trips out of hot paths.** `float(x)`, `int(x)` and `to_numpy` copy
to the host and synchronize with the GPU. Under `jax.jit` they fail on traced values.

## Testing on every backend

The XBArray test suite runs each test once per installed backend with a parametrised
pytest fixture. You can do the same in your own project:

```python
import importlib
import pytest

_BACKENDS = {
    "numpy": ("xbarray.backends.numpy", "NumpyComputeBackend", "numpy"),
    "pytorch": ("xbarray.backends.pytorch", "PyTorchComputeBackend", "torch"),
    "jax": ("xbarray.backends.jax", "JaxComputeBackend", "jax"),
}


@pytest.fixture(params=list(_BACKENDS))
def backend(request):
    module_name, cls_name, requirement = _BACKENDS[request.param]
    pytest.importorskip(requirement)
    return getattr(importlib.import_module(module_name), cls_name)


def test_augment_keeps_shape(backend):
    rng = backend.random.random_number_generator(0)
    points = backend.zeros((4, 10, 3), dtype=backend.float32)
    rng, out = augment(backend, points, rng)
    assert tuple(out.shape) == (4, 10, 3)
    assert out.dtype == backend.float32
```

Backends whose library is not installed are skipped.
