# Getting started

## Install

XBArray needs Python 3.10 or newer.

```bash
pip install xbarray
```

This installs NumPy, `array-api-compat`, `array-api-extra` and `typing_extensions`.
PyTorch and JAX are optional. Install them through the extras, or install them yourself:

```bash
pip install "xbarray[torch]"      # adds torch
pip install "xbarray[jax]"        # adds jax
pip install "xbarray[torch,jax]"  # both
```

The extras install the default `torch` and `jax` packages from PyPI. For a CUDA build
of either library, follow the PyTorch or JAX install instructions, then install
`xbarray`.

## A first example

Each backend is a class in `xbarray.backends`. Call Array API functions on it the way
you would call them on `numpy` or `torch`:

=== "NumPy"

    ```python
    from xbarray.backends.numpy import NumpyComputeBackend as backend

    x = backend.arange(6, dtype=backend.float32)
    x = backend.reshape(x, (2, 3))
    print(backend.sum(x, axis=1))           # [ 3. 12.]
    print(type(x))                          # <class 'numpy.ndarray'>
    ```

=== "PyTorch"

    ```python
    from xbarray.backends.pytorch import PyTorchComputeBackend as backend

    x = backend.arange(6, dtype=backend.float32)
    x = backend.reshape(x, (2, 3))
    print(backend.sum(x, axis=1))           # tensor([ 3., 12.])
    print(type(x))                          # <class 'torch.Tensor'>
    ```

=== "JAX"

    ```python
    import jax
    from xbarray.backends.jax import JaxComputeBackend as backend

    x = backend.arange(6, dtype=backend.float32)
    x = backend.reshape(x, (2, 3))
    print(backend.sum(x, axis=1))           # [ 3. 12.]
    print(isinstance(x, jax.Array))         # True
    ```

The three snippets differ only in the import. The arrays you get back are plain
NumPy arrays, PyTorch tensors and JAX arrays. XBArray does not wrap them.

## The `ComputeBackend` object

`ComputeBackend` is the interface that all three backends implement. The concrete
backends are:

| Backend | Import | `str(backend)` |
| --- | --- | --- |
| `NumpyComputeBackend` | `from xbarray.backends.numpy import NumpyComputeBackend` | `numpy` |
| `PyTorchComputeBackend` | `from xbarray.backends.pytorch import PyTorchComputeBackend` | `pytorch` |
| `JaxComputeBackend` | `from xbarray.backends.jax import JaxComputeBackend` | `jax` |

A backend is a class that you use directly. You never create an instance:

```python
from xbarray.backends.numpy import NumpyComputeBackend

print(NumpyComputeBackend.simplified_name)  # numpy
NumpyComputeBackend()  # RuntimeError: NumpyComputeBackend is not meant to be instantiated. ...
```

Importing a backend module needs its library. `xbarray.backends.pytorch` imports
`torch`, and `xbarray.backends.jax` imports `jax`.

### Getting the backend for an array

If you already have an array, `get_backend_from_tensor` returns the matching backend:

```python
import numpy as np
from xbarray import get_backend_from_tensor

backend = get_backend_from_tensor(np.zeros(3))
print(backend)  # numpy
```

It recognises `numpy.ndarray`, `torch.Tensor` and `jax.Array`, and raises `ValueError`
for anything else. It also works on JAX tracers inside `jax.jit`, since those are
`jax.Array` instances.

### Type hints

`ComputeBackend` is generic over the array, device, dtype and random generator types.
Use it in annotations to say "any backend":

```python
from xbarray import ComputeBackend, BArrayType


def double(backend: ComputeBackend, x: BArrayType) -> BArrayType:
    return backend.multiply(x, 2)
```

See [Generic typing](concepts/typing.md) for the type variables.

## Next steps

- [Concepts](concepts/index.md): what a backend contains, dtypes and devices, random numbers.
- [Write backend-agnostic code](guides/backend-agnostic-code.md): patterns and pitfalls.
- [API reference](api/index.md): every function and signature.
