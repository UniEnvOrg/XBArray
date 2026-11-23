# Converting between backends

## To and from NumPy

Every backend can convert its arrays to and from NumPy:

```python
import numpy as np
from xbarray.backends.pytorch import PyTorchComputeBackend as backend

t = backend.from_numpy(np.arange(3.0), dtype=backend.float32)
print(t)                    # tensor([0., 1., 2.])
print(backend.to_numpy(t))  # [0. 1. 2.]
```

`from_numpy(data, *, dtype=None, device=None)`:

- **PyTorch** uses `torch.from_numpy`, then `.to(device=..., dtype=...)`. Without a
  dtype or device change the tensor shares memory with the NumPy array. NumPy
  `uint16`/`uint32`/`uint64` arrays become `int16`/`int32`/`int64` tensors.
- **JAX** uses `jax.numpy.asarray(data, dtype=dtype, device=device)`. Without 64-bit
  mode, `float64` input becomes `float32`.
- **NumPy** returns `data` as is. It ignores `dtype=` and `device=`.

`to_numpy(x)` copies to host memory where needed. PyTorch and JAX `bfloat16` arrays
are converted to `float32` first, because NumPy has no `bfloat16`.

To check whether a value is an array of a given backend, use `is_backendarray`:

```python
import numpy as np
from xbarray.backends.numpy import NumpyComputeBackend as backend

print(backend.is_backendarray(np.zeros(2)))  # True
print(backend.is_backendarray([0.0, 0.0]))   # False
```

## Between backends

`target.from_other_backend(source, data)` converts `data`, an array of the `source`
backend, into an array of the `target` backend:

```python
import torch
from xbarray.backends.pytorch import PyTorchComputeBackend
from xbarray.backends.jax import JaxComputeBackend

t = torch.arange(4, dtype=torch.float32)
j = JaxComputeBackend.from_other_backend(PyTorchComputeBackend, t)
print(j)  # [0. 1. 2. 3.]

back = PyTorchComputeBackend.from_other_backend(JaxComputeBackend, j)
print(back)  # tensor([0., 1., 2., 3.])
```

How the data moves depends on the target:

| Target | Method |
| --- | --- |
| PyTorch | From NumPy: `from_numpy(data)`. Otherwise `torch.from_dlpack(source.to_dlpack(data))`, cloned when the source is JAX |
| JAX | From host (CPU) data: `from_numpy(source.to_numpy(data))`, onto JAX's default device. Otherwise DLPack import of a copy made by the source library |
| NumPy | `source.to_numpy(data)` |

[DLPack](https://dmlc.github.io/dlpack/latest/) passes the buffer without a copy when
both libraries support the array's device. `to_dlpack(x)` returns `x` itself on NumPy
and JAX, and `x.detach()` on PyTorch, since NumPy arrays, PyTorch tensors and JAX
arrays all implement `__dlpack__`. JAX arrays are immutable, so a conversion into or out
of JAX copies the data: a later in-place write on the PyTorch or NumPy side cannot
change a JAX array. Between NumPy and PyTorch the buffer is shared, as with
`torch.from_numpy`.

If you do not know the source backend, find it first:

```python
import numpy as np
from xbarray import get_backend_from_tensor
from xbarray.backends.pytorch import PyTorchComputeBackend

x = np.ones(3)
t = PyTorchComputeBackend.from_other_backend(get_backend_from_tensor(x), x)
print(t)  # tensor([1., 1., 1.], dtype=torch.float64)
```

## Nested data

Observations, batches and model states are often dicts or lists of arrays.
`map_fn_over_arrays(data, func)` applies `func` to every array in such a structure and
returns a structure of the same shape:

```python
from xbarray.backends.numpy import NumpyComputeBackend as backend

batch = {
    "image": backend.ones((2, 4, 4)),
    "state": [backend.zeros(3), backend.ones(3)],
    "name": "episode-0",
}
doubled = backend.map_fn_over_arrays(batch, lambda x: x * 2)
print(doubled["state"][1], doubled["name"])  # [2. 2. 2.] episode-0
```

`map_fn_over_arrays_ex(data, *others, func=...)` walks `data` and the structures in
`others` together, and calls `func(array, *matching_values)`:

```python
from xbarray.backends.numpy import NumpyComputeBackend as backend

obs = {"pos": backend.asarray([1.0, 2.0]), "vel": backend.asarray([3.0, 4.0])}
scale = {"pos": 10.0, "vel": 0.5}
print(backend.map_fn_over_arrays_ex(obs, scale, func=lambda x, s: x * s))
# {'pos': array([10., 20.]), 'vel': array([1.5, 2. ])}
```

XBArray walks mappings, sequences (except `str` and `bytes`), namedtuples and
dataclasses on every backend. `func` is called only on arrays of this backend; other
values are returned unchanged. Each container is rebuilt with its original type
(falling back to a `dict` or `list` if that fails); dataclasses are rebuilt with
`dataclasses.replace`, so nested dataclasses keep their types. On JAX, other
registered pytree nodes are walked with `jax.tree.map`.
