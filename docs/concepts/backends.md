# Backends and the `ComputeBackend` interface

A backend is a class that bundles everything you need to work with one array library.
XBArray ships three:

| Class | Module | `simplified_name` | `ARRAY_TYPE` | `DEVICE_TYPE` | `DTYPE_TYPE` | `RNG_TYPE` |
| --- | --- | --- | --- | --- | --- | --- |
| `NumpyComputeBackend` | `xbarray.backends.numpy` | `"numpy"` | `numpy.ndarray` | `Any` | `numpy.dtype` | `numpy.random.Generator` |
| `PyTorchComputeBackend` | `xbarray.backends.pytorch` | `"pytorch"` | `torch.Tensor` | `torch.device` | `torch.dtype` | `torch.Generator` |
| `JaxComputeBackend` | `xbarray.backends.jax` | `"jax"` | `jax.Array` | `jax.Device` or `jax.sharding.Sharding` | `numpy.dtype` | `jax.Array` (a PRNG key) |

The `ARRAY_TYPE`, `DEVICE_TYPE`, `DTYPE_TYPE` and `RNG_TYPE` columns are class
attributes. You can use them for `isinstance` checks or in your own type hints.

Backends are used as classes. Their methods are plain functions, so there is no
`self` and nothing to instantiate. Calling `NumpyComputeBackend()` raises
`RuntimeError`. `str()` and `repr()` of a backend return its `simplified_name`.

## What a backend contains

A backend's attributes come from four places.

**1. The Array API namespace.** All functions and constants of the
[Array API Standard](https://data-apis.org/array-api/latest/API_specification/index.html):
array creation (`zeros`, `arange`, `asarray`, ...), element-wise math, `matmul`,
`where`, `sort`, `argsort`, the `linalg` and `fft` extensions, dtype objects such as
`float32`, and `__array_api_version__`. NumPy and PyTorch get these from
`array_api_compat.numpy` and `array_api_compat.torch`. JAX uses `jax.numpy`, or
`jax.experimental.array_api` on older JAX versions.

**2. array-api-compat helpers.** `device(x)`, `to_device(x, device)`, `size(x)`,
`array_namespace(*xs)`, `is_array_api_obj(x)` and the `is_*_array` checks.

**3. array-api-extra functions.** Every public function of
[array-api-extra](https://data-apis.org/array-api-extra/), bound to the backend's
namespace so you do not pass `xp=`. Examples are `at`, `isclose`, `pad`, `one_hot`,
`nunique`, `cov`, `kron`, `sinc`, `expand_dims` and `apply_where`. The exact set
depends on the installed array-api-extra version. Where a name exists in both the
namespace and array-api-extra, the array-api-extra version is the one on the backend.

**4. XBArray extras.** The parts the standard does not cover:

| Attribute | What it does | Page |
| --- | --- | --- |
| `simplified_name` | `"numpy"`, `"pytorch"` or `"jax"` | |
| `default_integer_dtype`, `default_index_dtype`, `default_floating_dtype`, `default_boolean_dtype` | Dtypes to use when you need one | [Devices and dtypes](devices-and-dtypes.md) |
| `dtype_is_real_integer`, `dtype_is_real_floating`, `dtype_is_boolean` | Dtype predicates | [Devices and dtypes](devices-and-dtypes.md) |
| `serialize_device`, `deserialize_device` | Device to and from a string | [Devices and dtypes](devices-and-dtypes.md) |
| `random` | Random number functions | [Random number generation](random.md) |
| `is_backendarray`, `from_numpy`, `to_numpy`, `from_other_backend`, `to_dlpack` | Conversion | [Converting between backends](interop.md) |
| `map_fn_over_arrays`, `map_fn_over_arrays_ex` | Apply a function to every array in nested data | [Converting between backends](interop.md#nested-data) |
| `pad_dim` | Pad one axis to a target size with a constant | below |
| `abbreviate_array` | Collapse an array whose elements are all equal to a Python scalar | API reference |

The NumPy backend also overrides `clip` with a faster version that keeps the Array API
behaviour: the result has the dtype of `x`, and NaN propagates.

## The `ComputeBackend` protocol

`xbarray.ComputeBackend` is a `typing.Protocol` that lists the attributes above. The
three backend classes satisfy it. Your code should accept a `ComputeBackend` and call
functions on it, rather than importing a specific backend:

```python
from xbarray import ComputeBackend


def center(backend: ComputeBackend, x):
    return x - backend.mean(x, axis=0, keepdims=True)
```

Random functions have their own protocol, `xbarray.RNGBackend`. It is the type of
`backend.random`.

## Choosing a backend

Import the class you want:

```python
from xbarray.backends.numpy import NumpyComputeBackend
from xbarray.backends.pytorch import PyTorchComputeBackend
from xbarray.backends.jax import JaxComputeBackend
```

Or find it from an existing array:

```python
import torch
from xbarray import get_backend_from_tensor

backend = get_backend_from_tensor(torch.ones(2))
print(backend)  # pytorch
```

Importing `xbarray` itself does not require PyTorch or JAX. `get_backend_from_tensor`
only checks the libraries that import successfully. Any error while importing one,
not only `ImportError`, makes that backend unavailable.

!!! note "Older import paths"
    `xbarray.numpy`, `xbarray.pytorch` and `xbarray.jax` re-export the backend
    classes for code written before version 0.0.1a9. New code should import from
    `xbarray.backends.*`. The PyTorch class was renamed from `PytorchComputeBackend`
    to `PyTorchComputeBackend` in 0.0.1a14.

## Updating arrays without in-place writes

JAX arrays are immutable, so `x[i] = v` fails on the JAX backend. Use `at` from
array-api-extra. By default it may write into `x` (NumPy, PyTorch) or return a new
array (JAX):

```python
from xbarray.backends.jax import JaxComputeBackend as backend

x = backend.zeros(4)
x = backend.at(x, 1).set(5.0)
x = backend.at(x, 2).add(1.0)
print(x)  # [0. 5. 1. 0.]
```

Assign the result back, as above, and do not reuse the old `x`. Pass `copy=True` to
`set`/`add`/... if the input must stay unchanged; see the
[array-api-extra docs](https://data-apis.org/array-api-extra/generated/array_api_extra.at.html).

## Padding one axis

`pad_dim(x, dim, target_size, value=0)` appends constant values along one axis until it
has `target_size` entries. It returns `x` unchanged if the axis is already that size,
and raises `ValueError` if it is larger.

```python
from xbarray.backends.numpy import NumpyComputeBackend as backend

x = backend.ones((2, 3))
print(backend.pad_dim(x, 1, 5, value=-1))
# [[ 1.  1.  1. -1. -1.]
#  [ 1.  1.  1. -1. -1.]]
```
