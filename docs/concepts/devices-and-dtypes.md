# Devices and dtypes

## Dtypes

Dtype objects are attributes of the backend, as in the Array API: `backend.float32`,
`backend.int64`, `backend.uint8`, `backend.bool`, and so on. They are the library's
own dtype objects, so `PyTorchComputeBackend.float32 is torch.float32`.

```python
from xbarray.backends.pytorch import PyTorchComputeBackend as backend

x = backend.zeros((2, 2), dtype=backend.float16)
print(x.dtype == backend.float16)  # True
```

### Default dtypes

When you need "a float" or "an index" without picking a width, use the backend's
defaults. They differ between backends:

| Attribute | NumPy | PyTorch | JAX |
| --- | --- | --- | --- |
| `default_floating_dtype` | `float` (gives `float64`) | `torch.float32` | `float` (gives `float32`) |
| `default_integer_dtype` | `int` (gives `int64`) | `torch.int32` | `int` (gives `int32`) |
| `default_index_dtype` | `int` (gives `int64`) | `torch.int64` | `int` (gives `int32`) |
| `default_boolean_dtype` | `bool` | `torch.bool` | `bool` |

NumPy and JAX use the Python types `float`, `int` and `bool`. Each library maps them
to its own default width. For JAX that is 32 bits unless you enable 64-bit mode
(`jax.config.update("jax_enable_x64", True)`).

Use `default_index_dtype` for arrays of indices. PyTorch needs `int64` indices in
functions such as `take_along_axis`, and `default_integer_dtype` on PyTorch is `int32`.

```python
from xbarray.backends.pytorch import PyTorchComputeBackend as backend

idx = backend.arange(3, dtype=backend.default_index_dtype)
print(idx.dtype)  # torch.int64
```

### Checking a dtype

Each backend has three predicates that accept that backend's dtypes:

```python
from xbarray.backends.numpy import NumpyComputeBackend as backend

x = backend.asarray([1, 2, 3])
print(backend.dtype_is_real_integer(x.dtype))   # True
print(backend.dtype_is_real_floating(x.dtype))  # False
print(backend.dtype_is_boolean(x.dtype))        # False
```

On PyTorch and JAX, `dtype_is_real_floating` is also true for `bfloat16`. The
standard `backend.isdtype(dtype, kind)` is available too.

## Devices

Array creation functions take a `device=` keyword, and `backend.device(x)` returns the
device of an array. To move an array, use `backend.to_device(x, device)`.

=== "NumPy"

    ```python
    from xbarray.backends.numpy import NumpyComputeBackend as backend

    x = backend.zeros(3)
    print(backend.device(x))  # cpu
    ```

    NumPy arrays always live on the CPU. Array creation functions accept
    `device=None` or `device="cpu"` and raise `ValueError` for anything else.
    XBArray's own functions (`from_numpy`, the random functions) ignore `device=`
    on NumPy.

=== "PyTorch"

    ```python
    import torch
    from xbarray.backends.pytorch import PyTorchComputeBackend as backend

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = backend.zeros(3, device=device)
    print(backend.device(x))
    ```

=== "JAX"

    ```python
    import jax
    from xbarray.backends.jax import JaxComputeBackend as backend

    device = jax.devices()[0]
    x = backend.zeros(3, device=device)
    print(backend.device(x))
    ```

To create an array on the same device as another one, pass
`device=backend.device(other)`. This is the usual pattern in backend-agnostic code.

### Serializing a device

`serialize_device` turns a device into a string and `deserialize_device` turns it
back. Use them to store a device in a config file or send it between processes.

```python
import torch
from xbarray.backends.pytorch import PyTorchComputeBackend as backend

s = backend.serialize_device(torch.device("cuda", 1))
print(s)                               # cuda:1
print(backend.deserialize_device(s))   # cuda:1
```

| Backend | `serialize_device` output | Notes |
| --- | --- | --- |
| NumPy | always `None` | `deserialize_device` also returns `None` |
| PyTorch | `"<type>:<index>"`, for example `"cuda:1"`, or `"<type>"` | A device without an index (`torch.device("cuda")`, the current CUDA device) is serialized without one. A string device is returned as is |
| JAX | `"<platform>:<id>"`, for example `"cpu:0"` | Deserializing looks the device up in `jax.devices(platform)`; if the id is not found it falls back to the first device of that platform |

Both functions map `None` to `None`.
