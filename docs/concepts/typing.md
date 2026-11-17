# Generic typing

Backend-agnostic code cannot name `numpy.ndarray` or `torch.Tensor` in its type hints.
XBArray provides type variables for the four things that change between backends.

| Type variable | Stands for | Bound |
| --- | --- | --- |
| `BArrayType` | The array type (`numpy.ndarray`, `torch.Tensor`, `jax.Array`) | `ArrayAPIArray` |
| `BDeviceType` | The device type (`torch.device`, `jax.Device`, ...) | `ArrayAPIDevice` |
| `BDtypeType` | The dtype type (`numpy.dtype`, `torch.dtype`) | `ArrayAPIDType` |
| `BRNGType` | The random generator type (`numpy.random.Generator`, `torch.Generator`, a JAX key) | none |

All four are importable from `xbarray`. `ComputeBackend` is generic over them, in that
order:

```python
from xbarray import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]
```

These hints are for readers and static type checkers. Nothing is checked at runtime.

## Annotating a function

Parametrise the backend and use the same type variables for the arrays and generators
it handles. This is how every function in `xbarray.transformations` is written:

```python
from typing import Tuple
from xbarray import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType


def add_noise(
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    x: BArrayType,
    std: float,
    rng: BRNGType,
) -> Tuple[BRNGType, BArrayType]:
    rng, noise = backend.random.random_normal(
        x.shape, rng=rng, std=std, dtype=x.dtype, device=backend.device(x)
    )
    return rng, x + noise
```

When you call `add_noise(PyTorchComputeBackend, ...)`, `BArrayType` is `torch.Tensor`
and `BRNGType` is `torch.Generator`.

For short helpers, a bare `ComputeBackend` annotation is fine.

## Annotating a class

Make the class `Generic` over the same variables and store the backend:

```python
from typing import Generic
from xbarray import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType


class Behavior(Generic[BArrayType, BDeviceType, BDtypeType, BRNGType]):
    def __init__(self, backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType]) -> None:
        self.backend = backend

    def create_array(self) -> BArrayType:
        return self.backend.zeros(5, dtype=self.backend.default_floating_dtype)
```

```python
from xbarray.backends.pytorch import PyTorchComputeBackend

behavior = Behavior(PyTorchComputeBackend)
print(behavior.create_array())  # tensor([0., 0., 0., 0., 0.])
```

## Other exported types

`xbarray` also exports the protocols the type variables are bound to:

- `ArrayAPIArray`, `ArrayAPIDevice`, `ArrayAPIDType`: protocols for an Array API array,
  device and dtype.
- `ArrayAPINamespace`: the Array API namespace plus the array-api-extra functions.
  `ComputeBackend` extends it.
- `RNGBackend`: the protocol of `backend.random`.
- `SupportsDLPack`: an object with `__dlpack__`, as returned by `to_dlpack`.
- `ArrayAPIGetIndex`, `ArrayAPISetIndex`: index types for `x[...]` and `at`.

These protocols live in the `array_api_typing` package, which is installed together
with `xbarray`.
