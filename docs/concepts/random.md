# Random number generation

XBArray has no global random state. You create a generator, pass it to every random
function, and get an updated generator back. This works the same way on all three
backends, including JAX, where random keys must be threaded through explicitly.

## Creating a generator

```python
from xbarray.backends.numpy import NumpyComputeBackend as backend

rng = backend.random.random_number_generator(42)
```

`random_number_generator(seed=None, *, device=None)` returns the library's own
generator type:

| Backend | Generator | Created with |
| --- | --- | --- |
| NumPy | `numpy.random.Generator` | `numpy.random.default_rng(seed)` |
| PyTorch | `torch.Generator` | `torch.Generator(device=device).manual_seed(seed)` |
| JAX | a typed PRNG key (`jax.Array`) | `jax.random.key(seed)` |

On PyTorch the generator belongs to a device. To draw on a GPU, create the generator
with `device="cuda"` and pass the same `device=` to the draws.

!!! note "Pass a seed"
    `seed=None` behaves differently on each backend:

    - NumPy seeds from fresh OS entropy.
    - PyTorch does not seed the new `torch.Generator`, so it starts from PyTorch's
      fixed default seed. Every unseeded generator produces the same numbers.
    - JAX picks a seed with `numpy.random.randint(65535)`. It depends on NumPy's
      global random state and has only 65535 possible values.

    Pass an explicit seed when you need reproducible or independent streams.

## Drawing numbers

Every draw takes the generator as the keyword `rng=` and returns a pair
`(rng, array)`. Always keep the returned generator and use it for the next draw:

=== "NumPy"

    ```python
    from xbarray.backends.numpy import NumpyComputeBackend as backend

    rng = backend.random.random_number_generator(0)
    rng, a = backend.random.random_uniform((2, 3), rng=rng, low=-1.0, high=1.0)
    rng, b = backend.random.random_discrete_uniform((4,), 0, 10, rng=rng)
    print(a.shape, b.dtype)  # (2, 3) int64
    ```

=== "PyTorch"

    ```python
    from xbarray.backends.pytorch import PyTorchComputeBackend as backend

    rng = backend.random.random_number_generator(0)
    rng, a = backend.random.random_uniform((2, 3), rng=rng, low=-1.0, high=1.0)
    rng, b = backend.random.random_discrete_uniform((4,), 0, 10, rng=rng)
    print(a.shape, b.dtype)  # torch.Size([2, 3]) torch.int64
    ```

=== "JAX"

    ```python
    from xbarray.backends.jax import JaxComputeBackend as backend

    rng = backend.random.random_number_generator(0)
    rng, a = backend.random.random_uniform((2, 3), rng=rng, low=-1.0, high=1.0)
    rng, b = backend.random.random_discrete_uniform((4,), 0, 10, rng=rng)
    print(a.shape, b.dtype)  # (2, 3) int32
    ```

NumPy and PyTorch generators are stateful. The returned `rng` is the same object you
passed in, advanced by the draw. JAX keys are immutable. The JAX backend splits the
key, uses one half for the draw and returns the other half as the new `rng`. If you
drop the returned key and reuse the old one on JAX, you get the same numbers again.
Writing `rng, x = ...` everywhere keeps code correct on all backends.

The same seed gives different numbers on different backends. Each library uses its
own algorithm.

## Available functions

All functions are on `backend.random`. `shape` can be an `int` or a tuple of ints.
Every function also accepts `dtype=` and `device=`.

| Function | Distribution |
| --- | --- |
| `random_discrete_uniform(shape, from_num, to_num, *, rng)` | Integers in `[from_num, to_num)` |
| `random_uniform(shape, *, rng, low=0.0, high=1.0)` | Floats in `[low, high)` |
| `random_normal(shape, *, rng, mean=0.0, std=1.0)` | Normal |
| `random_exponential(shape, *, rng, lambd=1.0)` | Exponential with rate `lambd` |
| `random_geometric(shape, *, p, rng)` | Geometric with success probability `p` |
| `random_permutation(n, *, rng)` | A random permutation of `0 .. n-1` |

!!! warning
    `random_permutation` currently raises `TypeError` on the JAX backend: it passes a
    `dtype=` argument that `jax.random.permutation` does not accept.

### Dtypes of the results

Without `dtype=`, each library picks its own default. Pass `dtype=` when the dtype
matters:

| Function | NumPy | PyTorch | JAX |
| --- | --- | --- | --- |
| `random_discrete_uniform` | `int64` | `int64` | `int32` |
| `random_uniform`, `random_normal`, `random_exponential` | `float64` | `float32` | `float32` |
| `random_geometric` | `int64` | `float32` | `int32` |

With an integer `dtype=`, `random_discrete_uniform` raises an error if the bounds do
not fit the dtype: `ValueError` on NumPy, `RuntimeError` on PyTorch. Asking for the
whole range of a dtype, for example `[0, 256)` for `uint8` images, is supported and
fast:

```python
from xbarray.backends.pytorch import PyTorchComputeBackend as backend

rng = backend.random.random_number_generator(0)
rng, image = backend.random.random_discrete_uniform(
    (64, 64, 3), 0, 256, rng=rng, dtype=backend.uint8
)
print(image.dtype, int(image.min()) >= 0, int(image.max()) <= 255)  # torch.uint8 True True
```

## Random numbers in your own functions

Take the generator as an argument and return the updated one with your result. The
functions in `xbarray.transformations` follow this rule, for example
`farthest_point_sampling` and `pointcloud_jitter`:

```python
from xbarray import ComputeBackend


def dropout_mask(backend: ComputeBackend, shape, p: float, rng):
    rng, u = backend.random.random_uniform(shape, rng=rng, dtype=backend.default_floating_dtype)
    return rng, u >= p


from xbarray.backends.numpy import NumpyComputeBackend as backend

rng = backend.random.random_number_generator(0)
rng, mask = dropout_mask(backend, (8,), 0.5, rng)
print(mask.dtype)  # bool
```

Under `jax.jit`, pass the key as a regular argument and return the new key. A key
created outside the function and closed over is a constant, and every call gets the
same numbers.
