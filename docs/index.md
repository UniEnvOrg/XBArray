# XBArray

XBArray is a Python array library that runs the same code on NumPy, PyTorch and JAX.
It is built on the [Python Array API Standard](https://data-apis.org/array-api/latest/).

You write a function once against a *compute backend* object. You pass in
`NumpyComputeBackend`, `PyTorchComputeBackend` or `JaxComputeBackend`, and the
function works on that library's arrays: `numpy.ndarray`, `torch.Tensor` or `jax.Array`.

Each backend gives you:

- The Array API namespace for its library (`zeros`, `matmul`, `where`, `linalg`, ...),
  through [array-api-compat](https://github.com/data-apis/array-api-compat).
- The helper functions from [array-api-extra](https://github.com/data-apis/array-api-extra),
  such as `at` for out-of-place index updates.
- A few things the standard does not cover: explicit random number generators,
  conversion to and from NumPy and other backends, device serialization, and
  mapping a function over nested containers of arrays.

On top of the backends, `xbarray.transformations` has rotation conversions and point
cloud operations written in the same backend-agnostic style.

## A short example

```python
from xbarray import ComputeBackend
from xbarray.backends.numpy import NumpyComputeBackend
from xbarray.backends.pytorch import PyTorchComputeBackend

def normalize(backend: ComputeBackend, x):
    """Scale each row of x to unit length."""
    norms = backend.linalg.vector_norm(x, axis=-1, keepdims=True)
    return x / backend.clip(norms, min=1e-12)

for backend in (NumpyComputeBackend, PyTorchComputeBackend):
    x = backend.asarray([[3.0, 4.0], [0.0, 2.0]])
    print(backend, normalize(backend, x))
```

```text
numpy [[0.6 0.8]
 [0.  1. ]]
pytorch tensor([[0.6000, 0.8000],
        [0.0000, 1.0000]])
```

The same `normalize` works with `JaxComputeBackend` once JAX is installed.

## Start here

<div class="grid cards" markdown>

-   **Getting started**

    ---

    Install XBArray, run a first example on each backend, and get a `ComputeBackend`.

    [Getting started](getting-started.md)

-   **Concepts**

    ---

    Backends, generic typing, devices and dtypes, random numbers, and moving arrays
    between backends.

    [Concepts](concepts/index.md)

-   **Write backend-agnostic code**

    ---

    The patterns that keep one function working on NumPy, PyTorch and JAX.

    [Backend-agnostic code](guides/backend-agnostic-code.md)

-   **Rotation conversions**

    ---

    Quaternions, rotation matrices, Euler angles, axis-angle and the 6D representation.

    [Rotation conversions](guides/rotation-conversions.md)

-   **Point clouds**

    ---

    Depth unprojection, farthest point sampling, ball query and jitter.

    [Point clouds](guides/point-clouds.md)

-   **API reference**

    ---

    Every module, class and function, generated from the source.

    [API reference](api/index.md)

</div>

Contributing? See [Development](guides/development.md) for running the tests and benchmarks.

## Part of UniEnv

XBArray is developed as part of [UniEnv](https://unienvorg.github.io/), which unifies
robot environment and data APIs. The source is at
[github.com/UniEnvOrg/XBArray](https://github.com/UniEnvOrg/XBArray) under the
Apache-2.0 license.

## Cite

XBArray is developed as part of UniEnv. If you use it in research, please cite UniEnv; the
current author list and BibTeX are on [unienvorg.github.io](https://unienvorg.github.io/#cite).
