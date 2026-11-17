# Rotation conversions

`xbarray.transformations.rotation_conversions` converts between rotation
representations. The code is ported from
[PyTorch3D's `rotation_conversions.py`](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py)
and runs on all three backends.

## Modules

| Module | Use |
| --- | --- |
| `xbarray.transformations.rotation_conversions.numpy` | Functions bound to `NumpyComputeBackend` |
| `xbarray.transformations.rotation_conversions.pytorch` | Functions bound to `PyTorchComputeBackend` |
| `xbarray.transformations.rotation_conversions.jax` | Functions bound to `JaxComputeBackend` |
| `xbarray.transformations.rotation_conversions.base` | Generic functions that take the backend as the first argument |

The bound modules are `functools.partial(base.f, Backend)` for each function `f`, so
`numpy.quaternion_to_matrix(q)` is `base.quaternion_to_matrix(NumpyComputeBackend, q)`.

## Conventions

- **Quaternions** have shape `(..., 4)` with the real part first: `(w, x, y, z)`.
- **Rotation matrices** have shape `(..., 3, 3)`.
- **Euler angles** have shape `(..., 3)`, in radians. The `convention` argument is three
  letters from `X`, `Y`, `Z`, such as `"XYZ"` or `"ZYX"`, with the middle letter
  different from its neighbours.
- **Axis-angle** vectors have shape `(..., 3)`. The direction is the axis, and the
  length is the angle in radians, anticlockwise around the axis.
- **6D representation** has shape `(..., 6)`: the first two rows of the rotation matrix,
  from [Zhou et al., 2019](http://arxiv.org/abs/1812.07035).

All functions accept any number of leading batch dimensions.

## Example

A rotation of 90 degrees around the z axis, in several representations:

=== "NumPy"

    ```python
    import math
    import numpy as np
    from xbarray.transformations.rotation_conversions import numpy as rc

    q = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)], dtype=np.float32)

    R = rc.quaternion_to_matrix(q)
    print(np.round(R, 3))
    # [[ 0. -1.  0.]
    #  [ 1.  0.  0.]
    #  [ 0.  0.  1.]]

    print(rc.matrix_to_euler_angles(R, "XYZ"))   # [-0.        0.        1.5707964]
    print(rc.quaternion_to_axis_angle(q))        # [0.        0.        1.5707963]
    print(rc.matrix_to_rotation_6d(R).shape)     # (6,)
    ```

=== "PyTorch"

    ```python
    import math
    import torch
    from xbarray.transformations.rotation_conversions import pytorch as rc

    q = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])

    R = rc.quaternion_to_matrix(q)
    print(R.round(decimals=3))
    # tensor([[ 0., -1.,  0.],
    #         [ 1.,  0.,  0.],
    #         [ 0.,  0.,  1.]])

    print(rc.matrix_to_euler_angles(R, "XYZ"))   # tensor([-0.0000, 0.0000, 1.5708])
    print(rc.quaternion_to_axis_angle(q))        # tensor([0.0000, 0.0000, 1.5708])
    print(rc.matrix_to_rotation_6d(R).shape)     # torch.Size([6])
    ```

=== "JAX"

    ```python
    import math
    import jax.numpy as jnp
    from xbarray.transformations.rotation_conversions import jax as rc

    q = jnp.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])

    R = rc.quaternion_to_matrix(q)
    print(jnp.round(R, 3))
    # [[ 0. -1.  0.]
    #  [ 1.  0.  0.]
    #  [ 0.  0.  1.]]

    print(rc.matrix_to_euler_angles(R, "XYZ"))   # [-0.        0.        1.5707964]
    print(rc.quaternion_to_axis_angle(q))        # [0.        0.        1.5707964]
    print(rc.matrix_to_rotation_6d(R).shape)     # (6,)
    ```

## Rotating points and composing rotations

`quaternion_apply` rotates 3D points. `quaternion_multiply(a, b)` composes two
rotations (apply `b`, then `a`) and returns a quaternion with a non-negative real part.

```python
import math
import numpy as np
from xbarray.transformations.rotation_conversions import numpy as rc

q = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])  # 90 deg about z
points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

print(np.round(rc.quaternion_apply(q, points), 6))
# [[ 0.  1.  0.]
#  [-1.  0.  0.]]

q180 = rc.quaternion_multiply(q, q)
print(np.round(q180, 6))                       # [0. 0. 0. 1.]
print(np.round(rc.quaternion_invert(q), 6))    # [ 0.707107 -0.       -0.       -0.707107]
```

`quaternion_invert` expects unit quaternions. `standardize_quaternion` flips the sign
of quaternions whose real part is negative, so `q` and `-q` map to the same value.

## Function list

| From \ To | Matrix | Quaternion | Euler angles | Axis-angle | 6D |
| --- | --- | --- | --- | --- | --- |
| **Matrix** | | `matrix_to_quaternion` | `matrix_to_euler_angles` | `matrix_to_axis_angle` | `matrix_to_rotation_6d` |
| **Quaternion** | `quaternion_to_matrix` | | | `quaternion_to_axis_angle` | |
| **Euler angles** | `euler_angles_to_matrix` | | | | |
| **Axis-angle** | `axis_angle_to_matrix` | `axis_angle_to_quaternion` | | | |
| **6D** | `rotation_6d_to_matrix` | | | | |

Quaternion operations: `quaternion_multiply`, `quaternion_invert`, `quaternion_apply`,
`standardize_quaternion`.

`axis_angle_to_matrix` and `matrix_to_axis_angle` take a `fast` flag, as in PyTorch3D.
The default, `fast=False`, goes through quaternions.

## Generic use

Inside your own backend-agnostic function, call the `base` module with the backend:

```python
from xbarray import ComputeBackend
from xbarray.transformations.rotation_conversions import base as rc


def rotate_by_euler(backend: ComputeBackend, points, euler, convention="XYZ"):
    """Rotate (..., N, 3) points by (..., 3) Euler angles."""
    R = rc.euler_angles_to_matrix(backend, euler, convention)       # (..., 3, 3)
    return backend.matmul(points, backend.matrix_transpose(R))      # (..., N, 3)


from xbarray.backends.pytorch import PyTorchComputeBackend as backend
import math

pts = backend.asarray([[1.0, 0.0, 0.0]])
print(rotate_by_euler(backend, pts, backend.asarray([0.0, 0.0, math.pi / 2])))
# tensor([[-4.3711e-08,  1.0000e+00,  0.0000e+00]])
```
