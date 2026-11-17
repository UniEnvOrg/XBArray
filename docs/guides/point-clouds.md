# Point clouds

`xbarray.transformations.pointcloud` turns depth images into point clouds, samples
points, finds neighbours and adds noise. Like the rotation conversions, it has one
module per backend plus a generic `base` module:

| Module | Use |
| --- | --- |
| `xbarray.transformations.pointcloud.numpy` | Functions bound to `NumpyComputeBackend` |
| `xbarray.transformations.pointcloud.pytorch` | Functions bound to `PyTorchComputeBackend`, with Triton kernels on CUDA |
| `xbarray.transformations.pointcloud.jax` | Functions bound to `JaxComputeBackend` |
| `xbarray.transformations.pointcloud.base` | Generic functions that take the backend as the first argument |

All functions accept leading batch dimensions, written `...` below.

## Depth images to points

`depth_image_to_world(depth_image, intrinsic_matrix, extrinsic_matrix=None)` unprojects
a depth image of shape `(..., H, W)`. The result has shape `(..., H, W, 4)`: `x, y, z`
and a validity flag. Pixels with depth `<= 0` or NaN get flag `0`.

- `intrinsic_matrix` is the `(..., 3, 3)` camera matrix.
- `extrinsic_matrix` is the `(..., 3, 4)` or `(..., 4, 4)` world-to-camera transform
  `[R | t]`, where `camera = R @ world + t`. Without it, points stay in the camera
  frame.

`world_to_pixel_coordinate_and_depth(world_coords, intrinsic_matrix, extrinsic_matrix=None)`
goes the other way. It takes `(..., N, 3)` points, or `(..., N, 4)` points whose last
column is the validity flag, and returns pixel coordinates `(..., N, 2)` in `(x, y)`
order plus depth `(..., N)`. Points behind the camera get depth `0` and pixel `(0, 0)`.

```python
import numpy as np
from xbarray.transformations.pointcloud import numpy as pc

depth = np.full((4, 6), 2.0, dtype=np.float32)
depth[0, 0] = 0.0                                   # an invalid pixel
K = np.array([[100.0, 0.0, 3.0],
              [0.0, 100.0, 2.0],
              [0.0, 0.0, 1.0]], dtype=np.float32)

points = pc.depth_image_to_world(depth, K)          # (4, 6, 4)
print(points[0, 1])                                 # [-0.04 -0.04  2.    1.  ]
print(points[0, 0, 3])                              # 0.0

flat = points.reshape(-1, 4)[:, :3]                 # (24, 3)
pixels, d = pc.world_to_pixel_coordinate_and_depth(flat, K)
print(pixels[:3], d[:3])
# [[0. 0.]
#  [1. 0.]
#  [2. 0.]] [0. 2. 2.]
```

Two helpers read image values at pixel coordinates. Both take `values` of shape
`(..., H, W, C)` and coordinates of shape `(..., N, 2)` in `(x, y)` order, clip the
coordinates to the image, and return `(..., N, C)`:

- `gather_pixel_value(values, pixel_coordinates)` takes integer coordinates. On
  PyTorch they must be `int64`.
- `bilinear_interpolate(values, pixel_coordinates)` takes float coordinates. Neighbour
  pixels that contain NaN are left out of the weighted average.

## Sampling points

Both samplers take points of shape `(..., N, D)` and a generator, and return
`(rng, indices, valid)`. `indices` has shape `(..., num_samples)`. `valid` is `None`
unless you pass a `points_valid` mask of shape `(..., N)`.

- `farthest_point_sampling(points, num_samples, rng, points_valid=None)` picks each
  next point as the one farthest from those already picked. Without a mask the first
  point is random. With a mask it is the first valid point, and the generator is not
  used.
- `random_point_sampling(points, num_samples, rng, points_valid=None)` picks a uniform
  random subset without replacement. With a mask, valid points come first; if a cloud
  has fewer valid points than `num_samples`, the remaining slots hold index `0` and are
  marked invalid.

Use `take_along_axis` to gather the sampled points:

```python
import numpy as np
from xbarray.backends.numpy import NumpyComputeBackend as backend
from xbarray.transformations.pointcloud import numpy as pc

points = np.random.default_rng(0).normal(size=(2, 1000, 3)).astype(np.float32)
rng = backend.random.random_number_generator(0)

rng, idx, valid = pc.farthest_point_sampling(points, 64, rng)
print(idx.shape, valid)                                   # (2, 64) None

sampled = backend.take_along_axis(points, idx[..., None], axis=-2)
print(sampled.shape)                                      # (2, 64, 3)
```

On JAX, `farthest_point_sampling` runs under `jax.jit` as long as `num_samples` is a
static Python int:

```python
import jax
import numpy as np
from xbarray.backends.jax import JaxComputeBackend as backend
from xbarray.transformations.pointcloud import jax as pc

points = jax.numpy.asarray(np.random.default_rng(0).normal(size=(2, 1000, 3)), dtype="float32")


@jax.jit
def sample(points, key):
    key, idx, _ = pc.farthest_point_sampling(points, 64, key)
    return key, idx


key, idx = sample(points, backend.random.random_number_generator(0))
print(idx.shape)  # (2, 64)
```

## Ball query

`ball_query(query_points, points, num_neighbors, radius, query_valid=None, points_valid=None)`
finds, for each query point, up to `num_neighbors` points strictly within `radius`.
They are the first such points in index order, not the nearest ones. It returns three
arrays of shape `(..., Q, num_neighbors)`:

- the neighbour indices, `0` in unused slots;
- the squared distances, `0` in unused slots;
- a validity mask. Each query's valid neighbours come first.

```python
import numpy as np
from xbarray.transformations.pointcloud import numpy as pc

points = np.array([[0.0, 0.0, 0.0],
                   [0.1, 0.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 0.2, 0.0]], dtype=np.float32)
queries = points[:2]

idx, dist2, valid = pc.ball_query(queries, points, num_neighbors=3, radius=0.21)
print(idx)
# [[0 1 3]
#  [0 1 0]]
print(valid)
# [[ True  True  True]
#  [ True  True False]]
```

A common pattern is farthest point sampling for centres, then a ball query around
each centre, as in PointNet++.

## Jitter

`pointcloud_jitter(points, sigma, rng, clip=None)` adds Gaussian noise with standard
deviation `sigma` to the first three channels of `(..., N, D)` points and returns
`(rng, jittered)`. Other channels, such as colours, are unchanged. `clip` limits the
absolute value of the noise.

```python
import torch
from xbarray.backends.pytorch import PyTorchComputeBackend as backend
from xbarray.transformations.pointcloud import pytorch as pc

points = torch.zeros((2, 500, 6))           # xyz + rgb
rng = backend.random.random_number_generator(0)
rng, jittered = pc.pointcloud_jitter(points, 0.01, rng, clip=0.05)
print(jittered[..., :3].abs().max() <= 0.05, jittered[..., 3:].abs().max())
# tensor(True) tensor(0.)
```

## Triton kernels on CUDA

The PyTorch module has [Triton](https://triton-lang.org/) GPU kernels for
`farthest_point_sampling` and `ball_query`. They replace the generic code, which loops
in Python once per sample, with a single kernel launch. The results are the same.

A kernel is used when all of these hold:

- `triton` is installed;
- the point tensors are CUDA `float32` tensors with 3 coordinates (`D == 3`);
- each cloud has at most 2<sup>20</sup> points.

In addition:

- `farthest_point_sampling` uses the kernel only when you pass `points_valid`. Without
  a mask the start point is random and the generic code runs. To use the kernel on a
  cloud with no invalid points, pass an all-true mask.
- `ball_query` uses the kernel only when no gradient is needed, that is when gradients
  are disabled or neither input has `requires_grad`. The kernel's distances carry no
  gradient.

Any other input falls back to the generic implementation. You do not need to change
your code to use the kernels.

```python
import torch
from xbarray.backends.pytorch import PyTorchComputeBackend as backend
from xbarray.transformations.pointcloud import pytorch as pc

device = "cuda" if torch.cuda.is_available() else "cpu"
points = torch.randn(8, 4096, 3, device=device)
mask = torch.ones(points.shape[:-1], dtype=torch.bool, device=device)
rng = backend.random.random_number_generator(0, device=device)

rng, idx, valid = pc.farthest_point_sampling(points, 512, rng, points_valid=mask)
centres = torch.take_along_dim(points, idx[..., None], dim=-2)            # (8, 512, 3)
nbr_idx, nbr_d2, nbr_valid = pc.ball_query(centres, points, 32, 0.2)       # (8, 512, 32)
print(idx.shape, nbr_idx.shape)
```
