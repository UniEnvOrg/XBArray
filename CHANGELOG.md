# Changelog

All notable changes to this project will be documented in this file.

## 1.0.0 (2026-11-23)

First stable release.

- Added `ball_query` to `transformations.pointcloud`: up to K points strictly within a radius, first in index order.
- PyTorch: `farthest_point_sampling` (with a mask) and `ball_query` run as Triton kernels on CUDA for 3D float32 points when `triton` is installed; other inputs use the generic code.
- `farthest_point_sampling` works on the JAX backend, including under `jit` (it wrote its result in place).
- PyTorch backend: `random_uniform` with the default `[0, 1)` bounds returns the draw as is (the two scalar ops it applied cost ~3x the draw for small tensors).
- PyTorch and JAX backends: `random_discrete_uniform` over the whole range of an integer dtype (e.g. `[0, 256)` for uint8 images) draws raw random bits: ~6x faster on CPU for uint8 on PyTorch, ~3x on JAX. It also fixes the full int32 range on JAX (`maxval = 2**31` overflowed) and the full int64 range on PyTorch. Random streams for these draws change for a given seed.

### Fixed (pre-1.0 review)

- `random_quaternions`, `random_rotations` and `random_rotation` always raised `TypeError` on every backend. They now return `(rng, result)`, like the other random functions, so the advanced generator is no longer lost (this matters on JAX).
- JAX backend: `random_permutation` always raised `TypeError` (`jax.random.permutation` has no `dtype` argument).
- PyTorch backend: `random_number_generator()` without a seed returned the same random stream every time, because `torch.Generator` starts from a fixed default seed. It now seeds from fresh entropy, like NumPy.
- `bilinear_interpolate` returned 0 on the last row and the last column of the image, and for coordinates clipped onto them. For example, on a 2×3 image the pixel at `(2, 0)` read as 0.
- `bilinear_interpolate` raised on integer (e.g. `uint8`) images on NumPy and PyTorch. The result is now floating point on every backend.
- `axis_angle_to_matrix(..., fast=True)` always raised `TypeError`.
- `matrix_to_axis_angle(..., fast=True)` returned the wrong axis for rotations near 180° whose axis has no x component (a half turn about y came back as a half turn about z). It also raised on JAX.
- `matrix_to_euler_angles` returned angles that do not rebuild the input matrix at gimbal lock. For example, the identity in `"ZXZ"` gave `(pi, 0, 0)`, and `"XYZ"` at a pitch of ±90° gave wrong roll and yaw. Like scipy, it now sets the third angle to 0 there. The central angle is also more accurate near ±90°.
- `quaternion_to_axis_angle` mishandled quaternions with a negative real part. It returned angles near 2π that lost precision in float32, and NaN on JAX for `(-1, 0, 0, 0)`. It now returns the equivalent rotation with an angle in [0, π].
- NumPy backend: `quaternion_invert` and `quaternion_apply` turned float32 and float16 inputs into float64.
- PyTorch backend: `from_numpy` silently wrapped unsigned integers (uint16 40000 became int16 −25536; uint32 3e9 became a negative int32). uint16 and uint32 now widen to int32 and int64. uint64 values above the int64 range raise `OverflowError` instead of wrapping.
- PyTorch backend: `from_numpy` rejected arrays with negative strides (e.g. `x[::-1]`), arrays with non-native byte order, and NumPy scalars. JAX `from_numpy` also rejected non-native byte order.
- PyTorch backend: `dtype_is_real_integer` returned False for `torch.uint16`, `uint32` and `uint64`, so `abbreviate_array` raised on those dtypes.
- NumPy backend: `from_numpy` ignored its `dtype=` argument.
- PyTorch backend: `to_numpy`, and every `from_other_backend` call converting from PyTorch, failed for tensors that require grad or have the conjugate bit set.
- JAX backend: `from_other_backend` failed for non-contiguous sources (a PyTorch `x[::2]` or `expand()`, a NumPy `broadcast_to` or `x[::-1]`).
- JAX backend: `from_other_backend` left NumPy and CPU data on the CPU, where `from_numpy` would have placed it on the default device. Mixing the result with GPU arrays then raised "incompatible devices".
- JAX backend: `from_other_backend` shared memory with the source, so a later in-place write to the PyTorch tensor or NumPy array changed the JAX array. Conversions into JAX now copy.
- PyTorch backend: `from_other_backend` from JAX shared the JAX buffer, so in-place ops on the tensor changed the JAX array. The tensor now gets its own copy.
- PyTorch backend: random functions raised "Expected a 'cpu' device type for generator" with a CUDA generator unless `device=` was passed. They now draw on the generator's device by default.
- PyTorch backend: `random_geometric` returned float32 by default. It now returns int64, matching NumPy and JAX, which return integers.
- JAX backend: `random_geometric` and `random_discrete_uniform` rejected floating and bool `dtype=` values that NumPy and PyTorch accept.
- `random_uniform` with `low`/`high` could return exactly `high` after rounding (for example float32 in `[100, 101)`) on every backend. Results are now always strictly below `high`.
- JAX backend: `random_number_generator(device=...)` ignored the device. Without a seed it now draws one of 2³² seeds (still from NumPy's global state, so `np.random.seed` keeps it reproducible) instead of one of only 65535.
- `abbreviate_array` returned the original array unless the result was a single scalar: `[[1, 2], [1, 2]]` stayed `(2, 2)`, and with `try_cast_scalar=False` nothing was ever abbreviated. It also raised on empty arrays.
- `map_fn_over_arrays` / `map_fn_over_arrays_ex` turned namedtuples into lists on NumPy and PyTorch, and wrapped a one-field namedtuple's value in a list. Nested dataclasses became dicts, with every array deep-copied.
- `map_fn_over_arrays` / `map_fn_over_arrays_ex` on JAX applied the function to every leaf, including ints, strings and NumPy arrays, and failed on dataclasses. All backends now walk mappings, sequences, namedtuples and dataclasses the same way and apply the function only to backend arrays.
- NumPy backend: `clip` raised `OverflowError` for a float bound outside an integer dtype's range (e.g. a uint8 array with `min=-1.0`). Such a bound can never be active and is now ignored, as in array-api-compat.
- `world_to_depth` always raised `TypeError` for `(..., N, 3)` points.
- `world_to_pixel_coordinate_and_depth` and `world_to_depth` treated the documented valid-mask column of `(..., N, 4)` points as a homogeneous coordinate. Invalid points kept their depth, or got `inf`/NaN depth with a 4×4 extrinsic. Invalid points now get depth 0 and pixel `(0, 0)`.
- `depth_image_to_world` and `pixel_coordinate_and_depth_to_world` failed for a batch of depth images that share one intrinsic matrix.
- PyTorch backend: `serialize_device(torch.device("cuda"))` gave `"cuda:0"`, which deserialized to GPU 0 even when another GPU was current. A device without an index is now serialized without one (`"cuda"`, `"cpu"`). Strings like `"cuda:0"` still deserialize.

## 0.0.1a20 (2026-10-16)

- NumPy backend: `random_discrete_uniform` / `random_uniform` draw directly in the requested integer, bool or float32 dtype (up to ~5x faster on large arrays). NumPy random streams for these dtypes change for a given seed, and `random_discrete_uniform` bounds that do not fit the dtype now raise `ValueError` (as on PyTorch) instead of silently wrapping.
- NumPy backend: `clip` calls NumPy directly instead of the generic array-api-compat implementation (~10x faster on small arrays, up to ~100x on large ones), with the same dtype, broadcasting and NaN behaviour.
- PyTorch and JAX backends: random draws accept an `int` or NumPy-integer `shape` (previously `random_discrete_uniform` / `random_normal` failed on PyTorch and every draw failed on JAX for NumPy integers).
- Added a pytest test suite under `tests/`.

## 0.0.1a19 (2026-08-31)

- Fix `matrix_to_euler_angles` on numpy/jax backends

## 0.0.1a18 (2026-06-05)

- Updated function export after `array_api_extra >= 0.10.2`

## 0.0.1a17 (2026-03-05)
- Added `map_fn_over_arrays_ex` function to the ComputeBackend interface, which extends `map_fn_over_arrays` by allowing the mapped function to use auxiliary data alongside the mapped array. This is useful for transformations that need to transform values based on multiple mappings.

## 0.0.01a16 (2026-02-18)

- Minor bugfix on `xbarray.transformations.rotation_conversions.quaternion_apply`

## 0.0.1a15 (2026-02-07)

- Added `xbarray.backends.get_backend_from_tensor` utility function to infer the appropriate ComputeBackend from a given tensor, improving interoperability with user code.

## 0.0.1a14 (2026-02-05)

- Added `xbarray.transformations.pointcloud.pointcloud_jitter` function to apply jitter noise to pointclouds for data augmentation.
- Rename `PytorchComputeBackend` to `PyTorchComputeBackend` for consistent capitalization with UniEnv.

## 0.0.1a13 (2026-01-21)

- Added `default_index_dtype` property to the ComputeBackend interface to specify the default dtype for indexing operations.
- Make `extrinsic_matrix` parameters optional in the pointcloud transformation functions, defaulting to identity matrices when not provided.
- Minor documentation improvements and bugfixes.

## 0.0.1a12 (2025-12-17)

- For pointcloud-based transformations, fixed a few bugs that caused the depth unprojection to output flipped x and y coordinates in the produced pointcloud
- Fix special case max recursion error when a string or byte data is inside a backend PyTree

## 0.0.1a11 (2025-11-02)

- Added farthest point sampling and random point sampling functions.

## 0.0.1a10 (2025-10-26)

Initial public release of xbarray on the UniEnvOrg Github Institutional account.