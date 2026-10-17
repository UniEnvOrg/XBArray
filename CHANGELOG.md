# Changelog

All notable changes to this project will be documented in this file.

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