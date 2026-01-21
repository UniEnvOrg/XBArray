# Changelog

All notable changes to this project will be documented in this file.

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