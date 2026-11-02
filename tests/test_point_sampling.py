import importlib

import numpy as np
import pytest


def _sampler(backend):
    name = {"NumpyComputeBackend": "numpy", "PyTorchComputeBackend": "pytorch", "JaxComputeBackend": "jax"}[backend.__name__]
    return importlib.import_module(f"xbarray.transformations.pointcloud.{name}").random_point_sampling


def _np(backend, x):
    return np.asarray(backend.to_numpy(x))


def test_valid_mask_picks_distinct_valid_points_first(backend):
    sample = _sampler(backend)
    rng = backend.random.random_number_generator(0)
    valid = np.zeros((3, 40), dtype=bool)
    valid[0, ::2] = True     # 20 valid >= 8 requested
    valid[1, :5] = True      # 5 valid < 8 requested
    valid[2, 7] = True       # 1 valid
    points = backend.from_numpy(np.random.default_rng(0).normal(size=(3, 40, 3)).astype(np.float32))
    rng, idx, idx_valid = sample(points, 8, rng=rng, points_valid=backend.from_numpy(valid))
    idx, idx_valid = _np(backend, idx), _np(backend, idx_valid)
    assert idx.shape == (3, 8) and idx_valid.shape == (3, 8)
    for b, n_valid in enumerate([20, 5, 1]):
        k = min(n_valid, 8)
        assert idx_valid[b].tolist() == [True] * k + [False] * (8 - k)  # valid prefix, as before
        picked = idx[b, :k]
        assert valid[b, picked].all() and len(set(picked.tolist())) == k
        assert (idx[b, k:] == 0).all()


def test_without_mask_is_a_permutation_prefix(backend):
    sample = _sampler(backend)
    rng = backend.random.random_number_generator(1)
    points = backend.from_numpy(np.zeros((2, 3, 30, 3), dtype=np.float32))  # leading batch dims are kept
    rng, idx, idx_valid = sample(points, 30, rng=rng)
    assert idx_valid is None
    idx = _np(backend, idx)
    assert idx.shape == (2, 3, 30)
    for row in idx.reshape(-1, 30):
        assert sorted(row.tolist()) == list(range(30))


def test_uniform_over_valid_points(np_backend):
    sample = _sampler(np_backend)
    rng = np_backend.random.random_number_generator(2)
    valid = np.zeros((1, 10), dtype=bool)
    valid[0, 2:8] = True  # 6 valid points, pick 3
    counts = np.zeros(10)
    points = np.zeros((1, 10, 3), dtype=np.float32)
    for _ in range(3000):
        rng, idx, _ = sample(points, 3, rng=rng, points_valid=valid)
        counts[idx[0]] += 1
    assert counts[[0, 1, 8, 9]].sum() == 0
    np.testing.assert_allclose(counts[2:8] / 3000, 0.5, atol=0.05)  # each valid point in half the draws
