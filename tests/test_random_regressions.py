import numpy as np
import pytest


def _np(backend, x):
    return np.asarray(backend.to_numpy(x))


def test_random_permutation(backend):
    rng = backend.random.random_number_generator(0)
    rng, p = backend.random.random_permutation(10, rng=rng)
    assert backend.dtype_is_real_integer(p.dtype)
    assert sorted(_np(backend, p).tolist()) == list(range(10))
    rng, p32 = backend.random.random_permutation(10, rng=rng, dtype=backend.int32)
    assert p32.dtype == backend.int32
    assert sorted(_np(backend, p32).tolist()) == list(range(10))


def test_unseeded_generators_differ(backend):
    draws = [
        _np(backend, backend.random.random_uniform(8, rng=backend.random.random_number_generator())[1])
        for _ in range(2)
    ]
    assert not np.array_equal(draws[0], draws[1])


def test_seeded_generators_are_reproducible(backend):
    a = _np(backend, backend.random.random_uniform(8, rng=backend.random.random_number_generator(123))[1])
    b = _np(backend, backend.random.random_uniform(8, rng=backend.random.random_number_generator(123))[1])
    np.testing.assert_array_equal(a, b)


def test_geometric_is_integer_by_default(backend):
    rng = backend.random.random_number_generator(0)
    rng, x = backend.random.random_geometric((1000,), p=0.3, rng=rng)
    assert backend.dtype_is_real_integer(x.dtype)
    assert _np(backend, x).min() >= 1


def test_float_and_bool_dtypes_accepted(backend):
    rng = backend.random.random_number_generator(0)
    rng, g = backend.random.random_geometric((100,), p=0.3, rng=rng, dtype=backend.float32)
    assert g.dtype == backend.float32 and _np(backend, g).min() >= 1
    rng, d = backend.random.random_discrete_uniform((100,), 0, 3, rng=rng, dtype=backend.float32)
    assert d.dtype == backend.float32 and set(_np(backend, d).tolist()) <= {0.0, 1.0, 2.0}
    rng, b = backend.random.random_discrete_uniform((100,), 0, 2, rng=rng, dtype=backend.bool)
    assert b.dtype == backend.bool and set(_np(backend, b).tolist()) == {False, True}


def test_uniform_never_returns_high(backend):
    rng = backend.random.random_number_generator(0)
    for low, high in [(100.0, 101.0), (-3.0, 1e-3)]:
        rng, x = backend.random.random_uniform((4_000_000,), rng=rng, low=low, high=high, dtype=backend.float32)
        x = _np(backend, x)
        assert x.dtype == np.float32
        assert x.min() >= np.float32(low) and x.max() < np.float32(high)
    rng, x = backend.random.random_uniform((64,), rng=rng, low=100.0, high=101.0, dtype=backend.float32)  # small numpy path
    assert _np(backend, x).max() < 101.0


def test_torch_generator_device_is_the_default_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from xbarray.backends.pytorch import PyTorchComputeBackend as B
    rng = B.random.random_number_generator(0, device=torch.device("cuda"))
    for fn, kwargs in [
        (B.random.random_uniform, {}), (B.random.random_normal, {}), (B.random.random_exponential, {}),
        (B.random.random_geometric, {"p": 0.5}),
    ]:
        rng, x = fn((3,), rng=rng, **kwargs)
        assert x.device.type == "cuda"
    rng, x = B.random.random_discrete_uniform((3,), 0, 5, rng=rng)
    assert x.device.type == "cuda"
    rng, x = B.random.random_discrete_uniform((3,), 0, 256, rng=rng, dtype=torch.uint8)
    assert x.device.type == "cuda"
    rng, x = B.random.random_permutation(4, rng=rng)
    assert x.device.type == "cuda"


def test_jax_generator_device():
    jax = pytest.importorskip("jax")
    from xbarray.backends.jax import JaxComputeBackend as B
    cpu = jax.devices("cpu")[0]
    rng = B.random.random_number_generator(0, device=cpu)
    assert rng.devices() == {cpu}
