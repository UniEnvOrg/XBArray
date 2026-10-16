import numpy as np
import pytest

# 64-bit dtypes are left out of the cross-backend tests: jax truncates them to 32 bits by default.
INT_DTYPES = ["int8", "uint8", "int16", "int32"]
FLOAT_DTYPES = ["float32"]
SHAPES = [(), (1,), (16,), (4, 64)]  # (4, 64) takes the large-array float32 path on numpy


def _to_numpy(backend, x):
    return np.asarray(backend.to_numpy(x))


@pytest.mark.parametrize("dtype_name", INT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_discrete_uniform_dtype_shape_bounds(backend, dtype_name, shape):
    dtype = getattr(backend, dtype_name)
    rng = backend.random.random_number_generator(0)
    rng, x = backend.random.random_discrete_uniform(shape, 3, 9, rng=rng, dtype=dtype)
    assert backend.is_backendarray(x)
    assert x.dtype == dtype
    assert tuple(x.shape) == shape
    values = _to_numpy(backend, x)
    assert values.min(initial=3) >= 3 and values.max(initial=8) <= 8  # high is exclusive


@pytest.mark.parametrize("dtype_name", ["int8", "uint8", "int32"])
def test_discrete_uniform_covers_range(backend, dtype_name):
    dtype = getattr(backend, dtype_name)
    rng = backend.random.random_number_generator(1)
    rng, x = backend.random.random_discrete_uniform((20_000,), 0, 8, rng=rng, dtype=dtype)
    counts = np.bincount(_to_numpy(backend, x).astype(np.int64), minlength=8)
    assert counts.shape == (8,)  # nothing outside [0, 8)
    assert np.all(np.abs(counts / counts.sum() - 1 / 8) < 0.02)


def test_discrete_uniform_full_dtype_range(backend):
    rng = backend.random.random_number_generator(2)
    rng, x = backend.random.random_discrete_uniform((50_000,), 0, 256, rng=rng, dtype=backend.uint8)
    values = _to_numpy(backend, x)
    assert x.dtype == backend.uint8
    assert values.min() == 0 and values.max() == 255


@pytest.mark.parametrize("dtype_name", FLOAT_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("low,high", [(0.0, 1.0), (-1.0, 1.0), (2.0, 5.0)])
def test_uniform_dtype_shape_bounds(backend, dtype_name, shape, low, high):
    dtype = getattr(backend, dtype_name)
    rng = backend.random.random_number_generator(0)
    rng, x = backend.random.random_uniform(shape, rng=rng, low=low, high=high, dtype=dtype)
    assert backend.is_backendarray(x)
    assert x.dtype == dtype
    assert tuple(x.shape) == shape
    values = _to_numpy(backend, x)
    assert np.all(values >= low) and np.all(values <= high)


@pytest.mark.parametrize("low,high", [(0.0, 1.0), (-3.0, 5.0)])
def test_uniform_distribution(backend, low, high):
    rng = backend.random.random_number_generator(3)
    rng, x = backend.random.random_uniform((50_000,), rng=rng, low=low, high=high, dtype=backend.float32)
    values = _to_numpy(backend, x).astype(np.float64)
    width = high - low
    assert values.min() >= low and values.max() < high
    assert abs(values.mean() - (low + high) / 2) < 0.01 * width
    assert abs(values.std() - width / np.sqrt(12)) < 0.01 * width


# --- numpy-specific behaviour -------------------------------------------------------

@pytest.mark.parametrize("shape", [(), (8,), (300,)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16, None])
def test_numpy_uniform_dtypes(np_backend, shape, dtype):
    rng = np_backend.random.random_number_generator(0)
    rng, x = np_backend.random.random_uniform(shape, rng=rng, low=-2.0, high=3.0, dtype=dtype)
    assert isinstance(x, np.ndarray)  # also for 0-d results
    assert x.dtype == (np.float64 if dtype is None else dtype)
    assert x.shape == shape
    assert np.all(x >= -2.0) and np.all(x <= 3.0)


@pytest.mark.parametrize("dtype", [np.int8, np.uint16, np.int64, np.bool_, np.float32, None])
def test_numpy_discrete_uniform_dtypes(np_backend, dtype):
    rng = np_backend.random.random_number_generator(0)
    high = 2 if dtype is np.bool_ else 7
    rng, x = np_backend.random.random_discrete_uniform((), 0, high, rng=rng, dtype=dtype)
    assert isinstance(x, np.ndarray)
    assert x.dtype == (np.int64 if dtype is None else dtype)
    rng, x = np_backend.random.random_discrete_uniform((1000,), 0, high, rng=rng, dtype=dtype)
    assert x.dtype == (np.int64 if dtype is None else dtype)
    assert set(np.unique(x).astype(np.int64).tolist()) == set(range(high))


def test_numpy_discrete_uniform_out_of_dtype_range_raises(np_backend):
    rng = np_backend.random.random_number_generator(0)
    with pytest.raises(ValueError):
        np_backend.random.random_discrete_uniform((4,), 0, 300, rng=rng, dtype=np.uint8)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.uint8])
def test_numpy_same_seed_same_stream(np_backend, dtype):
    def draw():
        rng = np_backend.random.random_number_generator(123)
        if np.issubdtype(dtype, np.floating):
            return np_backend.random.random_uniform((500,), rng=rng, low=-1.0, high=4.0, dtype=dtype)[1]
        return np_backend.random.random_discrete_uniform((500,), 0, 100, rng=rng, dtype=dtype)[1]

    np.testing.assert_array_equal(draw(), draw())


def test_numpy_uniform_float32_large_bounds(np_backend):
    # The float32 range does not fit in float32 (high - low overflows): must stay finite
    rng = np_backend.random.random_number_generator(0)
    big = float(np.finfo(np.float32).max)
    rng, x = np_backend.random.random_uniform((1000,), rng=rng, low=-big, high=big, dtype=np.float32)
    assert x.dtype == np.float32
    assert np.all(np.isfinite(x))
