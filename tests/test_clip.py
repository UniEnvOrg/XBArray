import itertools
import warnings

import numpy as np
import pytest


def _to_numpy(backend, x):
    return np.asarray(backend.to_numpy(backend.asarray(x)))


# --- all backends ---------------------------------------------------------------------

@pytest.mark.parametrize("dtype_name,low,high", [
    ("float32", -0.5, 0.5),
    ("float32", -1, 1),  # Python ints on a float array
    ("int32", -2, 3),
    ("uint8", 1, 200),
])
@pytest.mark.parametrize("shape", [(), (7,), (2, 3)])
def test_clip_scalar_bounds_keep_dtype(backend, dtype_name, low, high, shape):
    dtype = getattr(backend, dtype_name)
    data = np.linspace(-3, 250, num=max(1, int(np.prod(shape)))).reshape(shape)
    if dtype_name == "uint8":
        data = np.abs(data)
    x = backend.asarray(data.astype(dtype_name), dtype=dtype)
    result = backend.clip(x, low, high)
    assert result.dtype == dtype
    assert tuple(result.shape) == shape
    expected = np.clip(data.astype(dtype_name), low, high).astype(dtype_name)
    np.testing.assert_array_equal(_to_numpy(backend, result), expected)


@pytest.mark.parametrize("dtype_name", ["float32", "int32"])
def test_clip_array_bounds_broadcast(backend, dtype_name):
    dtype = getattr(backend, dtype_name)
    data = np.arange(12).reshape(3, 4).astype(dtype_name) - 5
    low = np.asarray([-1, 0, 1, 2], dtype=dtype_name)
    high = np.asarray([[0], [3], [5]], dtype=dtype_name)
    x = backend.asarray(data, dtype=dtype)
    result = backend.clip(x, backend.asarray(low, dtype=dtype), backend.asarray(high, dtype=dtype))
    assert result.dtype == dtype
    assert tuple(result.shape) == (3, 4)
    np.testing.assert_array_equal(_to_numpy(backend, result), np.minimum(np.maximum(data, low), high))

    # bounds with more dimensions than x broadcast the result
    row = backend.asarray(np.arange(4).astype(dtype_name), dtype=dtype)
    result = backend.clip(row, None, backend.asarray(high, dtype=dtype))
    assert tuple(result.shape) == (3, 4)


def test_clip_none_bounds(backend):
    data = np.asarray([-2.0, 0.0, 2.0], dtype=np.float32)
    x = backend.asarray(data, dtype=backend.float32)
    np.testing.assert_array_equal(_to_numpy(backend, backend.clip(x, min=-1.0)), [-1.0, 0.0, 2.0])
    np.testing.assert_array_equal(_to_numpy(backend, backend.clip(x, max=1.0)), [-2.0, 0.0, 1.0])
    np.testing.assert_array_equal(_to_numpy(backend, backend.clip(x)), data)


def test_clip_nan(backend):
    x = backend.asarray(np.asarray([np.nan, -2.0, 2.0], dtype=np.float32), dtype=backend.float32)
    result = _to_numpy(backend, backend.clip(x, -1.0, 1.0))
    np.testing.assert_array_equal(result, [np.nan, -1.0, 1.0])  # NaN in x propagates
    nan_bound = backend.asarray(np.asarray([np.nan, 0.0, np.nan], dtype=np.float32), dtype=backend.float32)
    result = _to_numpy(backend, backend.clip(x, nan_bound, 1.0))
    np.testing.assert_array_equal(result, [np.nan, 0.0, np.nan])  # NaN in a bound propagates


def test_clip_min_greater_than_max(backend):
    x = backend.asarray(np.asarray([-5.0, 0.0, 5.0], dtype=np.float32), dtype=backend.float32)
    np.testing.assert_array_equal(_to_numpy(backend, backend.clip(x, 2.0, 1.0)), [1.0, 1.0, 1.0])


# --- numpy: equivalence with the array_api_compat implementation --------------------

DTYPES = ["int8", "uint8", "int32", "int64", "uint64", "float16", "float32", "float64", "bool"]
X_SHAPES = [(), (5,), (3, 4)]


def _sample_x(dtype, shape):
    dtype = np.dtype(dtype)
    n = int(np.prod(shape))
    if dtype.kind == "b":
        data = np.arange(n) % 2 == 0
    elif dtype.kind == "f":
        data = np.linspace(-150, 150, num=n).astype(dtype)
        if n > 2:
            data[1] = np.nan
            data[2] = np.inf
    else:
        info = np.iinfo(dtype)
        data = np.linspace(max(info.min, -150), min(info.max, 250), num=n).astype(dtype)
    return np.asarray(data, dtype=dtype).reshape(shape)


def _bound_variants(dtype):
    dtype = np.dtype(dtype)
    variants = [None, 0, 3, -7, 100, 2.5, -0.5, float("nan"), True, np.float64(1.5), np.int16(4)]
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        variants += [int(info.min), int(info.max), int(info.min) - 1, int(info.max) + 1]
    variants += [
        np.asarray(5, dtype=dtype),
        np.asarray([0, 1, 2, 3], dtype=dtype),            # broadcasts against (3, 4) and ()
        np.asarray([[1], [2], [3]], dtype=np.float64),    # different dtype, broadcasts
        np.asarray([-1.0, np.nan, 2.0, 50.0], dtype=np.float32),
    ]
    return variants


def _outcome(fn):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            return fn(), None
        except Exception as e:  # noqa: BLE001 - compare the failure mode as well
            return None, type(e)


def _equivalence_cases():
    for dtype, shape in itertools.product(DTYPES, X_SHAPES):
        bounds = _bound_variants(dtype)
        for i, j in itertools.product(range(len(bounds)), repeat=2):
            if (i + j) % 3 and not (i == 0 or j == 0):
                continue  # keep the grid manageable: all single-bound cases + a third of the pairs
            yield dtype, shape, i, j


@pytest.mark.parametrize("dtype,shape,min_idx,max_idx", list(_equivalence_cases()))
def test_numpy_clip_matches_array_api_compat(np_backend, dtype, shape, min_idx, max_idx):
    from array_api_compat import numpy as compat_numpy

    bounds = _bound_variants(dtype)
    low, high = bounds[min_idx], bounds[max_idx]
    x = _sample_x(dtype, shape)
    expected, expected_err = _outcome(lambda: compat_numpy.clip(x, low, high))
    result, err = _outcome(lambda: np_backend.clip(x, low, high))
    if low is None or high is None:
        assert err is expected_err
    else:
        # with two invalid bounds, which one is reported first may differ
        assert (err is None) == (expected_err is None)
    if expected_err is not None:
        return
    assert type(result) is type(expected)
    assert result.dtype == expected.dtype == x.dtype
    assert np.shape(result) == np.shape(expected)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("x", [np.float32(0.5), np.float64(-3.0), np.int64(7), np.uint8(200)])
@pytest.mark.parametrize("low,high", [(0, 5), (-1.0, 0.25), (None, None), (None, 100)])
def test_numpy_clip_scalar_input(np_backend, x, low, high):
    from array_api_compat import numpy as compat_numpy

    expected, expected_err = _outcome(lambda: compat_numpy.clip(x, low, high))
    result, err = _outcome(lambda: np_backend.clip(x, low, high))
    assert err is expected_err  # e.g. -1.0 does not fit uint8
    assert type(result) is type(expected)
    assert result == expected


def test_numpy_clip_returns_new_array(np_backend):
    x = np.asarray([1.0, 2.0, 3.0])
    for low, high in [(None, None), (0.0, None), (None, 5.0), (0.0, 5.0)]:
        result = np_backend.clip(x, low, high)
        assert not np.shares_memory(result, x)


def test_numpy_clip_out(np_backend):
    x = np.asarray([-2.0, 0.5, 2.0], dtype=np.float32)
    out = np.empty_like(x)
    result = np_backend.clip(x, -1.0, 1.0, out=out)
    np.testing.assert_array_equal(out, [-1.0, 0.5, 1.0])
    np.testing.assert_array_equal(result, out)
