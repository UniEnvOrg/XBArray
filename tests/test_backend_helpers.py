import collections
import dataclasses

import numpy as np
import pytest


def _np(backend, x):
    return np.asarray(backend.to_numpy(x))


def test_abbreviate_array_partial(backend):
    x = backend.from_numpy(np.array([[1, 2], [1, 2], [1, 2]]))
    out = backend.abbreviate_array(x)
    assert _np(backend, out).tolist() == [[1, 2]]
    out = backend.abbreviate_array(backend.from_numpy(np.array([[1, 1], [2, 2]])), try_cast_scalar=False)
    assert _np(backend, out).tolist() == [[1], [2]]


def test_abbreviate_array_without_scalar_cast(backend):
    out = backend.abbreviate_array(backend.from_numpy(np.full((3, 3), 2.5)), try_cast_scalar=False)
    assert backend.is_backendarray(out)
    assert tuple(out.shape) == (1, 1) and _np(backend, out).item() == 2.5
    assert backend.abbreviate_array(backend.from_numpy(np.full((3, 3), 2.5))) == 2.5


def test_abbreviate_array_not_abbreviable_and_empty(backend):
    x = backend.from_numpy(np.array([[1, 2], [3, 4]]))
    assert _np(backend, backend.abbreviate_array(x)).tolist() == [[1, 2], [3, 4]]
    empty = backend.from_numpy(np.zeros((0, 3)))
    assert tuple(backend.abbreviate_array(empty).shape) == (0, 3)


NT = collections.namedtuple("NT", ["x", "y"])
NT1 = collections.namedtuple("NT1", ["x"])


@dataclasses.dataclass
class Inner:
    v: object


@dataclasses.dataclass
class Outer:
    inner: Inner
    w: object
    tag: str = "t"


def test_map_fn_over_arrays_structures(backend):
    a = backend.from_numpy(np.array([1.0]))
    double = lambda x: x * 2
    out = backend.map_fn_over_arrays({"a": a, "b": 3, "c": "s", "d": None, "e": np.array([1.0])}, double)
    assert _np(backend, out["a"]).tolist() == [2.0]
    assert out["b"] == 3 and out["c"] == "s" and out["d"] is None  # non-arrays pass through on every backend
    if not backend.is_backendarray(np.array([1.0])):
        np.testing.assert_array_equal(out["e"], [1.0])
    nt = backend.map_fn_over_arrays(NT(a, a), double)
    assert type(nt) is NT and _np(backend, nt.y).tolist() == [2.0]
    nt1 = backend.map_fn_over_arrays(NT1(a), double)
    assert type(nt1) is NT1 and backend.is_backendarray(nt1.x)
    outer = backend.map_fn_over_arrays(Outer(Inner(a), [a, (a,)]), double)
    assert type(outer) is Outer and type(outer.inner) is Inner and outer.tag == "t"
    assert _np(backend, outer.inner.v).tolist() == [2.0]
    assert _np(backend, outer.w[1][0]).tolist() == [2.0]


def test_map_fn_over_arrays_ex_structures(backend):
    a = backend.from_numpy(np.array([1.0]))
    mul = lambda x, s: x * s
    out = backend.map_fn_over_arrays_ex({"a": a, "n": 1}, {"a": 3.0, "n": 0, "extra": 1}, func=mul)
    assert _np(backend, out["a"]).tolist() == [3.0] and out["n"] == 1
    nt = backend.map_fn_over_arrays_ex(NT(a, a), (2.0, 4.0), func=mul)
    assert type(nt) is NT and _np(backend, nt.y).tolist() == [4.0]
    outer = backend.map_fn_over_arrays_ex(Outer(Inner(a), a), Outer(Inner(5.0), 7.0), func=mul)
    assert type(outer.inner) is Inner and _np(backend, outer.inner.v).tolist() == [5.0] and _np(backend, outer.w).tolist() == [7.0]


@pytest.mark.parametrize("x", [np.uint8(200), np.asarray([200, 3], dtype=np.uint8), np.asarray(200, dtype=np.uint8)])
@pytest.mark.parametrize("low,high", [(-1.0, 0.25), (-1.0, None), (None, 300.5), (-1.0, 300.5)])
def test_numpy_clip_float_bounds_outside_integer_range(np_backend, x, low, high):
    from array_api_compat import numpy as compat_numpy
    expected = compat_numpy.clip(x, low, high)
    result = np_backend.clip(x, low, high)
    assert result.dtype == x.dtype
    np.testing.assert_array_equal(result, expected)
