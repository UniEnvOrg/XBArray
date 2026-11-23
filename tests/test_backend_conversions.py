import importlib

import numpy as np
import pytest

from conftest import load_backend

BACKEND_NAMES = ["numpy", "pytorch", "jax"]


def _np(backend, x):
    return np.asarray(backend.to_numpy(x))


def test_from_numpy_respects_dtype(backend):
    x = backend.from_numpy(np.array([1.5, 2.5]), dtype=backend.float32)
    assert x.dtype == backend.float32
    np.testing.assert_array_equal(_np(backend, x), [1.5, 2.5])


@pytest.mark.parametrize("data", [
    np.arange(5)[::-1],                           # negative stride
    np.arange(12).reshape(3, 4)[::-1, ::2],       # negative and non-unit strides
    np.array([1, 2, 300], dtype=">i4"),           # non-native byte order
    np.float32(3.5),                              # numpy scalar
    np.array(4.0),                                # 0-d
    np.array([True, False]),
], ids=["neg-stride", "neg-2d", "big-endian", "np-scalar", "0d", "bool"])
def test_from_numpy_unusual_layouts(backend, data):
    x = backend.from_numpy(data)
    np.testing.assert_array_equal(_np(backend, x), np.asarray(data))


@pytest.mark.parametrize("np_dtype,values", [
    (np.uint16, [40000, 65535, 1]),
    (np.uint32, [3_000_000_000, 1]),
])
def test_from_numpy_unsigned_values_are_kept(backend, np_dtype, values):
    x = backend.from_numpy(np.array(values, dtype=np_dtype))
    assert _np(backend, x).astype(np.int64).tolist() == values
    assert backend.dtype_is_real_integer(x.dtype)


def test_torch_from_numpy_uint64_overflow_is_loud():
    torch = pytest.importorskip("torch")
    B = load_backend("pytorch")
    assert B.from_numpy(np.array([2**62], dtype=np.uint64)).tolist() == [2**62]
    with pytest.raises(OverflowError):
        B.from_numpy(np.array([2**63 + 5], dtype=np.uint64))


def test_torch_dtype_is_real_integer_unsigned():
    torch = pytest.importorskip("torch")
    B = load_backend("pytorch")
    for name in ["uint16", "uint32", "uint64"]:
        if hasattr(torch, name):
            assert B.dtype_is_real_integer(getattr(torch, name))
    assert B.abbreviate_array(torch.full((2,), 3, dtype=torch.uint16)) == 3


def test_torch_to_numpy_requires_grad_and_conj():
    torch = pytest.importorskip("torch")
    B = load_backend("pytorch")
    t = torch.ones(3, requires_grad=True) * 2
    np.testing.assert_array_equal(B.to_numpy(t), [2, 2, 2])
    np.testing.assert_array_equal(B.to_numpy(torch.tensor([1 + 2j]).conj()), [1 - 2j])
    for target in BACKEND_NAMES:
        T = load_backend(target)
        np.testing.assert_array_equal(_np(T, T.from_other_backend(B, t)), [2, 2, 2])


def _sources():
    torch = pytest.importorskip("torch")
    sources = [
        ("numpy", lambda: np.arange(10.0)[::2]),
        ("numpy", lambda: np.arange(5.0)[::-1]),
        ("numpy", lambda: np.broadcast_to(np.arange(3.0), (2, 3))),
        ("numpy", lambda: np.array(5.0)),
        ("pytorch", lambda: torch.arange(10.0)[::2]),
        ("pytorch", lambda: torch.arange(3.0).expand(2, 3)),
        ("pytorch", lambda: torch.arange(6.0).reshape(2, 3).T),
        ("pytorch", lambda: torch.tensor([True, False])),
    ]
    if torch.cuda.is_available():
        sources += [
            ("pytorch", lambda: torch.arange(10.0, device="cuda")[::2]),
            ("pytorch", lambda: torch.arange(3.0, device="cuda").expand(2, 3)),
            ("pytorch", lambda: torch.arange(6.0, device="cuda").reshape(2, 3).T),
        ]
    return sources


@pytest.mark.parametrize("target", BACKEND_NAMES)
def test_from_other_backend_strided_sources(target):
    T = load_backend(target)
    for source_name, make in _sources():
        S = load_backend(source_name)
        data = make()
        out = T.from_other_backend(S, data)
        np.testing.assert_array_equal(_np(T, out), _np(S, data))


def test_jax_from_other_backend_does_not_alias_and_uses_default_device():
    jax = pytest.importorskip("jax")
    torch = pytest.importorskip("torch")
    J, N, P = load_backend("jax"), load_backend("numpy"), load_backend("pytorch")
    a = np.zeros(3, dtype=np.float32)
    x = J.from_other_backend(N, a)
    assert x.devices() == J.from_numpy(a).devices() == {jax.devices()[0]}
    _ = x + jax.device_put(J.zeros(3), jax.devices()[0])  # no "incompatible devices" error
    a[0] = 9
    assert float(x[0]) == 0.0
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        t = torch.zeros(3, device=device)
        y = J.from_other_backend(P, t)
        t += 5
        np.testing.assert_array_equal(np.asarray(y), [0, 0, 0])


def test_torch_from_jax_does_not_alias():
    jax = pytest.importorskip("jax")
    torch = pytest.importorskip("torch")
    J, P = load_backend("jax"), load_backend("pytorch")
    x = J.asarray([0.0, 1.0, 2.0])
    t = P.from_other_backend(J, x)
    t += 10
    np.testing.assert_array_equal(np.asarray(x), [0, 1, 2])
