import numpy as np
import pytest

from xbarray.transformations.rotation_conversions import base as rc

Rotation = pytest.importorskip("scipy.spatial.transform").Rotation

CONVENTIONS = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX", "XYX", "XZX", "YXY", "YZY", "ZXZ", "ZYZ"]


def _np(backend, x):
    return np.asarray(backend.to_numpy(x), dtype=np.float64)


def _arr(backend, x):
    return backend.from_numpy(np.asarray(x, dtype=np.float64))


def _tol(backend, x):
    return 1e-9 if backend.to_numpy(x).dtype == np.float64 else 2e-5


def _special_rotations():
    s = 1 / np.sqrt(2)
    rotvecs = [
        [0, 0, 0], [np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi], [np.pi * s, np.pi * s, 0],
        [0, np.pi - 1e-7, 0], [0, 0, np.pi - 1e-3], [1e-9, 0, 0], [0.3, -0.2, 0.1],
    ]
    return Rotation.concatenate([Rotation.from_rotvec(rotvecs), Rotation.random(40, random_state=0)])


def test_random_quaternions_and_rotations(backend):
    rng = backend.random.random_number_generator(0)
    rng, q = rc.random_quaternions(backend, rng, 16)
    assert tuple(q.shape) == (16, 4)
    q = _np(backend, q)
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1.0, atol=1e-5)
    assert (q[:, 0] >= 0).all()
    rng, q2 = rc.random_quaternions(backend, rng, 16)
    assert not np.allclose(q, _np(backend, q2))  # the returned rng was advanced
    rng, m = rc.random_rotations(backend, rng, 5)
    m = _np(backend, m)
    np.testing.assert_allclose(m @ np.swapaxes(m, -1, -2), np.broadcast_to(np.eye(3), (5, 3, 3)), atol=1e-5)
    rng, m1 = rc.random_rotation(backend, rng)
    assert tuple(m1.shape) == (3, 3)


def test_axis_angle_to_matrix_fast(backend):
    rotations = _special_rotations()
    x = _arr(backend, rotations.as_rotvec())
    out = rc.axis_angle_to_matrix(backend, x, fast=True)
    np.testing.assert_allclose(_np(backend, out), rotations.as_matrix(), atol=_tol(backend, x) * 10)
    single = rc.axis_angle_to_matrix(backend, _arr(backend, [0.1, 0.2, 0.3]), fast=True)
    np.testing.assert_allclose(_np(backend, single), Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix(), atol=1e-5)


@pytest.mark.parametrize("fast", [False, True])
def test_matrix_to_axis_angle(backend, fast):
    rotations = _special_rotations()
    m = _arr(backend, rotations.as_matrix())
    out = _np(backend, rc.matrix_to_axis_angle(backend, m, fast=fast))
    # the rotation vector must rebuild the input (180 degree turns about y or z used to come back about another axis)
    np.testing.assert_allclose(Rotation.from_rotvec(out).as_matrix(), rotations.as_matrix(), atol=_tol(backend, m) * 50)
    assert (np.linalg.norm(out, axis=-1) <= np.pi + 1e-5).all()
    single = _np(backend, rc.matrix_to_axis_angle(backend, _arr(backend, Rotation.from_rotvec([0, np.pi, 0]).as_matrix()), fast=fast))
    np.testing.assert_allclose(np.abs(single), [0, np.pi, 0], atol=1e-5)


@pytest.mark.parametrize("convention", CONVENTIONS)
def test_matrix_to_euler_angles_roundtrip_including_gimbal_lock(backend, convention):
    rotations = Rotation.concatenate([
        _special_rotations(),
        Rotation.from_euler(convention, [[0.3, np.pi / 2, 0.4], [0.3, -np.pi / 2, 0.4], [0.3, 0.0, 0.4],
                                         [0.3, np.pi, 0.4], [0.3, np.pi / 2 - 1e-6, 0.4]]),
        Rotation.from_rotvec([[0, 0, 0.5]]),
    ])
    m = _arr(backend, rotations.as_matrix())
    angles = rc.matrix_to_euler_angles(backend, m, convention)
    rebuilt = _np(backend, rc.euler_angles_to_matrix(backend, angles, convention))
    np.testing.assert_allclose(rebuilt, rotations.as_matrix(), atol=_tol(backend, m) * 50)
    # same convention as scipy's intrinsic (upper-case) Euler angles
    np.testing.assert_allclose(Rotation.from_euler(convention, _np(backend, angles)).as_matrix(), rotations.as_matrix(), atol=_tol(backend, m) * 50)


def test_matrix_to_euler_angles_identity_is_zero(backend):
    for convention in ["ZXZ", "ZYZ", "XYX", "XYZ"]:
        angles = _np(backend, rc.matrix_to_euler_angles(backend, _arr(backend, np.eye(3)), convention))
        np.testing.assert_allclose(angles, 0.0, atol=1e-6)


def test_quaternion_to_axis_angle_negative_real_part(backend):
    # q and -q are the same rotation; float32 lost precision (angle near 2 pi) and JAX gave NaN for -1
    eps = 5e-4
    q = np.array([[-1.0, 0, 0, 0], [-np.cos(eps), 0, -np.sin(eps), 0], [np.cos(eps), 0, np.sin(eps), 0]], dtype=np.float32)
    out = _np(backend, rc.quaternion_to_axis_angle(backend, backend.from_numpy(q)))
    expected = np.array([[0, 0, 0], [0, 2 * eps, 0], [0, 2 * eps, 0]])
    np.testing.assert_allclose(out, expected, atol=1e-6)


@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_quaternion_invert_and_apply_keep_dtype(backend, dtype_name):
    dtype = getattr(backend, dtype_name)
    q = backend.astype(_arr(backend, [[0.9, 0.1, 0.2, 0.3]]), dtype)
    p = backend.astype(_arr(backend, [[1.0, 2.0, 3.0]]), dtype)
    assert rc.quaternion_invert(backend, q).dtype == dtype
    assert rc.quaternion_apply(backend, q, p).dtype == dtype
