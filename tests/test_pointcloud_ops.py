import importlib

import numpy as np
import pytest

from xbarray.transformations.pointcloud import base as pc


def _np(backend, x):
    return np.asarray(backend.to_numpy(x))


def _image():
    return np.arange(6, dtype=np.float32).reshape(2, 3, 1) * 10  # H=2, W=3; pixel (x, y) = 10 * (3y + x)


def test_bilinear_interpolate_last_row_and_column(backend):
    coords = np.array([[0.5, 0.5], [2.0, 0.0], [2.0, 0.5], [1.0, 1.0], [0.0, 1.0], [2.0, 1.0], [1.5, 0.0], [7.0, -3.0]], dtype=np.float32)
    out = pc.bilinear_interpolate(backend, backend.from_numpy(_image()), backend.from_numpy(coords))
    # (7, -3) is clipped onto the border pixel (2, 0)
    np.testing.assert_allclose(_np(backend, out)[:, 0], [20, 20, 35, 40, 30, 50, 15, 20], atol=1e-5)


def test_bilinear_interpolate_single_pixel_wide_image(backend):
    img = np.array([[[1.0]], [[3.0]]], dtype=np.float32)  # H=2, W=1
    out = pc.bilinear_interpolate(backend, backend.from_numpy(img), backend.from_numpy(np.array([[0.0, 0.5], [0.0, 1.0]], dtype=np.float32)))
    np.testing.assert_allclose(_np(backend, out)[:, 0], [2.0, 3.0], atol=1e-6)


def test_bilinear_interpolate_nan_neighbours_are_skipped(backend):
    img = _image()
    img[0, 1, 0] = np.nan
    out = pc.bilinear_interpolate(backend, backend.from_numpy(img), backend.from_numpy(np.array([[0.5, 0.0]], dtype=np.float32)))
    np.testing.assert_allclose(_np(backend, out)[:, 0], [0.0], atol=1e-6)  # only pixel (0, 0) is left


def test_bilinear_interpolate_integer_image(backend):
    img = _image().astype(np.uint8)
    out = pc.bilinear_interpolate(backend, backend.from_numpy(img), backend.from_numpy(np.array([[0.5, 0.5], [2.0, 0.5]], dtype=np.float32)))
    assert backend.dtype_is_real_floating(out.dtype)
    np.testing.assert_allclose(_np(backend, out)[:, 0], [20, 35], atol=1e-5)


def test_world_to_depth(backend):
    w = backend.from_numpy(np.array([[0., 0., 5.], [1., 2., -1.]], dtype=np.float32))
    E = np.eye(4, dtype=np.float32)
    E[2, 3] = 1.0
    np.testing.assert_allclose(_np(backend, pc.world_to_depth(backend, w)), [5, 0])
    np.testing.assert_allclose(_np(backend, pc.world_to_depth(backend, w, backend.from_numpy(E))), [6, 0])
    np.testing.assert_allclose(_np(backend, pc.world_to_depth(backend, w, backend.from_numpy(E[:3]))), [6, 0])


def test_world_to_pixel_respects_valid_mask(backend):
    K = backend.from_numpy(np.array([[100., 0, 50], [0, 100., 40], [0, 0, 1]], dtype=np.float32))
    E = np.eye(4, dtype=np.float32)
    E[2, 3] = 1.0
    w4 = backend.from_numpy(np.array([[0., 0., 5., 1.], [0., 0., 5., 0.]], dtype=np.float32))
    px, d = pc.world_to_pixel_coordinate_and_depth(backend, w4, K)
    np.testing.assert_allclose(_np(backend, d), [5, 0])
    np.testing.assert_allclose(_np(backend, px), [[50, 40], [0, 0]])
    px, d = pc.world_to_pixel_coordinate_and_depth(backend, w4, K, backend.from_numpy(E))
    np.testing.assert_allclose(_np(backend, d), [6, 0])
    np.testing.assert_allclose(_np(backend, pc.world_to_depth(backend, w4, backend.from_numpy(E))), [6, 0])


def test_pixel_world_roundtrip_with_mask_column(backend):
    Rotation = pytest.importorskip("scipy.spatial.transform").Rotation
    K = np.array([[100., 0, 50], [0, 100., 40], [0, 0, 1]], dtype=np.float32)
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
    E[:3, 3] = [0.5, -0.2, 1.0]
    px = np.array([[10., 20.], [60., 30.], [5., 5.]], dtype=np.float32)
    depth = np.array([2.0, 3.0, 0.0], dtype=np.float32)
    f = backend.from_numpy
    world = pc.pixel_coordinate_and_depth_to_world(backend, f(px), f(depth), f(K), f(E))  # (N, 4), last column = valid
    px2, d2 = pc.world_to_pixel_coordinate_and_depth(backend, world, f(K), f(E))
    np.testing.assert_allclose(_np(backend, px2), [[10, 20], [60, 30], [0, 0]], atol=1e-3)
    np.testing.assert_allclose(_np(backend, d2), [2, 3, 0], atol=1e-5)


def test_depth_image_to_world_batched_depth_shared_intrinsics(backend):
    K = np.array([[100., 0, 2], [0, 100., 2], [0, 0, 1]], dtype=np.float32)
    d = np.stack([np.ones((4, 5), np.float32), 2 * np.ones((4, 5), np.float32)])
    shared = pc.depth_image_to_world(backend, backend.from_numpy(d), backend.from_numpy(K))
    batched = pc.depth_image_to_world(backend, backend.from_numpy(d), backend.from_numpy(np.stack([K, K])))
    assert tuple(shared.shape) == (2, 4, 5, 4)
    np.testing.assert_allclose(_np(backend, shared), _np(backend, batched))
    np.testing.assert_allclose(_np(backend, shared)[1, ..., 2], 2.0)
