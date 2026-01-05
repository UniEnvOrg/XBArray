# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ---------- Note ----------
# This file is pulled from PyTorch3D (https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py)
# with minor modifications for cross-backend compatibility.
# Please refer to the original file for the full license and copyright notice.
# Please see https://github.com/facebookresearch/pytorch3d/issues/2002 for some issues involving axis angle rotations
# --------------------------

from typing import Optional
from xbarray.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

__all__ = [
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "euler_angles_to_matrix",
    "matrix_to_euler_angles",
    "random_quaternions",
    "random_rotations",
    "random_rotation",
    "standardize_quaternion",
    "quaternion_multiply",
    "quaternion_invert",
    "quaternion_apply",
    "axis_angle_to_matrix",
    "matrix_to_axis_angle",
    "axis_angle_to_quaternion",
    "quaternion_to_axis_angle",
    "rotation_6d_to_matrix",
    "matrix_to_rotation_6d",
]

def quaternion_to_matrix(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    quaternions: BArrayType,
) -> BArrayType:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        backend: The backend to use for the computation.
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = backend.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )
    return backend.reshape(
        o, quaternions.shape[:-1] + (3, 3)
    )


def _copysign(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    a: BArrayType, b: BArrayType
) -> BArrayType:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        backend: The backend to use for the computation.
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return backend.where(signs_differ, -a, a)

def _sqrt_positive_part(backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType], x: BArrayType) -> BArrayType:
    """
    Returns backend.sqrt(backend.max(0, x))
    but with a zero subgradient where x is 0.
    """
    positive_mask = x > 0
    ret = backend.where(positive_mask, backend.sqrt(x), x)
    return ret


def matrix_to_quaternion(backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType], matrix: BArrayType) -> BArrayType:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        backend: The backend to use for the computation.
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2], matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2], matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    q_abs = _sqrt_positive_part(
        backend,
        backend.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            axis=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = backend.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            backend.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            backend.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            backend.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            backend.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # Due to https://github.com/data-apis/array-api-compat/issues/271, cannot use `backend.maximum` here
    quat_candidates = quat_by_rijk / (2.0 * backend.clip(q_abs[..., None], min=0.1))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    indices = backend.argmax(q_abs, axis=-1, keepdims=True)
    expand_dims = list(batch_dim) + [1, 4]
    gather_indices = backend.broadcast_to(indices[..., None], expand_dims)
    out = backend.take_along_axis(quat_candidates, gather_indices, axis=-2)[..., 0, :]
    return standardize_quaternion(backend, out)


def _axis_angle_rotation(
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    axis: str, angle: BArrayType
) -> BArrayType:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        backend: The backend to use for the computation.
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = backend.cos(angle)
    sin = backend.sin(angle)
    one = backend.ones_like(angle)
    zero = backend.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return backend.reshape(
        backend.stack(R_flat, axis=-1),
        angle.shape + (3, 3)
    )

def euler_angles_to_matrix(
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    euler_angles: BArrayType, convention: str
) -> BArrayType:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        backend: The backend to use for the computation.
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.ndim == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(backend, c, e)
        for c, e in zip(convention, [euler_angles[..., i] for i in range(3)])
    ]
    # return functools.reduce(torch.matmul, matrices)
    return backend.matmul(backend.matmul(matrices[0], matrices[1]), matrices[2])

def _angle_from_tan(
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    axis: str, other_axis: str, 
    data : BArrayType, 
    horizontal: bool, tait_bryan: bool
) -> BArrayType:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        backend: The backend to use for the computation.
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return backend.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return backend.atan2(-data[..., i2], data[..., i1])
    return backend.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],    
    matrix: BArrayType, convention: str
) -> BArrayType:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = backend.asin(
            backend.clip(matrix[..., i0, i2], -1.0, 1.0)
            * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = backend.acos(backend.clip(matrix[..., i0, i0], -1.0, 1.0))

    o = (
        _angle_from_tan(
            backend,
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            backend,
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return backend.stack(o, -1)

def random_quaternions(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType], rng : BRNGType,
    n: int, dtype: Optional[BDtypeType] = None, device: Optional[BDeviceType] = None,
) -> BArrayType:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        backend: The backend to use for the computation.
        rng: A random number generator of the appropriate type for the backend.
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    o = backend.random.random_normal((n, 4), rng=rng, dtype=dtype, device=device)
    s = backend.sum(o * o, axis=1)
    o = o / _copysign(backend, backend.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType], rng : BRNGType,
    n: int, dtype: Optional[BDtypeType] = None, device: Optional[BDeviceType] = None
) -> BArrayType:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        backend: The backend to use for the computation.
        rng: A random number generator of the appropriate type for the backend.
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(backend, rng, n, dtype=dtype, device=device)
    return quaternion_to_matrix(backend, quaternions)


def random_rotation(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType], rng : BRNGType,
    dtype: Optional[BDtypeType] = None, device: Optional[BDeviceType] = None
) -> BArrayType:
    """
    Generate a single random 3x3 rotation matrix.

    Args:
        backend: The backend to use for the computation.
        rng: A random number generator of the appropriate type for the backend.
        dtype: Type to return
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type

    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return random_rotations(backend, rng, 1, dtype, device)[0]


def standardize_quaternion(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],    
    quaternions: BArrayType) -> BArrayType:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        backend: The backend to use for the computation.
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return backend.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    a: BArrayType, b: BArrayType
) -> BArrayType:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        backend: The backend to use for the computation.
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return backend.stack((ow, ox, oy, oz), axis=-1)


def quaternion_multiply(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType], 
    a: BArrayType, b: BArrayType
) -> BArrayType:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        backend: The backend to use for the computation.
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(backend, a, b)
    return standardize_quaternion(backend, ab)


def quaternion_invert(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],    
    quaternion: BArrayType
) -> BArrayType:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        backend: The backend to use for the computation.
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = backend.reshape(
        backend.asarray([1, -1, -1, -1], device=backend.device(quaternion)),
        [1] * (len(quaternion.shape) - 1) + [4]
    )
    return quaternion * scaling


def quaternion_apply(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],    
    quaternion: BArrayType, point: BArrayType
) -> BArrayType:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        backend: The backend to use for the computation.
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.shape[-1] != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = backend.zeros(point.shape[:-1] + (1,), dtype=point.dtype, device=backend.device(point))
    point_as_quaternion = backend.concat((real_parts, point), axis=-1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(backend, quaternion, point_as_quaternion),
        quaternion_invert(backend, quaternion),
    )
    return out[..., 1:]


def axis_angle_to_matrix(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    axis_angle: BArrayType, fast: bool = False) -> BArrayType:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        backend: The backend to use for the computation.
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if not fast:
        return quaternion_to_matrix(backend, axis_angle_to_quaternion(backend, axis_angle))

    shape = axis_angle.shape
    device, dtype = backend.device(axis_angle), axis_angle.dtype

    angles = backend.linalg.vector_norm(axis_angle, ord=2, axis=-1, keepdims=True)[..., backend.newaxis]

    rx, ry, rz = axis_angle[..., 0], axis_angle[..., 1], axis_angle[..., 2]
    zeros = backend.zeros(shape[:-1], dtype=dtype, device=device)
    cross_product_matrix = backend.reshape(backend.stack(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=-1
    ), (shape + (3,)))
    cross_product_matrix_sqrd = cross_product_matrix @ cross_product_matrix

    identity = backend.eye(3, dtype=dtype, device=device)
    angles_sqrd = angles * angles
    angles_sqrd = backend.where(angles_sqrd == 0, 1, angles_sqrd)
    return (
        backend.reshape(identity, [1] * (len(shape) - 1) + (3,3))
        + backend.sinc(angles / backend.pi) * cross_product_matrix
        + ((1 - backend.cos(angles)) / angles_sqrd) * cross_product_matrix_sqrd
    )


def matrix_to_axis_angle(
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    matrix: BArrayType, fast: bool = False
) -> BArrayType:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        backend: The backend to use for the computation.
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        fast: Whether to use the new faster implementation (based on the
            Rodrigues formula) instead of the original implementation (which
            first converted to a quaternion and then back to a rotation matrix).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.

    """
    if not fast:
        return quaternion_to_axis_angle(backend, matrix_to_quaternion(backend, matrix))

    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    omegas = backend.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        axis=-1,
    )
    norms = backend.linalg.vector_norm(omegas, ord=2, axis=-1, keepdims=True)
    traces = backend.sum(backend.linalg.diagonal(matrix), axis=-1, keepdims=True)
    angles = backend.atan2(norms, traces - 1)

    zeros = backend.zeros(3, dtype=matrix.dtype, device=backend.device(matrix))
    omegas = backend.where(backend.isclose(angles, 0), zeros, omegas)

    near_pi = backend.isclose(angles, backend.pi)[..., 0]

    axis_angles = backend.empty_like(omegas)
    axis_angles = backend.at(axis_angles, backend.logical_not(near_pi)).set(
        0.5 * omegas[~near_pi] / backend.sinc(angles[~near_pi] / backend.pi)
    )

    # this derives from: nnT = (R + 1) / 2
    n = 0.5 * (
        matrix[near_pi][..., 0, :]
        + backend.eye(1, 3, dtype=matrix.dtype, device=backend.device(matrix))
    )
    # TODO(Yunhao): The original pytorch3d file contains `torch.norm` which ignores batch shape, not sure if this is exactly what we wanted
    axis_angles[near_pi] = angles[near_pi] * n / backend.linalg.vector_norm(n, axis=-1, keepdims=True, ord=2)

    return axis_angles


def axis_angle_to_quaternion(
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    axis_angle: BArrayType
) -> BArrayType:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        backend: The backend to use for the computation.
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = backend.linalg.vector_norm(axis_angle, ord=2, axis=-1, keepdims=True)
    sin_half_angles_over_angles = 0.5 * backend.sinc(angles * 0.5 / backend.pi)
    return backend.concat(
        [backend.cos(angles * 0.5), axis_angle * sin_half_angles_over_angles], axis=-1
    )


def quaternion_to_axis_angle(
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    quaternions: BArrayType
) -> BArrayType:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        backend: The backend to use for the computation.
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = backend.linalg.vector_norm(quaternions[..., 1:], ord=2, axis=-1, keepdims=True)
    half_angles = backend.atan2(norms, quaternions[..., :1])
    sin_half_angles_over_angles = 0.5 * backend.sinc(half_angles / backend.pi)
    # angles/2 are between [-pi/2, pi/2], thus sin_half_angles_over_angles
    # can't be zero
    return quaternions[..., 1:] / sin_half_angles_over_angles


def rotation_6d_to_matrix(
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    d6: BArrayType
) -> BArrayType:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        backend: The backend to use for the computation.
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / backend.linalg.vector_norm(a1, ord=2, axis=-1, keepdims=True)
    b2 = a2 - backend.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / backend.linalg.vector_norm(b2, ord=2, axis=-1, keepdims=True)
    b3 = backend.linalg.cross(b1, b2, axis=-1)
    return backend.stack((b1, b2, b3), axis=-2)

def matrix_to_rotation_6d(
    backend: ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    matrix: BArrayType
) -> BArrayType:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        backend: The backend to use for the computation.
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.shape[:-2]
    return backend.reshape(matrix[..., :2, :], (batch_dim + (6,)))