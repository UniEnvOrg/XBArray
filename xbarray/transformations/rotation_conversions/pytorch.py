from . import base as base_impl
from functools import partial
from xbarray.backends.pytorch import PyTorchComputeBackend as BindingBackend

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

quaternion_to_matrix = partial(base_impl.quaternion_to_matrix, BindingBackend)
matrix_to_quaternion = partial(base_impl.matrix_to_quaternion, BindingBackend)
euler_angles_to_matrix = partial(base_impl.euler_angles_to_matrix, BindingBackend)
matrix_to_euler_angles = partial(base_impl.matrix_to_euler_angles, BindingBackend)
random_quaternions = partial(base_impl.random_quaternions, BindingBackend)
random_rotations = partial(base_impl.random_rotations, BindingBackend)
random_rotation = partial(base_impl.random_rotation, BindingBackend)
standardize_quaternion = partial(base_impl.standardize_quaternion, BindingBackend)
quaternion_multiply = partial(base_impl.quaternion_multiply, BindingBackend)
quaternion_invert = partial(base_impl.quaternion_invert, BindingBackend)
quaternion_apply = partial(base_impl.quaternion_apply, BindingBackend)
axis_angle_to_matrix = partial(base_impl.axis_angle_to_matrix, BindingBackend)
matrix_to_axis_angle = partial(base_impl.matrix_to_axis_angle, BindingBackend)
axis_angle_to_quaternion = partial(base_impl.axis_angle_to_quaternion, BindingBackend)
quaternion_to_axis_angle = partial(base_impl.quaternion_to_axis_angle, BindingBackend)
rotation_6d_to_matrix = partial(base_impl.rotation_6d_to_matrix, BindingBackend)
matrix_to_rotation_6d = partial(base_impl.matrix_to_rotation_6d, BindingBackend)