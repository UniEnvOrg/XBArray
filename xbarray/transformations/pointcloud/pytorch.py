from . import base as base_impl
from functools import partial
from xbarray.backends.pytorch import PytorchComputeBackend as BindingBackend

__all__ = [
    "gather_pixel_value",
    "bilinear_interpolate",
    "pixel_coordinate_and_depth_to_world",
    "depth_image_to_world",
    "world_to_pixel_coordinate_and_depth",
    "world_to_depth",
    "farthest_point_sampling",
    "random_point_sampling",
]

gather_pixel_value = partial(base_impl.gather_pixel_value, BindingBackend)
bilinear_interpolate = partial(base_impl.bilinear_interpolate, BindingBackend)
pixel_coordinate_and_depth_to_world = partial(base_impl.pixel_coordinate_and_depth_to_world, BindingBackend)
depth_image_to_world = partial(base_impl.depth_image_to_world, BindingBackend)
world_to_pixel_coordinate_and_depth = partial(base_impl.world_to_pixel_coordinate_and_depth, BindingBackend)
world_to_depth = partial(base_impl.world_to_depth, BindingBackend)
farthest_point_sampling = partial(base_impl.farthest_point_sampling, BindingBackend)
random_point_sampling = partial(base_impl.random_point_sampling, BindingBackend)