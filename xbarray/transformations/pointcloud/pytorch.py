from . import base as base_impl
from functools import partial
from xbarray.backends.pytorch import PytorchComputeBackend as BindingBackend

__all__ = [
    "pixel_coordinate_and_depth_to_world",
    "depth_image_to_world",
    "world_to_pixel_coordinate_and_depth",
    "world_to_depth",
]

pixel_coordinate_and_depth_to_world = partial(base_impl.pixel_coordinate_and_depth_to_world, BindingBackend)
depth_image_to_world = partial(base_impl.depth_image_to_world, BindingBackend)
world_to_pixel_coordinate_and_depth = partial(base_impl.world_to_pixel_coordinate_and_depth, BindingBackend)
world_to_depth = partial(base_impl.world_to_depth, BindingBackend)