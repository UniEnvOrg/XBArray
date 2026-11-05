from . import base as base_impl
from . import _pytorch_triton as triton_impl
import torch
from functools import partial
from xbarray.backends.pytorch import PyTorchComputeBackend as BindingBackend

__all__ = [
    "gather_pixel_value",
    "bilinear_interpolate",
    "pixel_coordinate_and_depth_to_world",
    "depth_image_to_world",
    "world_to_pixel_coordinate_and_depth",
    "world_to_depth",
    "farthest_point_sampling",
    "random_point_sampling",
    "ball_query",
    "pointcloud_jitter"
]

gather_pixel_value = partial(base_impl.gather_pixel_value, BindingBackend)
bilinear_interpolate = partial(base_impl.bilinear_interpolate, BindingBackend)
pixel_coordinate_and_depth_to_world = partial(base_impl.pixel_coordinate_and_depth_to_world, BindingBackend)
depth_image_to_world = partial(base_impl.depth_image_to_world, BindingBackend)
world_to_pixel_coordinate_and_depth = partial(base_impl.world_to_pixel_coordinate_and_depth, BindingBackend)
world_to_depth = partial(base_impl.world_to_depth, BindingBackend)
random_point_sampling = partial(base_impl.random_point_sampling, BindingBackend)
pointcloud_jitter = partial(base_impl.pointcloud_jitter, BindingBackend)


def farthest_point_sampling(points, num_samples, rng, points_valid=None):
    """``base.farthest_point_sampling``, with a GPU kernel where it applies. Without a mask the start is random, drawn by the generic code."""
    if points_valid is not None and 0 < num_samples <= points.shape[-2] and triton_impl.supports(points):
        return (rng, *triton_impl.farthest_point_sampling(points, num_samples, points_valid))
    return base_impl.farthest_point_sampling(BindingBackend, points, num_samples, rng, points_valid=points_valid)


def ball_query(query_points, points, num_neighbors, radius, query_valid=None, points_valid=None):
    """``base.ball_query``, with a GPU kernel where it applies. The kernel's distances carry no gradient."""
    needs_grad = torch.is_grad_enabled() and (query_points.requires_grad or points.requires_grad)
    if not needs_grad and 0 < num_neighbors <= points.shape[-2] and radius > 0 and triton_impl.supports(query_points, points):
        if query_valid is None:
            query_valid = torch.ones(query_points.shape[:-1], dtype=torch.bool, device=points.device)
        if points_valid is None:
            points_valid = torch.ones(points.shape[:-1], dtype=torch.bool, device=points.device)
        return triton_impl.ball_query(query_points, points, num_neighbors, radius, query_valid, points_valid)
    return base_impl.ball_query(BindingBackend, query_points, points, num_neighbors, radius, query_valid=query_valid, points_valid=points_valid)
