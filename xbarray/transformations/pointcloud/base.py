from typing import Optional
from xbarray.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

__all__ = [
    "pixel_coordinate_and_depth_to_world",
    "depth_image_to_world",
    "world_to_pixel_coordinate_and_depth",
    "world_to_depth",
]

def pixel_coordinate_and_depth_to_world(
    backend : ComputeBackend,
    pixel_coordinates : BArrayType, 
    depth : BArrayType,
    intrinsic_matrix : BArrayType,
    extrinsic_matrix : BArrayType
) -> BArrayType:
    """
    Convert pixel coordinates and depth to world coordinates.
    Args:
        backend (ComputeBackend): The compute backend to use.
        pixel_coordinates (BArrayType): The pixel coordinates of shape (..., N, 2).
        depth (BArrayType): The depth values of shape (..., N). Assume invalid depth is either nan or <= 0.
        intrinsic_matrix (BArrayType): The camera intrinsic matrix of shape (..., 3, 3).
        extrinsic_matrix (BArrayType): The camera extrinsic matrix of shape (..., 3, 4) or (..., 4, 4).
    Returns:
        BArrayType: The world coordinates of shape (..., N, 4). The last dimension is (x, y, z, valid_mask).
    """
    xs = pixel_coordinates[..., 0]  # (..., N)
    ys = pixel_coordinates[..., 1]  # (..., N)
    xs_norm = (xs - intrinsic_matrix[..., None, 0, 2]) / intrinsic_matrix[..., None, 0, 0]  # (..., N)
    ys_norm = (ys - intrinsic_matrix[..., None, 1, 2]) / intrinsic_matrix[..., None, 1, 1]  # (..., N)

    camera_coords = backend.stack([
        xs_norm,
        ys_norm,
        backend.ones_like(depth)
    ], dim=-1) # (..., N, 3)
    camera_coords *= depth[..., None]  # (..., N, 3)

    R = extrinsic_matrix[..., :3, :3]  # (..., 3, 3)
    t = extrinsic_matrix[..., :3, 3]  # (..., 3)

    shifted_camera_coords = camera_coords - t[..., None, :]  # (..., N, 3)
    world_coords = backend.matmul(shifted_camera_coords, R) # (..., N, 3)

    valid_depth_mask = backend.logical_not(backend.logical_or(
        backend.isnan(depth),
        depth <= 0
    )) # (..., N)
    return backend.concat([
        world_coords,
        valid_depth_mask[..., None]
    ], dim=-1) # (..., N, 4)

def depth_image_to_world(
    backend : ComputeBackend,
    depth_image : BArrayType,
    intrinsic_matrix : BArrayType,
    extrinsic_matrix : BArrayType
) -> BArrayType:
    """
    Convert a depth image to world coordinates.
    Args:
        backend (ComputeBackend): The compute backend to use.
        depth_image (BArrayType): The depth image of shape (..., H, W).
        intrinsic_matrix (BArrayType): The camera intrinsic matrix of shape (..., 3, 3).
        extrinsic_matrix (BArrayType): The camera extrinsic matrix of shape (..., 3, 4) or (..., 4, 4).
    Returns:
        BArrayType: The world coordinates of shape (..., H, W, 4). The last dimension is (x, y, z, valid_mask).
    """
    H, W = depth_image.shape[-2:]
    ys, xs = backend.meshgrid(
        backend.arange(H, device=backend.device(depth_image), dtype=depth_image.dtype),
        backend.arange(W, device=backend.device(depth_image), dtype=depth_image.dtype),
        indexing='ij'
    ) # (H, W), (H, W)
    pixel_coordinates = backend.stack([xs, ys], dim=-1) # (H, W, 2)
    pixel_coordinates = backend.reshape(pixel_coordinates, [1] * (len(depth_image.shape) - 2) + [H * W, 2]) # (..., H * W, 2)
    world_coords = pixel_coordinate_and_depth_to_world(
        backend,
        pixel_coordinates,
        depth_image.reshape(depth_image.shape[:-2] + [H * W]), # (..., H * W)
        intrinsic_matrix,
        extrinsic_matrix
    ) # (..., H * W, 4)
    world_coords = backend.reshape(world_coords, depth_image.shape[:-2] + [H, W, 4]) # (..., H, W, 4)
    return world_coords

def world_to_pixel_coordinate_and_depth(
    backend : ComputeBackend,
    world_coords : BArrayType,
    intrinsic_matrix : BArrayType,
    extrinsic_matrix : Optional[BArrayType] = None
) -> BArrayType:
    """
    Convert world coordinates to pixel coordinates and depth.
    Args:
        backend (ComputeBackend): The compute backend to use.
        world_coords (BArrayType): The world coordinates of shape (..., N, 3) or (..., N, 4). If the last dimension is 4, the last element is treated as a valid mask.
        intrinsic_matrix (BArrayType): The camera intrinsic matrix of shape (..., 3, 3).
        extrinsic_matrix (Optional[BArrayType]): The camera extrinsic matrix of shape (..., 3, 4) or (..., 4, 4). If None, assume identity matrix.
    Returns:
        BArrayType: The pixel coordinates xy of shape (..., N, 2). 
        BArrayType: The depth values of shape (..., N). Invalid points (where valid mask is False) will have depth 0.
    """
    if world_coords.shape[-1] == 3:
        world_coords_h = backend.pad_dim(
            world_coords,
            dim=-1,
            value=0
        )
    else:
        assert world_coords.shape[-1] == 4
        world_coords_h = world_coords
    
    if extrinsic_matrix is not None:
        camera_coords = backend.matmul(
            extrinsic_matrix, # (..., 3, 4) or (..., 4, 4)
            backend.matrix_transpose(world_coords_h) # (..., 4, N)
        ) # (..., 3, N) or (..., 4, N)
        camera_coords = backend.matrix_transpose(camera_coords) # (..., N, 3) or (..., N, 4)
        if camera_coords.shape[-1] == 4:
            camera_coords = camera_coords[..., :3] / camera_coords[..., 3:4]
    else:
        camera_coords = world_coords_h[..., :3] # (..., N, 3)
    
    point_px_homogeneous = backend.matmul(
        intrinsic_matrix, # (..., 3, 3)
        backend.matrix_transpose(camera_coords) # (..., 3, N)
    ) # (..., 3, N)
    point_px_homogeneous = backend.matrix_transpose(point_px_homogeneous) # (..., N, 3)
    point_px = point_px_homogeneous[..., :2] / point_px_homogeneous[..., 2:3] # (..., N, 2)

    depth = camera_coords[..., 2] # (..., N)
    depth_valid = depth > 0
    depth = backend.where(depth_valid, depth, 0)
    point_px = backend.where(
        depth_valid[..., None],
        point_px,
        0
    )
    return point_px, depth


def world_to_depth(
    backend : ComputeBackend,
    world_coords : BArrayType,
    extrinsic_matrix : Optional[BArrayType] = None
) -> BArrayType:
    """
    Convert world coordinates to pixel coordinates and depth.
    Args:
        backend (ComputeBackend): The compute backend to use.
        world_coords (BArrayType): The world coordinates of shape (..., N, 3) or (..., N, 4). If the last dimension is 4, the last element is treated as a valid mask.
        extrinsic_matrix (Optional[BArrayType]): The camera extrinsic matrix of shape (..., 3, 4) or (..., 4, 4). If None, assume identity matrix.
    Returns:
        BArrayType: The depth values of shape (..., N). Invalid points (where valid mask is False) will have depth 0.
    """
    if world_coords.shape[-1] == 3:
        world_coords_h = backend.pad_dim(
            world_coords,
            dim=-1,
            value=0
        )
    else:
        assert world_coords.shape[-1] == 4
        world_coords_h = world_coords
    
    if extrinsic_matrix is not None:
        camera_coords = backend.matmul(
            extrinsic_matrix, # (..., 3, 4) or (..., 4, 4)
            backend.matrix_transpose(world_coords_h) # (..., 4, N)
        ) # (..., 3, N) or (..., 4, N)
        camera_coords = backend.matrix_transpose(camera_coords) # (..., N, 3) or (..., N, 4)
        if camera_coords.shape[-1] == 4:
            camera_coords = camera_coords[..., :3] / camera_coords[..., 3:4]
    else:
        camera_coords = world_coords_h[..., :3] # (..., N, 3)
    
    depth = camera_coords[..., 2] # (..., N)
    depth_valid = depth > 0
    depth = backend.where(depth_valid, depth, 0)
    return depth

