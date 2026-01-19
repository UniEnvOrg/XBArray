from typing import Optional, Tuple
from xbarray.backends.base import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType

__all__ = [
    "gather_pixel_value",
    "bilinear_interpolate",
    "pixel_coordinate_and_depth_to_world",
    "depth_image_to_world",
    "world_to_pixel_coordinate_and_depth",
    "world_to_depth",
    "farthest_point_sampling",
    "random_point_sampling"
]

def gather_pixel_value(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    values : BArrayType,
    pixel_coordinates : BArrayType
) -> BArrayType:
    """
    Gather pixel values at given pixel coordinates.
    Args:
        backend (ComputeBackend): The compute backend to use.
        values (BArrayType): The input values of shape (..., H, W, C).
        pixel_coordinates (BArrayType): The pixel coordinates of shape (..., N, 2) in (x, y) order.
    Returns:
        BArrayType: The gathered values of shape (..., N, C).
    """
    assert backend.dtype_is_real_integer(pixel_coordinates.dtype), "pixel_coordinates must be of integer type."
    flat_values = backend.reshape(values, (*values.shape[:-3], -1, values.shape[-1]))  # (..., H * W, C)
    H, W = values.shape[-3], values.shape[-2]
    pixel_coordinates_x = pixel_coordinates[..., 0]  # (..., N)
    pixel_coordinates_y = pixel_coordinates[..., 1]  # (..., N)
    pixel_coordinates_x = backend.clip(pixel_coordinates_x, 0, W - 1)
    pixel_coordinates_y = backend.clip(pixel_coordinates_y, 0, H - 1)
    flat_indices = pixel_coordinates_y * W + pixel_coordinates_x  # (..., N)
    gathered_values = backend.take_along_axis(
        flat_values, 
        flat_indices[..., None], 
        axis=-2
    )  # (..., N, C)
    return gathered_values

def bilinear_interpolate(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    values : BArrayType,
    pixel_coordinates : BArrayType,
    index_dtype : Optional[BDtypeType] = None,
    uniform_weights : bool = False
) -> BArrayType:
    """
    Obtain bilinearly interpolated values at given pixel coordinates.
    Args:
        backend (ComputeBackend): The compute backend to use.
        values (BArrayType): The input values of shape (..., H, W, C).
        pixel_coordinates (BArrayType): The pixel coordinates of shape (..., N, 2) in (x, y) order.
    Returns:
        BArrayType: The interpolated values of shape (..., N, C).
    """
    H, W = values.shape[-3], values.shape[-2]
    x = pixel_coordinates[..., 0]  # (..., N)
    y = pixel_coordinates[..., 1]  # (..., N)

    index_dtype = index_dtype if index_dtype is not None else (pixel_coordinates.dtype if backend.dtype_is_real_integer(pixel_coordinates.dtype) else backend.default_index_dtype)

    x = backend.clip(x, 0, W - 1)
    y = backend.clip(y, 0, H - 1)
    x0 = backend.astype(backend.floor(x), index_dtype)
    x1 = backend.astype(backend.clip(x0 + 1, max=W - 1), index_dtype)
    y0 = backend.astype(backend.floor(y), index_dtype)
    y1 = backend.astype(backend.clip(y0 + 1, max=H - 1), index_dtype)

    pc00 = backend.stack([x0, y0], axis=-1)  # (..., N, 2)
    pc01 = backend.stack([x0, y1], axis=-1)  # (..., N, 2)
    pc10 = backend.stack([x1, y0], axis=-1)  # (..., N, 2)
    pc11 = backend.stack([x1, y1], axis=-1)  # (..., N, 2)
    all_queries = backend.concat([pc00, pc01, pc10, pc11], axis=-2)  # (..., 4 * N, 2)
    gathered_values = gather_pixel_value(
        backend,
        values,
        all_queries
    )  # (..., 4 * N, C)
    values_00 = gathered_values[..., :gathered_values.shape[-2] // 4, :]  # (..., N, C)
    values_01 = gathered_values[..., gathered_values.shape[-2] // 4:2 * gathered_values.shape[-2] // 4, :]  # (..., N, C)
    values_10 = gathered_values[..., 2 * gathered_values.shape[-2] // 4:3 * gathered_values.shape[-2] // 4, :]  # (..., N, C)
    values_11 = gathered_values[..., 3 * gathered_values.shape[-2] // 4:, :]  # (..., N, C)
    
    weight_00 = backend.all(backend.logical_not(backend.isnan(values_00)), axis=-1)  # (..., N)
    weight_01 = backend.all(backend.logical_not(backend.isnan(values_01)), axis=-1)  # (..., N)
    weight_10 = backend.all(backend.logical_not(backend.isnan(values_10)), axis=-1)  # (..., N)
    weight_11 = backend.all(backend.logical_not(backend.isnan(values_11)), axis=-1)  # (..., N)
    values_00 = backend.where(
        weight_00[..., None],
        values_00,
        0
    )  # (..., N, C)
    values_01 = backend.where(
        weight_01[..., None],
        values_01,
        0
    )  # (..., N, C)
    values_10 = backend.where(
        weight_10[..., None],
        values_10,
        0
    )  # (..., N, C)
    values_11 = backend.where(
        weight_11[..., None],
        values_11,
        0
    )  # (..., N, C)
    weight_00 = backend.astype(weight_00, values.dtype)  # (..., N)
    weight_01 = backend.astype(weight_01, values.dtype)  # (..., N)
    weight_10 = backend.astype(weight_10, values.dtype)  # (..., N)
    weight_11 = backend.astype(weight_11, values.dtype)  # (..., N)

    if not uniform_weights:
        weight_00 *= (x1 - x) * (y1 - y)
        weight_01 *= (x1 - x) * (y - y0)
        weight_10 *= (x - x0) * (y1 - y)
        weight_11 *= (x - x0) * (y - y0)
    weights_sum = weight_00 + weight_01 + weight_10 + weight_11  # (..., N)
    weights_sum = backend.clip(weights_sum, min=1e-6)
    weight_00 /= weights_sum
    weight_01 /= weights_sum
    weight_10 /= weights_sum
    weight_11 /= weights_sum
    interpolated_values = (
        values_00 * weight_00[..., None] +
        values_01 * weight_01[..., None] +
        values_10 * weight_10[..., None] +
        values_11 * weight_11[..., None]
    )  # (..., N, C)
    return interpolated_values

def pixel_coordinate_and_depth_to_world(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    pixel_coordinates : BArrayType, 
    depth : BArrayType,
    intrinsic_matrix : BArrayType,
    extrinsic_matrix : Optional[BArrayType] = None
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
    ], axis=-1) # (..., N, 3)
    camera_coords *= depth[..., None]  # (..., N, 3)

    if extrinsic_matrix is not None:
        R = extrinsic_matrix[..., :3, :3]  # (..., 3, 3)
        t = extrinsic_matrix[..., :3, 3]  # (..., 3)

        shifted_camera_coords = camera_coords - t[..., None, :]  # (..., N, 3)
        world_coords = backend.matmul(shifted_camera_coords, R) # (..., N, 3)
    else:
        world_coords = camera_coords  # (..., N, 3)

    valid_depth_mask = backend.logical_not(backend.logical_or(
        backend.isnan(depth),
        depth <= 0
    )) # (..., N)
    return backend.concat([
        world_coords,
        valid_depth_mask[..., None]
    ], axis=-1) # (..., N, 4)

def depth_image_to_world(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    depth_image : BArrayType,
    intrinsic_matrix : BArrayType,
    extrinsic_matrix : Optional[BArrayType] = None
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
    xs, ys = backend.meshgrid(
        backend.arange(W, device=backend.device(depth_image), dtype=depth_image.dtype),
        backend.arange(H, device=backend.device(depth_image), dtype=depth_image.dtype),
        indexing="xy"
    ) # (H, W), (H, W)
    assert xs.shape == (H, W) and ys.shape == (H, W)

    pixel_coordinates = backend.stack([xs, ys], axis=-1) # (H, W, 2)
    pixel_coordinates = backend.reshape(pixel_coordinates, [1] * (len(depth_image.shape) - 2) + [H * W, 2]) # (..., H * W, 2)
    world_coords = pixel_coordinate_and_depth_to_world(
        backend,
        pixel_coordinates,
        backend.reshape(depth_image, list(depth_image.shape[:-2]) + [H * W]), # (..., H * W)
        intrinsic_matrix,
        extrinsic_matrix
    ) # (..., H * W, 4)
    world_coords = backend.reshape(world_coords, list(depth_image.shape[:-2]) + [H, W, 4]) # (..., H, W, 4)
    return world_coords

def world_to_pixel_coordinate_and_depth(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    world_coords : BArrayType,
    intrinsic_matrix : BArrayType,
    extrinsic_matrix : Optional[BArrayType] = None
) -> Tuple[BArrayType, BArrayType]:
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
            target_size=4,
            value=1
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
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
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

def farthest_point_sampling(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    points : BArrayType,
    num_samples : int,
    rng : BRNGType,
    points_valid : Optional[BArrayType] = None
) -> Tuple[BRNGType, BArrayType, Optional[BArrayType]]:
    """
    Perform farthest point sampling on a set of points.
    Args:
        backend (ComputeBackend): The compute backend to use.
        points (BArrayType): The input points of shape (..., N, D).
        num_samples (int): The number of points to sample.
        rng (BRNGType): The random number generator.
        points_valid (Optional[BArrayType]): A boolean mask of shape (..., N) indicating valid points. If None, all points are considered valid.
    Returns:
        BRNGType: The updated random number generator.
        BArrayType: The indices of the sampled points of shape (..., num_samples).
        Optional[BArrayType]: The valid mask of shape (..., num_samples). Returned only if points_valid is provided.
    """
    assert 0 < num_samples <= points.shape[-2], "num_samples must be in (0, N]"
    device = backend.device(points)

    flat_points = backend.reshape(points, [-1, *points.shape[-2:]])  # (B, N, D)
    B, N, D = flat_points.shape
    flat_points_valid = None if points_valid is None else backend.reshape(points_valid, [-1, N])  # (B, N)
    
    batch_indices = backend.arange(B, dtype=backend.default_index_dtype, device=device)

    centroids_idx = backend.zeros((B, num_samples), dtype=backend.default_index_dtype, device=device)  # sampled point indices
    centroids_valid = None if flat_points_valid is None else backend.zeros((B, num_samples), dtype=backend.default_boolean_dtype, device=device)  # valid mask of sampled points

    distance = backend.full((B, N), backend.inf, device=device)  # distance of each point to its nearest centroid
    if flat_points_valid is not None:
        distance = backend.where(
            flat_points_valid,
            distance,
            -backend.inf
        )

    if flat_points_valid is not None:
        farthest_idx = backend.argmax(
            backend.astype(flat_points_valid, backend.default_index_dtype),
            axis=1
        )
    else:
        rng, farthest_idx = backend.random.random_discrete_uniform(
            (B,),
            0, N, 
            rng=rng,
            dtype=backend.default_index_dtype, 
            device=device
        )  # initial random farthest point
    centroids_idx[:, 0] = farthest_idx
    if centroids_valid is not None and flat_points_valid is not None:
        centroids_valid[:, 0] = flat_points_valid[batch_indices, farthest_idx]
    
    for i in range(1, num_samples):
        last_centroid = flat_points[batch_indices, farthest_idx][:, None, :]  # (B, 1, D)
        perpoint_dist_to_last_centroid = backend.sum((flat_points - last_centroid) ** 2, axis=-1)  # (B, N)
        distance = backend.minimum(
            distance,
            perpoint_dist_to_last_centroid
        )  # (B, N)
        farthest_idx = backend.argmax(distance, axis=1) # (B,)
        centroids_idx[:, i] = farthest_idx
        if centroids_valid is not None and flat_points_valid is not None:
            centroids_valid[:, i] = flat_points_valid[batch_indices, farthest_idx]
    
    unflat_centroids_idx = backend.reshape(centroids_idx, list(points.shape[:-2]) + [num_samples])  # (..., num_samples)
    unflat_centroids_valid = None if centroids_valid is None else backend.reshape(centroids_valid, list(points.shape[:-2]) + [num_samples])  # (..., num_samples)
    return rng, unflat_centroids_idx, unflat_centroids_valid

def random_point_sampling(
    backend : ComputeBackend[BArrayType, BDeviceType, BDtypeType, BRNGType],
    points : BArrayType,
    num_samples : int,
    rng : BRNGType,
    points_valid : Optional[BArrayType] = None
) -> Tuple[BRNGType, BArrayType, Optional[BArrayType]]:
    """
    Perform random point sampling on a set of points.
    Args:
        backend (ComputeBackend): The compute backend to use.
        points (BArrayType): The input points of shape (..., N, D).
        num_samples (int): The number of points to sample.
        rng (BRNGType): The random number generator.
        points_valid (Optional[BArrayType]): A boolean mask of shape (..., N) indicating valid points. If None, all points are considered valid.
    Returns:
        BRNGType: The updated random number generator.
        BArrayType: The indices of the sampled points of shape (..., num_samples).
        Optional[BArrayType]: The valid mask of shape (..., num_samples). Returned only if points_valid is provided.
    """
    assert 0 < num_samples <= points.shape[-2], "num_samples must be in (0, N]"
    device = backend.device(points)

    flat_points = backend.reshape(points, [-1, *points.shape[-2:]])  # (B, N, D)
    B, N, D = flat_points.shape
    flat_points_valid = None if points_valid is None else backend.reshape(points_valid, [-1, N])  # (B, N)
    
    if flat_points_valid is None:
        sampled_idx = backend.empty((B, num_samples), dtype=backend.default_index_dtype, device=device)
        for b in range(B):
            rng, idx_b = backend.random.random_permutation(
                N,
                rng=rng,
                device=device
            )
            sampled_idx[b] = idx_b[:num_samples]
        unflat_sampled_idx = backend.reshape(sampled_idx, list(points.shape[:-2]) + [num_samples])
        return rng, unflat_sampled_idx, None
    else:
        # valid_counts = backend.sum(
        #     backend.astype(flat_points_valid, backend.default_index_dtype),
        #     axis=1
        # )  # (B,)
        # assert bool(backend.all(valid_counts >= num_samples)), "Not enough valid points to sample from."
        sampled_idx = backend.zeros((B, num_samples), dtype=backend.default_index_dtype, device=device)
        sampled_valid = backend.zeros((B, num_samples), dtype=backend.default_boolean_dtype, device=device)
        for b in range(B):
            valid_indices_b = backend.nonzero(flat_points_valid[b])[0] # (valid_count_b,)
            rng, permuted_valid_indices_b = backend.random.random_permutation(
                valid_indices_b.shape[0],
                rng=rng,
                device=device
            )
            sampled_idx_b = valid_indices_b[permuted_valid_indices_b[:num_samples]]  # (num_samples,)
            sampled_idx[b, :sampled_idx_b.shape[0]] = sampled_idx_b
            sampled_valid[b, :sampled_idx_b.shape[0]] = True
            sampled_valid[b, sampled_idx_b.shape[0]:] = False
        unflat_sampled_idx = backend.reshape(sampled_idx, list(points.shape[:-2]) + [num_samples])
        unflat_sampled_valid = backend.reshape(sampled_valid, list(points.shape[:-2]) + [num_samples])
        return rng, unflat_sampled_idx, unflat_sampled_valid
