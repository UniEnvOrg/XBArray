"""Triton GPU kernels for ``farthest_point_sampling`` and ``ball_query`` on CUDA tensors.

Same results as the generic implementations in ``base.py``, without a Python-level
loop per sample: one kernel launch instead of thousands of tensor operations. They
are used for 3D float32 points on CUDA whose result needs no gradient; anything else
goes to the generic implementations.
"""
from typing import Tuple

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None

__all__ = ["supports", "farthest_point_sampling", "ball_query"]

_MAX_BLOCK = 1 << 20  # Triton's largest tensor in a kernel


def supports(*points: torch.Tensor) -> bool:
    return triton is not None and all(
        p.is_cuda and p.dtype == torch.float32 and p.shape[-1] == 3 and p.shape[-2] <= _MAX_BLOCK
        for p in points
    )


def _block_size(n: int) -> int:
    return 1 << max(int(n) - 1, 1).bit_length()


if triton is not None:

    @triton.jit
    def _fps_kernel(points_ptr, valid_ptr, idx_ptr, N, K, BLOCK: tl.constexpr):
        cloud = tl.program_id(0)
        base = points_ptr + cloud * N * 3
        offs = tl.arange(0, BLOCK)
        in_cloud = offs < N
        valid = tl.load(valid_ptr + cloud * N + offs, mask=in_cloud, other=0) != 0
        x = tl.load(base + offs * 3, mask=valid, other=0.0)
        y = tl.load(base + offs * 3 + 1, mask=valid, other=0.0)
        z = tl.load(base + offs * 3 + 2, mask=valid, other=0.0)
        nearest2 = tl.where(valid, float("inf"), -float("inf"))
        selected = tl.argmax(valid.to(tl.int32), 0)
        for k in range(0, K):
            tl.store(idx_ptr + cloud * K + k, selected)
            dx = x - tl.load(base + selected * 3)
            dy = y - tl.load(base + selected * 3 + 1)
            dz = z - tl.load(base + selected * 3 + 2)
            nearest2 = tl.minimum(nearest2, dx * dx + dy * dy + dz * dz)
            selected = tl.argmax(nearest2, 0)

    @triton.jit
    def _ball_kernel(
        query_ptr, query_valid_ptr, points_ptr, points_valid_ptr, idx_ptr, distances2_ptr,
        Q, N, radius2, K: tl.constexpr, BLOCK: tl.constexpr,
    ):
        program = tl.program_id(0)
        if tl.load(query_valid_ptr + program) == 0:
            return
        cloud = program // Q
        query = query_ptr + program * 3
        base = points_ptr + cloud * N * 3
        offs = tl.arange(0, BLOCK)
        in_cloud = offs < N
        valid = tl.load(points_valid_ptr + cloud * N + offs, mask=in_cloud, other=0) != 0
        dx = tl.load(base + offs * 3, mask=valid, other=0.0) - tl.load(query)
        dy = tl.load(base + offs * 3 + 1, mask=valid, other=0.0) - tl.load(query + 1)
        dz = tl.load(base + offs * 3 + 2, mask=valid, other=0.0) - tl.load(query + 2)
        distances2 = dx * dx + dy * dy + dz * dz
        in_ball = valid & (distances2 < radius2)
        rank = tl.cumsum(in_ball.to(tl.int32), 0)
        keep = in_ball & (rank <= K)
        slot = program * K + rank - 1
        tl.store(idx_ptr + slot, offs.to(tl.int64), mask=keep)
        tl.store(distances2_ptr + slot, distances2, mask=keep)


def farthest_point_sampling(points: torch.Tensor, num_samples: int, points_valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Indices and valid mask, both ``(..., num_samples)``; see ``base.farthest_point_sampling``."""
    batch_shape, N = points.shape[:-2], points.shape[-2]
    flat_points = points.reshape(-1, N, 3).contiguous()
    flat_valid = points_valid.reshape(-1, N).contiguous()
    B = flat_points.shape[0]
    idx = torch.empty((B, num_samples), dtype=torch.int64, device=points.device)
    _fps_kernel[(B,)](flat_points, flat_valid.to(torch.uint8), idx, N, num_samples, BLOCK=_block_size(N), num_warps=8)
    valid = torch.gather(flat_valid, 1, idx)
    return idx.reshape(*batch_shape, num_samples), valid.reshape(*batch_shape, num_samples)


def ball_query(
    query_points: torch.Tensor,
    points: torch.Tensor,
    num_neighbors: int,
    radius: float,
    query_valid: torch.Tensor,
    points_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Indices, squared distances and valid mask, all ``(..., Q, num_neighbors)``; see ``base.ball_query``."""
    batch_shape, Q, N = points.shape[:-2], query_points.shape[-2], points.shape[-2]
    flat_query = query_points.reshape(-1, Q, 3).contiguous()
    flat_points = points.reshape(-1, N, 3).contiguous()
    B = flat_points.shape[0]
    idx = torch.full((B, Q, num_neighbors), -1, dtype=torch.int64, device=points.device)
    distances2 = torch.zeros((B, Q, num_neighbors), dtype=points.dtype, device=points.device)
    radius32 = np.float32(radius)
    _ball_kernel[(B * Q,)](
        flat_query, query_valid.reshape(B, Q).contiguous().to(torch.uint8),
        flat_points, points_valid.reshape(B, N).contiguous().to(torch.uint8),
        idx, distances2, Q, N, float(radius32 * radius32),
        K=num_neighbors, BLOCK=_block_size(N), num_warps=8,
    )
    valid = idx >= 0
    shape = (*batch_shape, Q, num_neighbors)
    return idx.clamp(min=0).reshape(shape), distances2.reshape(shape), valid.reshape(shape)
