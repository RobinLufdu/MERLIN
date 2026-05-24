import torch
from typing import Tuple, Sequence, Optional

def make_grid(
    shape: Tuple[int, ...],
    ranges: Optional[Sequence[Tuple[float, float]]] = None,
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
    flatten: bool = True,         # flatten to (prod(shape), D) if True
) -> torch.Tensor:
    """
    Build a regular D-dimensional grid.

    Args:
        shape: Tuple of sizes per axis, e.g., (H, W) or (D, H, W).
        ranges: Optional list of (lo, hi) per axis. Defaults to (0, 1) for all axes.
        device: Torch device (e.g., "cpu", "cuda").
        dtype: Torch dtype (e.g., torch.float32).
        flatten: If True, return points flattened to shape (N, D) where N=prod(shape).

    Returns:
        Tensor of shape:
            - (prod(shape), D) if flatten,
            - (*shape, D) if not flatten.
    """
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = "cpu"

    D = len(shape)
    if D == 0:
        raise ValueError("`shape` must have at least one dimension.")

    if ranges is None:
        ranges = [(0.0, 1.0)] * D
    if len(ranges) != D:
        raise ValueError(f"`ranges` length ({len(ranges)}) must match dimensionality ({D}).")

    # Build 1D axes
    axes = []
    for n, (lo, hi) in zip(shape, ranges):
        if n <= 0:
            raise ValueError("All entries of `shape` must be positive.")
        axes.append(torch.linspace(lo, hi, steps=n, device=device, dtype=dtype))

    # Mesh into a grid; for D==1 this just returns the axis
    meshes = torch.meshgrid(*axes, indexing="ij")
    grid = torch.stack(meshes, dim=-1)  # (*shape, D)

    if flatten:
        grid = grid.reshape(-1, D)      # (N, D)

    return grid


def delay_stack_last_channel(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Build a delay-embedded feature by stacking the last d time steps along the LAST channel dim.

    Input:
      x: Tensor of shape [B, T, ..., C] (channels-last).
      d: Delay length (must satisfy 1 <= d <= T).

    Output:
      Tensor of shape [B, T - d + 1, ..., C * d].

    For each output time index τ = 0..T-d:
      out[:, τ, ...] = concat( x[:, τ + d - 1, ...], x[:, τ + d - 2, ...], ..., x[:, τ, ...] ) along the last dim.
    """
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    if x.dim() < 3:
        raise ValueError(f"Expected at least [B, T, C], got shape {tuple(x.shape)}")
    if x.size(1) < d:
        raise ValueError(f"T={x.size(1)} must be >= d={d}")

    # Unfold along time: creates sliding windows of size d with step 1.
    # PyTorch places the new window dimension at the LAST axis:
    #   x.unfold(1, d, 1) -> [B, T - d + 1, ..., C, d]
    xw = x.unfold(dimension=1, size=d, step=1)

    # Reverse the window so each window is ordered [t, t-1, ..., t-d+1]
    # (still [B, T - d + 1, ..., C, d])
    xw = xw.flip(dims=[-1])

    # Merge the last two dims (C, d) -> (C * d):
    #   [B, T - d + 1, ..., C * d]
    out = xw.reshape(*xw.shape[:-2], xw.shape[-2] * xw.shape[-1])
    return out


def mask_to_spatial_indices(mask: torch.Tensor, S: int | None = None, threshold: float = 0.5) -> torch.Tensor:
    """
    Convert a 2D observation mask into per-batch flattened spatial indices.

    Args:
        mask: Tensor of shape [B, T, H, W, C], where values above `threshold`
            indicate observed/kept points. The first time step and first channel
            define the spatial mask.
        S: Optional number of kept points per batch. If omitted, use the minimum
            number of observed points across the batch.
        threshold: Mask threshold for selecting points.

    Returns:
        Long tensor of shape [B, S] containing flattened H*W spatial indices.
    """
    if mask.dim() != 5:
        raise ValueError(f"Expected mask shape [B, T, H, W, C], got {tuple(mask.shape)}")

    B, _, H, W, _ = mask.shape
    device = mask.device

    m2d = mask[:, 0, :, :, 0] > threshold
    lin = torch.arange(H, device=device).view(H, 1) * W + torch.arange(W, device=device).view(1, W)
    lin = lin.view(1, H, W).expand(B, -1, -1)
    idx_list = [lin[b][m2d[b]].reshape(-1).long() for b in range(B)]

    S = min(x.numel() for x in idx_list) if S is None else int(S)
    if any(x.numel() < S for x in idx_list):
        raise ValueError(f"Some batches have fewer kept points than S={S}.")

    return torch.stack([x[:S] for x in idx_list], dim=0)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather batched point features by batched indices.

    Args:
        points: Tensor of shape [B, N, C].
        idx: Long tensor of shape [B, S].

    Returns:
        Tensor of shape [B, S, C].
    """
    if points.dim() != 3:
        raise ValueError(f"Expected points shape [B, N, C], got {tuple(points.shape)}")
    if idx.dim() != 2:
        raise ValueError(f"Expected idx shape [B, S], got {tuple(idx.shape)}")
    if points.shape[0] != idx.shape[0]:
        raise ValueError(f"Batch mismatch: points B={points.shape[0]} vs idx B={idx.shape[0]}")

    device = points.device
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1).expand_as(idx)
    return points[batch_indices, idx.to(device=device, dtype=torch.long), :]


def mask_to_bs_index(mask: torch.Tensor, S: int | None = None, threshold: float = 0.5) -> torch.Tensor:
    """Backward-compatible alias for `mask_to_spatial_indices`."""
    return mask_to_spatial_indices(mask=mask, S=S, threshold=threshold)
