# ── Channel-wise normalizer for [B, T, C, ...spatial] ─────────────────────────
import torch
import torch.nn as nn

class UnitNormalizer(nn.Module):
    """
    Channel-wise standardization for tensors shaped [B, T, C, ...spatial] (batch, time, channel, spatial).

    - Statistics (mean/std) are computed across B, T, and all spatial dims,
      i.e., per-channel statistics with keepdim=True for easy broadcasting.
    - Stored as buffers so they move with .to(device) and save in checkpoints.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(eps))

    @classmethod
    def fit(cls, X: torch.Tensor, eps: float = 1e-6):
        """
        Fit per-channel mean/std over dims (B, T, spatial...), keeping dims.
        """
        assert X.dim() >= 3, "Expected shape [B, T, C, ...]"
        spatial_dim = X.dim() - 3
        reduce_dims = (0, 1) + tuple(range(3, 3 + spatial_dim))
        mean = X.mean(dim=reduce_dims, keepdim=True)    # (1, 1, C, ...)
        std  = X.std(dim=reduce_dims, keepdim=True, unbiased=False)
        return cls(mean, std, eps=eps)

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        # Cast buffers to input dtype to avoid dtype mismatches
        return (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)

    @torch.no_grad()
    def decode(self, z: torch.Tensor):
        return z * self.std.to(z.dtype) + self.mean.to(z.dtype)
