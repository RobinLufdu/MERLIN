from dataclasses import dataclass, field
from typing import Tuple, Union, List, Optional, Dict, Iterable

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from einops import rearrange

from utilities.normalizer import UnitNormalizer
from utilities.utils import seed_everything, make_worker_init_fn
from utilities.data_proc import make_grid



@dataclass
class Dataloader_Configs:
    dataset: str 
    n_train_trajs: int
    n_test_trajs: int
    n_samples_per_traj: int
    train_bs: int 
    test_bs: int
    num_workers: int 

    n_frames_train: int
    n_frames_out: int
    n_frames_cond: int

    # train_traj_ids: List[int] | None = None
    # test_traj_ids: List[int] | None = None
    train_traj_ids: Optional[List[int]] = field(default=None, repr=False, compare=False, metadata={"log": False})
    test_traj_ids:  Optional[List[int]] = field(default=None, repr=False, compare=False, metadata={"log": False})
    limit_trajs: Optional[int] = None
    
    normalize: bool = True
    sample_strategy: str = "random"
    mode: str = "autoregressive"

    dt_eval: float = 0.25   # physical dt
    seed: int = 42

    # mask configs (for 2d datasets)
    mask_ratio: float = 0.0
    block_size: Union[int, Tuple[int, int]] = (1, 1)
    same_over_time: bool = True
    same_over_channel: bool = True
    
    

class PDEDataset(Dataset):
    def __init__(self, data_tensor: torch.Tensor,
                 cfg: Dataloader_Configs,
                 n_frames_train: int, n_frames_cond: int,
                 n_frames_out: int,
                 traj_indices: List[int],
                 n_sample_per_traj: int,
                 sample_strategy: str = "random",
                 mode: str = "autoregressive",
                 group: str = "train",
                 *,
                 samples: Optional[List[Tuple[int, int]]] = None,
                 mask_tensor: torch.Tensor | None = None,
                 np_rng: np.random.Generator | None = None
                 ):
        """
        data_tensor: [N, T, C, (*spatial_dims)]
        sample_strategy: "random" / "disjoint"
        mode: "autoregressive" / "interpolation"
        group: "train" / "test" / "train_eval"
            - "train":      [:n_frames_cond] -> [:n_frames_train]
            - "train_eval": [:n_frames_cond] -> [:n_frames_train + n_frames_out]
            - "test":       [:n_frames_cond] -> [:n_frames_train + n_frames_out], with initial condition different from "train"
        """
        super().__init__()
        self.data = data_tensor
        self.spatial_dim = data_tensor.dim() - 3
        self.traj_indices = traj_indices
        self.mask_tensor = mask_tensor

        self.cfg = cfg
        self.mode = mode
        self.group = group
        assert self.mode in {"autoregressive", "interpolation"}, "Unknown dataset generating mode"
        assert self.group in {"train", "test", "train_eval"}, "Unknown dataset groups"

        self.n_frames_train, self.n_frames_cond, self.n_frames_out = n_frames_train, n_frames_cond, n_frames_out

        self.samples = []
        self.sample_strategy = sample_strategy
        self.np_rng = np_rng or np.random.default_rng(cfg.seed)
        max_t0 = data_tensor.shape[1] - self.n_frames_train - self.n_frames_out
        if max_t0 < 0:
            raise ValueError(f"Not enough frames: T={data_tensor.shape[1]}, "
                            f"need at least n_frames_train + n_frames_out = "
                            f"{self.n_frames_train + self.n_frames_out}.")
        if samples is not None:
            self.samples = [(int(traj_indice), int(t0)) for (traj_indice, t0) in samples]
        else:
            if self.sample_strategy == "random":
                self.n_sample_per_traj = min(n_sample_per_traj, max_t0 + 1)
                for traj_indice in self.traj_indices:
                    t0s = self.np_rng.choice((max_t0+1), size=self.n_sample_per_traj, replace=False)
                    self.samples.extend((int(traj_indice), int(t0)) for t0 in t0s)
            elif self.sample_strategy == "disjoint":
                t0s = range(0, max_t0+1, self.n_frames_train + self.n_frames_out)
                self.n_sample_per_traj = min(len(t0s), n_sample_per_traj)
                for traj_indice in self.traj_indices:
                    self.samples.extend((int(traj_indice), int(t0)) for t0 in t0s[:self.n_sample_per_traj])
        

    def __len__(self):
        return len(self.samples)


    def _stack_time_as_channels(self, x: torch.Tensor, time_dim: int = 0, channel_dim: int = 1) -> torch.Tensor:
        """
        Merge the time dimension into the channel dimension: [T, C, ..., ] -> [..., T*C].
        """
        ndim = x.ndim
        t = time_dim % ndim
        c = channel_dim % ndim
        if t == c:
            raise ValueError("time_dim and channel_dim must be different.")
        # Keep all other dims (batch/spatial/etc.) in-place
        other_dims = [d for d in range(ndim) if d not in (t, c)]
        # Reorder to [..., T, C]
        x_perm = x.permute(*other_dims, t, c).contiguous()
        # Collapse T and C -> T*C
        *prefix, T, C = x_perm.shape
        return x_perm.reshape(*prefix, T * C)   # [..., T*C]
    

    def _channel_last(self, x: torch.Tensor) -> torch.Tensor:
        # x: [t, c, (*spatial_dims)] -> [t, (*spatial_dims), c]
        x_perm = rearrange(x, 't c ... -> t ... c')
        return x_perm
    

    def _rearrange_tc(self, x: torch.Tensor, with_batch: bool = False):
        """
        with_batch: [B, T, C, (*spatial_dims)] --> [B, (*spatial_dims), T, C]
        otherwise: [T, C, (*spatial_dims)] --> [(*spatial_dims), T, C]
        """
        if with_batch:
            x_perm = rearrange(x, 'b t c ... -> b ... t c')  # -> [B, *spatial_dims, T, C]
        else:
            x_perm = rearrange(x, 't c ... -> ... t c')
        return x_perm


    def __getitem__(self, idx: int):
        traj_indice, t0 = self.samples[idx]
        data_full = self.data[traj_indice, t0:t0+self.n_frames_train+self.n_frames_out]    # [T, C, (*spatial_dims)]
        indice_end = t0+self.n_frames_train if self.group == "train" else t0+self.n_frames_train+self.n_frames_out

        if self.mode == "interpolation":
            out = {
                "data": self._channel_last(self.data[traj_indice, t0:indice_end]),    # [T, (*spatial_dims), n_ch]
                "t": torch.arange(indice_end - t0, dtype=self.data.dtype) * self.cfg.dt_eval,
                "index": idx,
            }
            # only add mask when it exists
            if self.spatial_dim == 2 and self.mask_tensor is not None:
                out["mask"] = self._channel_last(self.mask_tensor[traj_indice, t0:indice_end])
            """else: 
                out["mask"] = None"""
            return out

        elif self.mode == "autoregressive":
            data_x  = self.data[traj_indice, t0:t0+self.n_frames_cond]
            data_y_ = self.data[traj_indice, t0+self.n_frames_cond:indice_end]
            data_xx = self._stack_time_as_channels(data_x)   # [..., T_cond*C]
            data_y  = self._rearrange_tc(data_y_)            # [..., T_out, C]

            out = {"data": (data_xx, data_y), "index": idx}
            # only add data_full for non-train splits
            if self.group != "train":
                data_full = self.data[traj_indice, t0:t0+self.n_frames_train+self.n_frames_out]
                out["data_full"] = self._rearrange_tc(data_full)     # [..., T_total, C]
            return out


class PDEDataProcessor():
    def __init__(self, data_tensor: torch.Tensor, cfg: Dataloader_Configs,
                 train_ids: Optional[List[int]] = None, test_ids: Optional[List[int]] = None):
        lim = getattr(cfg, "limit_trajs", None)
        if lim is not None:
            lim = int(lim)
            if data_tensor.shape[0] < lim:
                raise ValueError(
                    f"limit_trajs={lim} is greater than available trajectories N={data_tensor.shape[0]}"
                )
            if data_tensor.shape[0] > lim:
                data_tensor = data_tensor[:lim].contiguous()
        # cfg: config for train-test dataloader
        self.dataset = cfg.dataset
        self.data = data_tensor    # [N, T, C, (*spatial_dims)]
        shape = self.data.shape
        self.total_trajs, self.total_len, self.channels = shape[:3]
        self.spatial_dims = tuple(shape[3:]) 
        self.x_grid = make_grid(shape=self.spatial_dims, flatten=False)
        self.cfg = cfg
        self.mode = cfg.mode

        # dataset configs (especially for DINO)
        self.n_frames_train = cfg.n_frames_train
        self.n_frames_cond = cfg.n_frames_cond
        self.n_frames_out = cfg.n_frames_out
        self.mask_ratio = cfg.mask_ratio
        
        self.torch_gen, self.np_gen = seed_everything(cfg.seed)

        self.n_train_trajs, self.n_test_trajs = cfg.n_train_trajs, cfg.n_test_trajs
        assert self.n_train_trajs + self.n_test_trajs == self.total_trajs
        
        # print("Using default train-test traj ids!!!!!!!!!!!")
        if train_ids is not None:
            # print("Using default train-test traj ids!!!!!!!!!!!")
            assert test_ids is not None and (len(train_ids) == self.n_train_trajs) and len(test_ids) == self.n_test_trajs
            self.train_traj_ids = train_ids
            self.test_traj_ids = test_ids
        else:
            # get train-test traj idx
            idx_all = np.arange(self.total_trajs)
            rng = np.random.default_rng(cfg.seed)
            rng.shuffle(idx_all)
            self.train_traj_ids = idx_all[:self.n_train_trajs].tolist()
            self.test_traj_ids = idx_all[self.n_train_trajs:].tolist()

        # ?
        self.cfg.train_traj_ids = self.train_traj_ids
        self.cfg.test_traj_ids = self.test_traj_ids

        self.build_normalizer()
        # Standardize data_tensor first!
        self.data_norm = self.normalizer.encode(self.data) if self.cfg.normalize else self.data

        # generate mask
        is_2d = (len(self.spatial_dims) == 2)
        if is_2d:
            self.mask_tensor = generate_block_mask(
                shape=self.data.shape,
                mask_ratio=cfg.mask_ratio,
                block_size=cfg.block_size,
                same_over_time=cfg.same_over_time,
                device="cpu",
                torch_gen=self.torch_gen,
            )
        else:
            self.mask_tensor = None


    def build_normalizer(self):
        train_data = self.data[self.train_traj_ids]
        self.normalizer = UnitNormalizer.fit(X=train_data)
    

    def compute_samples(self, group: str) -> List[Tuple[int, int]]:
        traj_indices = {
            "train": self.train_traj_ids,
            "train_eval": self.train_traj_ids,
            "test": self.test_traj_ids,
        }[group]

        T = self.total_len
        max_t0 = T - self.n_frames_train - self.n_frames_out
        if max_t0 < 0:
            raise ValueError(
                f"Not enough frames: T={T}, need at least n_frames_train + n_frames_out = "
                f"{self.n_frames_train + self.n_frames_out}."
            )
        rng = np.random.default_rng(self.cfg.seed + (0 if group=="train" else (1 if group=="train_eval" else 2)))
        samples: List[Tuple[int,int]] = []
        if self.cfg.sample_strategy == "random":
            n_per = min(self.cfg.n_samples_per_traj, max_t0 + 1)
            for traj_indice in traj_indices:
                t0s = rng.choice(max_t0+1, size=n_per, replace=False)
                samples.extend((int(traj_indice), int(t0)) for t0 in t0s)
        elif self.cfg.sample_strategy == "disjoint":
            t0s = range(0, max_t0+1, self.n_frames_train + self.n_frames_out)
            n_per = min(self.cfg.n_samples_per_traj, len(t0s))
            for traj_indice in traj_indices:
                samples.extend((int(traj_indice), int(t0)) for t0 in list(t0s)[:n_per])
        else:
            raise ValueError(f"Unknown sample_strategy: {self.cfg.sample_strategy}")
        return samples


    def get_dataloader(self, group="train", record_sample_idx: bool = False,
                       samples: Optional[List[Tuple[int,int]]] = None):
        # group: "train" / "train_eval" / "test"
        traj_indices = {
            "train": self.train_traj_ids,
            "train_eval": self.train_traj_ids,
            "test": self.test_traj_ids,
        }[group]
        if samples is None:
            samples = self.compute_samples(group)
        dataset = PDEDataset(
            data_tensor=self.data_norm, cfg=self.cfg,
            n_frames_train=self.cfg.n_frames_train,
            n_frames_cond=self.cfg.n_frames_cond,
            n_frames_out=self.cfg.n_frames_out,
            traj_indices=traj_indices,
            n_sample_per_traj=self.cfg.n_samples_per_traj,
            sample_strategy=self.cfg.sample_strategy,
            mode=self.cfg.mode,
            group=group,
            samples=samples,
            mask_tensor=self.mask_tensor,
            np_rng=self.np_gen
        )
        batch_size = self.cfg.train_bs if group in ("train", "train_eval") else self.cfg.test_bs
        is_train = (group == "train")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                                num_workers=self.cfg.num_workers, generator=self.torch_gen,
                                pin_memory=True, drop_last=False,
                                # worker_init_fn=make_worker_init_fn(self.cfg.seed), persistent_workers=True)
                                )
        if record_sample_idx:
            return dataloader, dataset.samples
        else:
            return dataloader


def generate_block_mask(
    shape: Tuple[int, int, int, int, int],
    mask_ratio: float,
    block_size: Union[int, Tuple[int, int]],
    same_over_time: bool = True,
    same_over_channel: bool = True,
    device: Union[str, torch.device] = 'cpu',
    torch_gen: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Generate a block-wise binary mask for tensors of shape [B, T, C, H, W].

    Semantics:
      - 1 = keep (unmasked), 0 = drop (masked).
      - Blocks are chosen independently for each (b, c) pair by default.
      - If `same_over_time=True`, the same set of blocks is used for all time steps.
        Otherwise (`False`), a fresh set of blocks is sampled for each time step.
      - If `same_over_channel=True`, the same set of blocks is shared by all channels
        within a (b, t) group (or within (b,) when `same_over_time=True`).

    Notes:
      - If H or W is not divisible by the block size, the trailing area remains unmasked (kept as 1).
      - The number of masked patches is computed as round(mask_ratio * total_patches),
        then clamped to [0, total_patches].
      - For reproducibility, pass a `torch.Generator` via `torch_gen`.

    Args:
      shape: (B, T, C, H, W) of the target tensor to be masked.
      mask_ratio: Fraction of spatial patches to mask per (b, t, c) or per (b, t) if same_over_channel=True
                  (and per (b,) if same_over_time=True).
      block_size: int or (block_h, block_w) for the block dimensions.
      same_over_time: Whether to reuse the same block pattern across time steps.
      same_over_channel: Whether all channels share the same block pattern.
      device: Torch device to create the mask on.
      torch_gen: Optional torch.Generator for deterministic sampling.

    Returns:
      mask: Float tensor of shape [B, T, C, H, W], where 0 marks masked pixels and 1 marks unmasked pixels.
    """
    B, T, C, H, W = shape

    # Parse block size
    if isinstance(block_size, int):
        block_h = block_w = block_size
    else:
        block_h, block_w = block_size

    # Basic checks
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

    # Number of non-overlapping patches that fit along H and W
    n_ph = H // block_h
    n_pw = W // block_w
    total_patches = n_ph * n_pw

    # Compute how many patches to mask
    n_mask = int(round(mask_ratio * total_patches))
    n_mask = min(max(n_mask, 0), total_patches)

    # Patch-level mask: [B, time_dim, C, n_ph, n_pw]
    time_dim = 1 if same_over_time else T
    patch_mask = torch.ones((B, time_dim, C, n_ph, n_pw), dtype=torch.float32, device=device)

    if n_mask > 0 and total_patches > 0:
        if same_over_channel:
            # One set of indices per (b, t) (or per b if same_over_time=True) shared by all channels
            if same_over_time:
                for b in range(B):
                    idx = torch.randperm(total_patches, generator=torch_gen, device=device)[:n_mask]
                    pi = idx // n_pw
                    pj = idx % n_pw
                    patch_mask[b, 0, :, pi, pj] = 0.0
            else:
                for b in range(B):
                    for t in range(T):
                        idx = torch.randperm(total_patches, generator=torch_gen, device=device)[:n_mask]
                        pi = idx // n_pw
                        pj = idx % n_pw
                        patch_mask[b, t, :, pi, pj] = 0.0
        else:
            # Independent indices per (b, c) (and per time step if same_over_time=False)
            for b in range(B):
                for c in range(C):
                    if same_over_time:
                        # One sample shared across all time steps
                        idx = torch.randperm(total_patches, generator=torch_gen, device=device)[:n_mask]
                        pi = idx // n_pw
                        pj = idx % n_pw
                        patch_mask[b, 0, c, pi, pj] = 0.0
                    else:
                        # Resample independently per time step
                        for t in range(T):
                            idx = torch.randperm(total_patches, generator=torch_gen, device=device)[:n_mask]
                            pi = idx // n_pw
                            pj = idx % n_pw
                            patch_mask[b, t, c, pi, pj] = 0.0

    # Expand from patch grid to pixel grid
    mask = (
        patch_mask
        .repeat_interleave(block_h, dim=-2)
        .repeat_interleave(block_w, dim=-1)
    )  # [B, time_dim, C, n_ph*block_h, n_pw*block_w]

    # Pad trailing pixels (if H/W not divisible by block size) with 1s (unmasked)
    if mask.shape[-2] < H:
        mask = F.pad(mask, (0, 0, 0, H - mask.shape[-2]), value=1.0)
    if mask.shape[-1] < W:
        mask = F.pad(mask, (0, W - mask.shape[-1], 0, 0), value=1.0)

    # If we used a time_dim=1 buffer for same_over_time, broadcast across T
    if same_over_time:
        mask = mask.expand(B, T, C, H, W).contiguous()
    else:
        # time_dim == T already
        mask = mask.contiguous()

    return mask