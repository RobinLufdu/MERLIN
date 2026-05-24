import json
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Subset

from data.data_process import PDEDataset


_LOADER_ATTRS = {
    "train": "train_loader",
    "train_eval": "train_eval_loader",
    "test": "test_loader",
}


def _loader_attr(group: str) -> str:
    try:
        return _LOADER_ATTRS[group]
    except KeyError as exc:
        valid = ", ".join(sorted(_LOADER_ATTRS))
        raise ValueError(f"Unknown loader group '{group}'. Expected one of: {valid}") from exc


def get_loader(exp: Any, group: str):
    """Return the dataloader attribute for a split without constructing it."""
    return getattr(exp, _loader_attr(group), None)


def ensure_loader(exp: Any, group: str) -> None:
    """Construct a split dataloader on demand if it is not already available."""
    if get_loader(exp, group) is None:
        exp.build_dataloader(group=group)


def get_dataset(exp: Any, group: str):
    """Return a split dataset, constructing the corresponding loader if needed."""
    ensure_loader(exp, group)
    loader = get_loader(exp, group)
    if loader is None:
        raise RuntimeError(f"Failed to build loader for group '{group}'")
    return loader.dataset


def sample_batch(exp: Any, group: str, batch_size: int, replace: bool = False):
    """
    Randomly sample one temporary batch from a split, independent of the
    split dataloader's configured batch size.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    dataset = get_dataset(exp, group)
    n_samples = len(dataset)
    if n_samples == 0:
        raise RuntimeError(f"Cannot sample from empty dataset for group '{group}'")
    if batch_size > n_samples:
        replace = True

    np_gen = getattr(exp.data_processor, "np_gen", None)
    if np_gen is None:
        np_gen = np.random.default_rng(exp.cfg.seed)

    indices = np_gen.choice(n_samples, size=batch_size, replace=replace).tolist()
    subset = Subset(dataset, indices)
    tmp_loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return next(iter(tmp_loader))


def make_subset_from_saved(
    exp: Any,
    group: str,
    saved_json_path: str,
    batch_size: int | None = None,
):
    """
    Build a temporary DataLoader containing the exact (traj_id, t0) pairs
    stored for `group` in a split metadata JSON.
    """
    dataset = get_dataset(exp, group)

    with open(saved_json_path, "r") as f:
        payload = json.load(f)

    idx_list = payload.get("samples", {}).get(group, None)
    if idx_list is None:
        raise ValueError(f"No saved indices found for group '{group}' in {saved_json_path}")

    if not hasattr(dataset, "samples"):
        raise AttributeError(f"Dataset for group '{group}' does not expose a samples attribute")

    position = {(int(tid), int(t0)): i for i, (tid, t0) in enumerate(dataset.samples)}
    take = []
    for tid, t0 in idx_list:
        key = (int(tid), int(t0))
        if key not in position:
            raise KeyError(f"Sample {key} not present in current dataset.samples")
        take.append(position[key])

    subset = Subset(dataset, take)
    if batch_size is None:
        batch_size = len(take)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def sample_from_fix(exp: Any, traj_id: int, t0: int, rollout_steps: int | None = None):
    """Build a one-sample test loader for a fixed trajectory and start time."""
    p = exp.data_processor
    cfg = p.cfg
    traj_id = int(traj_id)
    samples = [(traj_id, int(t0))]
    data_tensor = p.data_norm if cfg.normalize else p.data
    n_frames_out = (
        cfg.n_frames_out
        if rollout_steps is None
        else rollout_steps - cfg.n_frames_train + cfg.n_frames_cond
    )

    dataset = PDEDataset(
        data_tensor=data_tensor,
        cfg=cfg,
        n_frames_train=cfg.n_frames_train,
        n_frames_cond=cfg.n_frames_cond,
        n_frames_out=n_frames_out,
        traj_indices=[traj_id],
        n_sample_per_traj=1,
        sample_strategy="disjoint",
        mode="interpolation",
        group="test",
        samples=samples,
        mask_tensor=p.mask_tensor,
        np_rng=p.np_gen,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


__all__ = [
    "ensure_loader",
    "get_dataset",
    "get_loader",
    "make_subset_from_saved",
    "sample_batch",
    "sample_from_fix",
]
