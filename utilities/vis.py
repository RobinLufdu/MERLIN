# -*- coding: utf-8 -*-
"""
Visualize prediction vs ground-truth for arrays shaped (B, H, W, T, C).

- mode="last": last time step only (rows = channels; cols = prediction / ground-truth [/error])
- mode="all" : export all time steps as GIF/MP4 or per-frame PNGs
- Optional error column: absolute |pred-gt| or signed (pred-gt)
- Per-row colorbars placed OUTSIDE axes (append_axes) so they never cover images
- Two dedicated header rows (time label + column titles) so nothing overlaps

Tunable layout (now callable from visualize_pred_vs_gt):
- header_time_ratio / header_cols_ratio: header row heights (fraction of a content row height)
- fig_w_per_col / fig_h_per_row: overall figure size scalers
- wspace / hspace / left / right / bottom: subplot spacing and margins
- time_fontsize / coltitle_fontsize: typography sizes
"""

import os
import math
import warnings
from typing import Literal, Optional, Tuple, Union, List

# Headless backend (for servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  # external (non-overlapping) colorbars

# Clean typography & spacing (built-in fonts only)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "axes.titlepad": 10.0,
    "axes.labelsize": 10,
    "figure.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 140,
    "savefig.dpi": 140,
})

import numpy as np

# Optional torch support
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Optional imageio v3 for GIF/MP4 export
try:
    import imageio.v3 as iio  # imageio>=2.22 provides the v3 API
except Exception:
    iio = None

warnings.filterwarnings("ignore")

ArrayLike = Union[np.ndarray, "torch.Tensor"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert torch.Tensor / np.ndarray to numpy.float32 without modifying the input."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.dtype.kind in "ui":
        x = x.astype(np.float32)
    elif x.dtype.kind == "f":
        x = x.astype(np.float32, copy=False)
    else:
        x = x.astype(np.float32)
    return x


def _ensure_dir(path: str):
    """Create parent directory for a file path (or the directory itself) if it doesn't exist."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _compute_channel_ranges(
    pred0: np.ndarray,
    gt0: np.ndarray,
    across_time: bool
) -> List[Tuple[float, float]]:
    """Compute per-channel (vmin, vmax) for consistent color scales."""
    assert pred0.shape == gt0.shape, "pred0 and gt0 must have identical shapes"
    assert pred0.ndim in (3, 4), "Expected (H, W, C) or (H, W, T, C)"
    if pred0.ndim == 4:
        _, _, _, C = pred0.shape
    else:
        _, _, C = pred0.shape

    ranges = []
    for c in range(C):
        if pred0.ndim == 4 and across_time:
            p = pred0[..., c].reshape(-1)
            g = gt0[..., c].reshape(-1)
        else:
            p = pred0[..., c].ravel()
            g = gt0[..., c].ravel()
        p = p[np.isfinite(p)]
        g = g[np.isfinite(g)]
        if p.size == 0 and g.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanmin([p.min() if p.size else np.inf, g.min() if g.size else np.inf]))
            vmax = float(np.nanmax([p.max() if p.size else -np.inf, g.max() if g.size else -np.inf]))
            if not np.isfinite(vmin): vmin = 0.0
            if not np.isfinite(vmax): vmax = 1.0
        if math.isclose(vmin, vmax, rel_tol=0, abs_tol=1e-12):
            eps = 1e-6 if vmin == 0 else abs(vmin) * 1e-6
            vmin, vmax = vmin - eps, vmax + eps
        ranges.append((vmin, vmax))
    return ranges


def _compute_error_ranges(
    pred0: np.ndarray,
    gt0: np.ndarray,
    across_time: bool,
    err_type: Literal["abs", "diff"] = "abs"
) -> List[Tuple[float, float]]:
    """Compute per-channel (vmin, vmax) for the error map."""
    assert pred0.shape == gt0.shape
    assert pred0.ndim in (3, 4)
    if pred0.ndim == 4:
        _, _, _, C = pred0.shape
    else:
        _, _, C = pred0.shape

    ranges = []
    for c in range(C):
        diff = (pred0[..., c] - gt0[..., c])
        diff = diff[np.isfinite(diff)]
        if diff.size == 0:
            ranges.append((0.0, 1.0))
            continue
        if err_type == "abs":
            vmax = float(np.nanmax(np.abs(diff)))
            vmax = vmax if vmax > 1e-12 else 1e-6
            ranges.append((0.0, vmax))
        else:
            m = float(np.nanmax(np.abs(diff)))
            m = m if m > 1e-12 else 1e-6
            ranges.append((-m, m))
    return ranges


# ──────────────────────────────────────────────────────────────────────────────
# Drawing (external colorbars, two-row header)
# ──────────────────────────────────────────────────────────────────────────────

def _draw_grid(
    pred_img: np.ndarray,
    gt_img: np.ndarray,
    ch_ranges: List[Tuple[float, float]],
    *,
    include_error: bool = True,
    err_img: Optional[np.ndarray] = None,
    err_ranges: Optional[List[Tuple[float, float]]] = None,
    err_type: Literal["abs", "diff"] = "abs",
    cmap: str = "viridis",
    err_cmap: Optional[str] = None,
    # Global time label drawn in a dedicated header row (LaTeX mathtext supported)
    time_label: Optional[str] = None,     # e.g. r"$t=7$"
    dpi: int = 140,
    add_colorbar: bool = True,
    # Layout knobs
    left: float = 0.06,
    right: float = 0.985,
    bottom: float = 0.08,
    wspace: float = 0.10,     # column spacing
    hspace: float = 0.10,     # row spacing
    # Two-header-row sizes (as a fraction of a content row)
    header_time_ratio: float = 0.40,      # time label row height
    header_cols_ratio: float = 0.38,      # column titles row height
    # Easy size controls (inches per column/row). Increase to make everything bigger.
    fig_w_per_col: float = 2.4,
    fig_h_per_row: float = 1.45,
    # Typography
    time_fontsize: int = 13,
    coltitle_fontsize: int = 12,
    use_tex: bool = False,    # True -> requires system LaTeX
):
    """
    Draw a grid with 2 columns (prediction/ground-truth) or 3 columns (+error).
    We build a figure with TWO header rows (time label + column titles) above all content rows.
    Colorbars are appended OUTSIDE axes, never overlaying images.

    The gap between the time label and content is primarily controlled by
    `header_time_ratio` (smaller -> closer) and `header_cols_ratio`.
    """
    assert pred_img.shape == gt_img.shape and pred_img.ndim == 3, "Inputs must be (H, W, C)"
    _, _, C = pred_img.shape
    ncols = 3 if include_error else 2

    if include_error:
        if err_img is None:
            diff = pred_img - gt_img
            err_img = np.abs(diff) if err_type == "abs" else diff
        else:
            assert err_img.shape == pred_img.shape
        assert err_ranges is not None, "err_ranges must be provided when include_error=True"

    if use_tex:
        import matplotlib as _mpl
        _mpl.rcParams["text.usetex"] = True

    # --- Figure size (inches): scalable by per-col / per-row factors ---
    fig_w = fig_w_per_col * ncols
    fig_h = fig_h_per_row * (C + header_time_ratio + header_cols_ratio)

    # Gridspec with 2 header rows + C content rows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=C + 2, ncols=ncols,
        height_ratios=[header_time_ratio, header_cols_ratio] + [1] * C,
        left=left, right=right, bottom=bottom, top=0.98,
        wspace=wspace, hspace=hspace
    )

    # Header row 1: time label
    ax_time = fig.add_subplot(gs[0, :]); ax_time.axis("off")
    if time_label:
        ax_time.text(0.5, 0.5, time_label, ha="center", va="center", fontsize=time_fontsize)

    # Header row 2: column titles (do NOT use axes titles on content axes)
    ax_cols = fig.add_subplot(gs[1, :]); ax_cols.axis("off")
    col_names = ["prediction", "ground-truth"] + (
        ["error (|pred-gt|)"] if include_error and err_type == "abs"
        else ["error (pred-gt)"] if include_error else []
    )
    for i, name in enumerate(col_names):
        x = (i + 0.5) / ncols  # center of each column
        ax_cols.text(x, 0.5, name, ha="center", va="center",
                     fontsize=coltitle_fontsize, fontweight="semibold")

    # Build content axes array (C rows × ncols)
    axes = np.empty((C, ncols), dtype=object)
    for r in range(C):
        for c in range(ncols):
            axes[r, c] = fig.add_subplot(gs[r + 2, c])

    if err_cmap is None:
        err_cmap = "magma" if err_type == "abs" else "coolwarm"

    # Draw rows
    for c in range(C):
        vmin, vmax = ch_ranges[c]
        ax_pred = axes[c, 0]
        ax_gt   = axes[c, 1]

        im_pred = ax_pred.imshow(pred_img[..., c], vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
        ax_pred.set_xticks([]); ax_pred.set_yticks([])
        ax_pred.set_ylabel(f"channel {c}", rotation=90, va="center")

        im_gt = ax_gt.imshow(gt_img[..., c], vmin=vmin, vmax=vmax, cmap=cmap, interpolation="nearest")
        ax_gt.set_xticks([]); ax_gt.set_yticks([])

        if add_colorbar:
            div_gt = make_axes_locatable(ax_gt)
            cax = div_gt.append_axes("right", size="3.0%", pad=0.04)
            cb = fig.colorbar(im_pred, cax=cax)
            cb.ax.tick_params(labelsize=8, length=2)

        if include_error:
            ax_err = axes[c, 2]
            evmin, evmax = err_ranges[c]
            im_err = ax_err.imshow(err_img[..., c], vmin=evmin, vmax=evmax, cmap=err_cmap, interpolation="nearest")
            ax_err.set_xticks([]); ax_err.set_yticks([])
            if add_colorbar:
                div_err = make_axes_locatable(ax_err)
                cax2 = div_err.append_axes("right", size="3.0%", pad=0.04)
                cb2 = fig.colorbar(im_err, cax=cax2)
                cb2.ax.tick_params(labelsize=8, length=2)

    return fig, axes


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def visualize_pred_vs_gt(
    pred: ArrayLike,
    gt: ArrayLike,
    mode: Literal["last", "all"] = "last",
    batch_index: int = 0,
    save_path: Optional[str] = None,
    *,
    # Visualization options
    cmap: str = "viridis",
    fps: int = 4,
    as_type: Literal["gif", "mp4", "frames_auto"] = "gif",
    dpi: int = 140,
    show_error: bool = True,
    err_type: Literal["abs", "diff"] = "abs",
    add_colorbar: bool = True,
    # Layout knobs exposed to the caller (tweak these to change the gap)
    left: float = 0.06,
    right: float = 0.985,
    bottom: float = 0.08,
    wspace: float = 0.10,
    hspace: float = 0.10,
    header_time_ratio: float = 0.40,
    header_cols_ratio: float = 0.38,
    fig_w_per_col: float = 2.4,
    fig_h_per_row: float = 1.45,
    time_fontsize: int = 13,
    coltitle_fontsize: int = 12,
    use_tex: bool = False,
) -> Optional[str]:
    """
    Visualize prediction vs ground-truth for inputs shaped (B, H, W, T, C).
    Returns the save path (file or directory) if saved, otherwise None.

    To bring the time label closer to the plots, LOWER `header_time_ratio`
    (e.g., 0.30). To change space between the column titles and content, tweak
    `header_cols_ratio`. Increase `fig_h_per_row` to make everything taller.
    """
    pred = _to_numpy(pred)
    gt   = _to_numpy(gt)

    assert pred.shape == gt.shape, "pred and gt must have the same shape"
    assert pred.ndim == 5, "Expected input shape (B, H, W, T, C)"
    B, H, W, T, C = pred.shape
    assert 0 <= batch_index < B, f"batch_index out of range: 0 ≤ idx < {B}"

    pred0 = pred[batch_index]  # (H, W, T, C)
    gt0   = gt[batch_index]    # (H, W, T, C)

    if mode == "last":
        # Last time step only
        pred_last = pred0[:, :, -1, :]
        gt_last   = gt0[:, :, -1, :]
        ch_ranges = _compute_channel_ranges(pred_last, gt_last, across_time=False)
        err_ranges = _compute_error_ranges(pred_last, gt_last, across_time=False, err_type=err_type) if show_error else None

        time_label = rf"$t={T-1}$"  # mathtext (no system LaTeX required)
        fig, _ = _draw_grid(
            pred_last, gt_last, ch_ranges,
            include_error=show_error,
            err_ranges=err_ranges,
            err_type=err_type,
            cmap=cmap,
            time_label=time_label,
            dpi=dpi,
            add_colorbar=add_colorbar,
            # Pass-through layout/typography knobs
            left=left, right=right, bottom=bottom,
            wspace=wspace, hspace=hspace,
            header_time_ratio=header_time_ratio,
            header_cols_ratio=header_cols_ratio,
            fig_w_per_col=fig_w_per_col,
            fig_h_per_row=fig_h_per_row,
            time_fontsize=time_fontsize,
            coltitle_fontsize=coltitle_fontsize,
            use_tex=use_tex,
        )

        if save_path:
            _ensure_dir(save_path)
            ext = os.path.splitext(save_path)[1].lower()
            if ext not in (".png", ".pdf", ".jpg", ".jpeg", ".svg"):
                save_path = save_path + ".png"
            fig.savefig(save_path)  # keep margins for consistent centering
            plt.close(fig)
            return save_path
        else:
            plt.show()
            return None

    elif mode == "all":
        # Full sequence export
        ch_ranges  = _compute_channel_ranges(pred0, gt0, across_time=True)
        err_ranges = _compute_error_ranges(pred0, gt0, across_time=True, err_type=err_type) if show_error else None

        if as_type in ("gif", "mp4"):
            assert iio is not None, "imageio is required: `pip install imageio imageio-ffmpeg`"
            assert save_path is not None, "Please provide save_path for sequence export"
            _ensure_dir(save_path)
            suf = os.path.splitext(save_path)[1].lower()
            if as_type == "gif" and suf != ".gif":
                save_path = save_path + ".gif"
            if as_type == "mp4" and suf != ".mp4":
                save_path = save_path + ".mp4"

            frames = []
            for t in range(T):
                pred_t = pred0[:, :, t, :]
                gt_t   = gt0[:, :, t, :]
                time_label = rf"$t={t}$"

                fig, _ = _draw_grid(
                    pred_t, gt_t, ch_ranges,
                    include_error=show_error,
                    err_ranges=err_ranges,
                    err_type=err_type,
                    cmap=cmap,
                    time_label=time_label,
                    dpi=dpi,
                    add_colorbar=add_colorbar,
                    left=left, right=right, bottom=bottom,
                    wspace=wspace, hspace=hspace,
                    header_time_ratio=header_time_ratio,
                    header_cols_ratio=header_cols_ratio,
                    fig_w_per_col=fig_w_per_col,
                    fig_h_per_row=fig_h_per_row,
                    time_fontsize=time_fontsize,
                    coltitle_fontsize=coltitle_fontsize,
                    use_tex=use_tex,
                )

                # Render the Matplotlib figure into an RGB array (backend-agnostic)
                fig.canvas.draw()
                w, h = fig.canvas.get_width_height()
                if hasattr(fig.canvas, "buffer_rgba"):
                    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
                    img = buf[..., :3].copy()  # drop alpha -> RGB
                else:
                    rgb = fig.canvas.tostring_rgb()
                    img = np.frombuffer(rgb, dtype=np.uint8).reshape(h, w, 3)
                frames.append(img)
                plt.close(fig)

            if as_type == "gif":
                iio.imwrite(save_path, frames, duration=1000 / max(1, fps))  # ms per frame
            else:
                iio.imwrite(save_path, frames, fps=max(1, fps))
            return save_path

        else:  # frames_auto -> per-frame PNGs
            assert save_path is not None, "Please provide save_path (directory or prefix)"
            frames_dir = os.path.join(save_path, "frames") if not save_path.endswith(os.sep) else os.path.join(save_path, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            for t in range(T):
                pred_t = pred0[:, :, t, :]
                gt_t   = gt0[:, :, t, :]
                time_label = rf"$t={t}$"

                fig, _ = _draw_grid(
                    pred_t, gt_t, ch_ranges,
                    include_error=show_error,
                    err_ranges=err_ranges,
                    err_type=err_type,
                    cmap=cmap,
                    time_label=time_label,
                    dpi=dpi,
                    add_colorbar=add_colorbar,
                    left=left, right=right, bottom=bottom,
                    wspace=wspace, hspace=hspace,
                    header_time_ratio=header_time_ratio,
                    header_cols_ratio=header_cols_ratio,
                    fig_w_per_col=fig_w_per_col,
                    fig_h_per_row=fig_h_per_row,
                    time_fontsize=time_fontsize,
                    coltitle_fontsize=coltitle_fontsize,
                    use_tex=use_tex,
                )
                p = os.path.join(frames_dir, f"frame_{t:04d}.png")
                fig.savefig(p)   # keep margins so frames align in videos
                plt.close(fig)
            return frames_dir

    else:
        raise ValueError("mode must be 'last' or 'all'")


# ──────────────────────────────────────────────────────────────────────────────
# Minimal self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Fake data: (B, H, W, T, C) = (2, 64, 64, 8, 1)
    B, H, W, T, C = 2, 64, 64, 8, 1
    rng = np.random.default_rng(0)
    gt  = rng.normal(size=(B, H, W, T, C)).astype(np.float32)
    pred = gt + 0.3 * rng.normal(size=(B, H, W, T, C)).astype(np.float32)

    # Example 1: default spacing
    out_png = visualize_pred_vs_gt(
        pred, gt,
        mode="last", batch_index=0,
        save_path="outputs/last_step_default",
        show_error=True, err_type="abs", add_colorbar=True
    )
    print("Saved last-step (default) to:", out_png)

    # Example 2: tighter gap between time label and content
    out_png_tight = visualize_pred_vs_gt(
        pred, gt,
        mode="last", batch_index=0,
        save_path="outputs/last_step_tight",
        show_error=True, err_type="abs", add_colorbar=True,
        header_time_ratio=0.28,   # ↓ smaller -> closer
        header_cols_ratio=0.34,   # optionally shrink the column-title row too
        time_fontsize=12          # slightly smaller time label
    )
    print("Saved last-step (tight header) to:", out_png_tight)

    # Full sequence as GIF with tighter headers
    out_gif = visualize_pred_vs_gt(
        pred, gt,
        mode="all", batch_index=0,
        save_path="outputs/sequence.gif",
        as_type="gif", fps=3,
        show_error=True, err_type="abs", add_colorbar=True,
        header_time_ratio=0.28, header_cols_ratio=0.34
    )
    print("Saved GIF to:", out_gif)

    # Per-frame PNGs
    frames_dir = visualize_pred_vs_gt(
        pred, gt,
        mode="all", batch_index=0,
        save_path="./outputs/",
        as_type="frames_auto", fps=4,
        show_error=True, err_type="diff", add_colorbar=True,
        header_time_ratio=0.28, header_cols_ratio=0.34
    )
    print("Saved frames under:", frames_dir)



