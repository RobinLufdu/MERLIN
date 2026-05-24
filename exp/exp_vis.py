"""Visualization and figure-export helpers for MERLIN experiments.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from utilities.vis import visualize_pred_vs_gt

@torch.no_grad()
def visualize_latent_pca2d(
    exp,
    group: str,
    n_traj: int = 16,
    max_time: int | None = None,
    save_path: str | None = None,
    time_style: str = "color+arrows",
    arrow_every: int = 5,
    show_colorbar: bool = True,
    center: str = "none",
    overlay_compare: bool = False,
):
    """Visualize encoded latent trajectories in a PCA-2D plane."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from torch.utils.data import DataLoader, Subset

    assert time_style in {"color", "arrows", "color+arrows"}
    assert center in {"none", "zstar", "dataset"}

    exp._ensure_loader(group)
    loader = {
        "train": exp.train_loader,
        "train_eval": exp.train_eval_loader,
        "test": exp.test_loader,
    }[group]
    dataset = loader.dataset
    n_traj = min(n_traj, len(dataset))
    idxs = np.arange(len(dataset))[:n_traj].tolist()
    tmp_loader = DataLoader(Subset(dataset, idxs), batch_size=1, shuffle=False, num_workers=0)

    lat_list = []
    for batch in tmp_loader:
        lat, _, _ = exp._encode_and_recon(batch)
        z = lat[:, 0].detach().cpu().float()
        if max_time is not None:
            z = z[:max_time]
        lat_list.append(z)

    if not lat_list:
        print("[visualize_latent_pca2d] no data.")
        return None

    X = torch.cat(lat_list, dim=0).float()
    mu_ds = X.mean(dim=0, keepdim=True)
    Xc = X - mu_ds
    C = Xc.T @ Xc / max(1, Xc.size(0) - 1)
    _, vecs = torch.linalg.eigh(C)
    W = vecs[:, -2:]

    zstar = getattr(exp, "latent_center", None)
    use_zstar = (center == "zstar") and (zstar is not None) and (zstar.numel() == X.shape[1])

    def _centerize(Z: torch.Tensor, how: str) -> torch.Tensor:
        if how == "none":
            return Z
        if how == "dataset":
            return Z - mu_ds
        if how == "zstar" and use_zstar:
            return Z - zstar.detach().cpu().view(1, -1)
        return Z

    fig, ax = plt.subplots()
    Tmax = max(z.size(0) for z in lat_list)

    def _plot_one(Z_np, label=None, style=None):
        if len(Z_np) < 2:
            ax.scatter(Z_np[0, 0], Z_np[0, 1], s=30, label=label)
            return
        if "color" in time_style:
            seg = np.stack([Z_np[:-1], Z_np[1:]], axis=1)
            lc = LineCollection(seg, array=np.arange(seg.shape[0]), cmap="viridis", linewidth=1.5)
            ax.add_collection(lc)
        else:
            ax.plot(Z_np[:, 0], Z_np[:, 1], linewidth=1.0, alpha=0.9, label=label, linestyle=style or "-")
        if "arrows" in time_style and len(Z_np) >= 2 and arrow_every and arrow_every > 0:
            idx = np.arange(0, len(Z_np) - 1, arrow_every)
            for i in idx:
                ax.annotate(
                    "",
                    xy=(Z_np[i + 1, 0], Z_np[i + 1, 1]),
                    xytext=(Z_np[i, 0], Z_np[i, 1]),
                    arrowprops=dict(arrowstyle="->", lw=0.8, alpha=0.9),
                )
        ax.scatter(Z_np[0, 0], Z_np[0, 1], s=32)
        ax.scatter(Z_np[-1, 0], Z_np[-1, 1], s=32, marker="x")

    do_overlay = overlay_compare and (center != "none")
    for z in lat_list:
        z_unc = z
        z_ctr = _centerize(z, center)
        Zu = (z_unc - mu_ds).mm(W).numpy()
        Zc = (z_ctr - mu_ds).mm(W).numpy()

        if do_overlay:
            _plot_one(Zu, label=None, style="--")
            _plot_one(Zc, label=None, style="-")
        else:
            _plot_one(Zc if center != "none" else Zu, label=None, style="-")

    if "color" in time_style and show_colorbar and Tmax >= 2:
        import matplotlib as mpl

        norm = mpl.colors.Normalize(vmin=0, vmax=Tmax - 2)
        sm = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax)
        cb.set_label("time step")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ttl = f"PCA(alpha_t) group={group}, n_traj={len(lat_list)}, center={center}"
    if do_overlay:
        ttl += " (overlay)"
    ax.set_title(ttl)
    ax.axhline(0.0, linewidth=0.5)
    ax.axvline(0.0, linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print("Saved latent PCA plot to:", save_path)
    plt.close(fig)
    return {"n_traj": len(lat_list), "center": center, "save_path": save_path}

def plot_Ad_spectrum(
    exp,
    Ad: torch.Tensor | None = None,
    ckpt_path: str | None = None,
    key: str = "Ad_phase1",
    save_dir: str | None = None,
    fname_prefix: str = "Ad_spectrum",
    *,
    # visibility knobs (unchanged except defaults you set before)
    s_min: float = 14.0,
    s_max: float = 110.0,
    alpha_min: float = 0.55,
    alpha_max: float = 0.95,
    dist_percentile: float = 95.0,
    dist_gamma: float = 0.7,
    figsize_complex=(6.2, 6.2),
    figsize_hist=(6.2, 3.6),
    hist_bins: int | None = None,
    save_matrix: bool = True,
    save_numpy_copy: bool = True,
    # --- new style knobs (for “four spines + bigger ticks/caption”) ---
    spine_lw: float = 1.2,        # linewidth for all four spines
    tick_labelsize: int = 14,     # axis tick numbers size
    axis_labelsize: int = 15,     # axis label size
    rho_fontsize: int = 15,       # caption font size for ρ(Ad)
    dpi: int = 450
):
    """
    Complex plane: hue = arg(λ) (cyclic scientific colormap), size/alpha ∝ |λ−1|.
    Unit circle = solid, thicker; spectral-radius circle dashed; ρ(Ad) annotated.
    No title, bold LaTeX axis labels, no colorbar. Also saves Ad to disk.
    """
    import os, numpy as np, matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib import rcParams
    from matplotlib.cm import get_cmap
    import torch

    # --- Resolve / validate Ad ---
    if Ad is None:
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if key not in ckpt:
                raise KeyError(f"Key '{key}' not found in checkpoint: {ckpt_path}")
            Ad = ckpt[key]
        elif hasattr(exp, "Ad_phase1") and (exp.Ad_phase1 is not None):
            Ad = exp.Ad_phase1
        else:
            raise ValueError("Ad is None and neither ckpt nor cached Ad found.")
    Ad = Ad.detach().to("cpu", dtype=torch.float64)
    if Ad.dim() != 2 or Ad.size(0) != Ad.size(1):
        raise ValueError(f"Ad must be square [D,D], got {tuple(Ad.shape)}")
    D = Ad.size(0)

    # Save matrix copies
    out_paths = []
    if save_dir and save_matrix:
        os.makedirs(save_dir, exist_ok=True)
        pt_path = os.path.join(save_dir, f"{fname_prefix}_Ad.pt")
        torch.save(Ad, pt_path); out_paths.append(pt_path)
        if save_numpy_copy:
            npy_path = os.path.join(save_dir, f"{fname_prefix}_Ad.npy")
            np.save(npy_path, Ad.numpy()); out_paths.append(npy_path)

    # --- Spectrum ---
    lam = torch.linalg.eigvals(Ad).cpu().numpy()      # complex128
    if lam.size == 0:
        raise RuntimeError("No eigenvalues found.")
    re, im = lam.real, lam.imag
    mod = np.abs(lam)
    rho = float(mod.max())

    # Hue by phase (cyclic)
    phase = np.angle(lam)                             # [-π, π]
    phase_norm = (phase + np.pi) / (2.0 * np.pi)      # [0, 1]
    try:
        cmap = get_cmap("twilight_shifted")
    except Exception:
        cmap = get_cmap("twilight")
    colors = cmap(phase_norm)                         # RGBA
    colors[:, :3] = np.clip(colors[:, :3] * 0.85, 0, 1)  # slightly darken

    # Size/alpha by |λ−1|
    dist1 = np.abs(lam - 1.0)
    scale = float(np.percentile(dist1, dist_percentile)) or 1.0
    dist_norm = np.clip((dist1 / scale) ** dist_gamma, 0.0, 1.0)
    sizes  = s_min + (s_max - s_min) * dist_norm
    alphas = alpha_min + (alpha_max - alpha_min) * dist_norm
    colors[:, 3] = alphas

    # --- Global style (keep grid; enable LaTeX math) ---
    rcParams.update({
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.22,
        "axes.formatter.use_mathtext": True,
    })

    # ======================= Complex plane =======================
    fig1, ax1 = plt.subplots(figsize=figsize_complex)

    # Four spines visible and thickened
    for side in ("left", "right", "top", "bottom"):
        ax1.spines[side].set_visible(True)
        ax1.spines[side].set_linewidth(spine_lw)

    # Larger tick labels (and minor ticks for niceness)
    ax1.tick_params(axis="both", which="major", labelsize=tick_labelsize, direction="out", length=4.5, width=0.9)
    ax1.tick_params(axis="both", which="minor", labelsize=tick_labelsize-1, direction="out", length=3.0, width=0.7)
    ax1.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    ax1.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))

    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    # Unit circle
    ax1.plot(np.cos(theta), np.sin(theta),
            linestyle="-", linewidth=2.0, color="#222222", zorder=1)
    # Spectral-radius circle
    ax1.plot(rho * np.cos(theta), rho * np.sin(theta),
            linestyle="--", linewidth=1.6, color="#c44e52", zorder=1)

    # Eigenvalues
    ax1.scatter(re, im,
                s=sizes, c=colors,
                linewidths=0.25, edgecolors=(0, 0, 0, 0.18), zorder=3)

    # Axes & limits
    ax1.axhline(0.0, linewidth=0.8, alpha=0.6, color="#4c4c4c")
    ax1.axvline(0.0, linewidth=0.8, alpha=0.6, color="#4c4c4c")
    ax1.set_aspect("equal", adjustable="box")
    R = 1.05 * max(1.0,
                np.max(np.abs(re)) if re.size else 1.0,
                np.max(np.abs(im)) if im.size else 1.0,
                rho)
    ax1.set_xlim([-R, R]); ax1.set_ylim([-R, R])

    # Bold labels; bigger font sizes
    ax1.set_xlabel(r"$\mathbf{Re}(\lambda)$", fontweight="bold", fontsize=axis_labelsize)
    ax1.set_ylabel(r"$\mathbf{Im}(\lambda)$", fontweight="bold", fontsize=axis_labelsize)

    # Spectral radius caption (bigger)
    ax1.text(
        0.02, 0.98, rf"$\rho(\mathbf{{A}}_d) = {rho:.4f}$",
        transform=ax1.transAxes, ha="left", va="top",
        fontsize=rho_fontsize,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                edgecolor="#c44e52", alpha=0.95)
    )

    if save_dir:
        p1 = os.path.join(save_dir, f"{fname_prefix}_complex.png")
        fig1.savefig(p1, dpi=dpi, bbox_inches="tight")
        out_paths.append(p1)

    # ======================= |λ| histogram =======================
    fig2, ax2 = plt.subplots(figsize=figsize_hist)

    # Four spines visible and thickened
    for side in ("left", "right", "top", "bottom"):
        ax2.spines[side].set_visible(True)
        ax2.spines[side].set_linewidth(spine_lw)

    # Larger tick labels & minor ticks
    ax2.tick_params(axis="both", which="major", labelsize=tick_labelsize, direction="out", length=4.5, width=0.9)
    ax2.tick_params(axis="both", which="minor", labelsize=tick_labelsize-1, direction="out", length=3.0, width=0.7)
    ax2.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    ax2.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))

    nbins = int(np.clip(D // 2, 16, 60)) if hist_bins is None else int(hist_bins)
    counts, edges = np.histogram(mod, bins=nbins)
    ax2.step(edges[:-1], counts, where="post", linewidth=1.6, color="#4c72b0")
    ax2.fill_between(edges[:-1], counts, step="post", alpha=0.15, color="#4c72b0")
    ax2.axvline(1.0, linestyle="-", linewidth=1.2, color="#222222")   # unit radius
    ax2.axvline(rho,  linestyle="--", linewidth=1.2, color="#c44e52") # spectral radius

    ax2.set_xlabel(r"$|\lambda|$", fontweight="bold", fontsize=axis_labelsize)
    ax2.set_ylabel("count", fontweight="bold", fontsize=axis_labelsize)

    if save_dir:
        p2 = os.path.join(save_dir, f"{fname_prefix}_abs_hist.png")
        fig2.savefig(p2, dpi=dpi, bbox_inches="tight")
        out_paths.append(p2)

    plt.close(fig1); plt.close(fig2)
    return {"rho": rho, "paths": out_paths}

@torch.no_grad()
def latent_channel_diagnostics(
    exp,
    group: str,
    n_traj: int = 64,
    max_time: int | None = None,
    center: str = "none",          # "none" | "zstar" | "dataset"
    save_dir: str | None = None,
    topk: int = 10,
):
    """
    Compute per-dimension stats (mean/std/RMS) and inter-channel relation (corr / 1-|corr|).
    Also quantifies how centering changes per-step norms if center="zstar" or "dataset".

    - group:     "train" | "train_eval" | "test"
    - n_traj:    number of sequences to use (each with batch_size=1 for stable masks)
    - max_time:  truncate each sequence if provided
    - center:    "none": as-is; "zstar": subtract exp.latent_center; "dataset": subtract dataset-wide mean
    - save_dir:  if given, save heatmaps & bar plots
    - topk:      print top-k most similar/dissimilar channel pairs by |corr|
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # ---------- 1) Collect latents as [N, D] ----------
    exp._ensure_loader(group)
    loader = {"train": exp.train_loader,
            "train_eval": exp.train_eval_loader,
            "test": exp.test_loader}[group]
    dataset = loader.dataset
    Nall = len(dataset)
    n_traj = min(n_traj, Nall)

    from torch.utils.data import Subset, DataLoader
    idxs = np.arange(Nall)[:n_traj].tolist()  # deterministic; change to random if preferred
    tmp_loader = DataLoader(Subset(dataset, idxs), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    lat_list = []
    for batch in tmp_loader:
        lat, _, _ = exp._encode_and_recon(batch)  # [T',1,D]
        z = lat[:, 0].detach().cpu().float()       # [T',D]
        if max_time is not None:
            z = z[:max_time]
        lat_list.append(z)
    if len(lat_list) == 0:
        print("[latent_channel_diagnostics] No samples collected.")
        return None

    X = torch.cat(lat_list, dim=0)                # [sum_T', D]
    Tsum, D = X.shape

    # ---------- 2) Centering options ----------
    X_raw = X.clone()
    used_center = "none"
    if center == "zstar" and getattr(exp, "latent_center", None) is not None:
        zc = exp.latent_center.detach().cpu().view(1, -1).to(X)
        if zc.size(1) == D:
            X = X - zc
            used_center = "zstar"
    elif center == "dataset":
        X = X - X.mean(dim=0, keepdim=True)
        used_center = "dataset"

    # ---------- 3) Per-channel stats ----------
    mu = X.mean(dim=0)                             # [D]
    std = X.std(dim=0, unbiased=True).clamp_min(1e-12)   # [D]
    rms = (X.pow(2).mean(dim=0)).sqrt()            # [D]

    # ---------- 4) Corr (Pearson) and a simple "difference" score ----------
    # cov = E[(x-μ)(x-μ)^T]; corr = cov / (σ_i σ_j)
    Xc = X - mu.view(1, -1)
    cov = (Xc.T @ Xc) / max(1, Tsum - 1)          # [D,D]
    denom = std.view(-1, 1) * std.view(1, -1)
    corr = (cov / denom).clamp(min=-1.0, max=1.0) # [D,D]
    diff = 1.0 - corr.abs()                       # "difference" score ∈ [0,1]

    # ---------- 5) Top-k similar / dissimilar pairs ----------
    # Use upper triangle without diagonal
    iu = torch.triu_indices(D, D, offset=1)
    corr_abs = corr.abs()[iu[0], iu[1]]           # [M]
    top_sim_val, top_sim_idx = torch.topk(corr_abs, k=min(topk, corr_abs.numel()))
    top_dis_val, top_dis_idx = torch.topk(-corr_abs, k=min(topk, corr_abs.numel()))  # smallest |corr|

    def pairs_from_indices(idx_tensor):
        pairs = []
        for j in idx_tensor.tolist():
            i0, i1 = iu[0][j].item(), iu[1][j].item()
            pairs.append((i0, i1))
        return pairs

    print(f"[latent_channel_diagnostics] group={group} | D={D} | N(time-steps)={Tsum} | center={used_center}")
    print("Per-dim summary: show first 8 dims (mean/std/RMS):")
    for d in range(min(8, D)):
        print(f"  dim {d:03d}: mean={float(mu[d]):.4e}  std={float(std[d]):.4e}  rms={float(rms[d]):.4e}")

    sim_pairs = pairs_from_indices(top_sim_idx)
    dis_pairs = pairs_from_indices(top_dis_idx)
    print(f"\nTop-{len(sim_pairs)} most similar pairs by |corr|:")
    for (i, j), v in zip(sim_pairs, top_sim_val):
        print(f"  ({i:03d},{j:03d})  |corr|={float(v):.4f}  corr={float(corr[i,j]):.4f}")

    print(f"\nTop-{len(dis_pairs)} most dissimilar pairs by |corr| (smallest |corr|):")
    for (i, j), v in zip(dis_pairs, -top_dis_val):
        print(f"  ({i:03d},{j:03d})  |corr|={float(v):.4f}  corr={float(corr[i,j]):.4f}")

    # ---------- 6) Optional figures ----------
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # (a) correlation heatmap
        fig1, ax1 = plt.subplots()
        im1 = ax1.imshow(corr.numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm", interpolation="none")
        ax1.set_title(f"Latent corr (center={used_center})")
        ax1.set_xlabel("dim")
        ax1.set_ylabel("dim")
        fig1.colorbar(im1, ax=ax1)
        p1 = os.path.join(save_dir, f"latent_corr_{used_center}.png")
        fig1.savefig(p1, dpi=200, bbox_inches="tight")
        plt.close(fig1)

        # (b) difference heatmap (1-|corr|)
        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(diff.numpy(), vmin=0.0, vmax=1.0, cmap="viridis", interpolation="none")
        ax2.set_title(f"Latent difference (1-|corr|), center={used_center}")
        ax2.set_xlabel("dim")
        ax2.set_ylabel("dim")
        fig2.colorbar(im2, ax=ax2)
        p2 = os.path.join(save_dir, f"latent_diff_{used_center}.png")
        fig2.savefig(p2, dpi=200, bbox_inches="tight")
        plt.close(fig2)

        # (c) per-dim bar of RMS (energy proxy)
        fig3, ax3 = plt.subplots()
        ax3.bar(np.arange(D), rms.numpy())
        ax3.set_title(f"Per-dim RMS, center={used_center}")
        ax3.set_xlabel("dimension")
        ax3.set_ylabel("RMS")
        p3 = os.path.join(save_dir, f"latent_rms_{used_center}.png")
        fig3.savefig(p3, dpi=200, bbox_inches="tight")
        plt.close(fig3)

    # ---------- 7) If we centered, quantify the reduction in per-step energy ----------
    info = {}
    if center in {"zstar", "dataset"}:
        l2_before = torch.linalg.norm(X_raw, dim=1)   # [N]
        l2_after  = torch.linalg.norm(X, dim=1)       # [N]
        red = 1.0 - (l2_after.mean() / l2_before.mean())
        print(f"\nEnergy reduction by centering ({used_center}): "
            f"{100.0*float(red):.2f}%  (mean L2 per step)")
        info["energy_reduction_meanL2"] = float(red)

    # Return a compact dict in case you want to log programmatically
    info.update({
        "center": used_center,
        "per_dim_mean": mu,
        "per_dim_std": std,
        "per_dim_rms": rms,
        "corr": corr,
    })
    return info

@torch.no_grad()
def latent_center_report(exp, topk: int = 20, save_dir: str | None = None):
    """
    Summarize the learned latent_center (z*), if available:
    - print L2/Linf norms
    - list top-|z*| dimensions
    - optional bar plot
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    zc = getattr(exp, "latent_center", None)
    if zc is None:
        print("[latent_center_report] latent_center is None (not set).")
        return None

    z = zc.detach().cpu().view(-1).float()
    D = z.numel()
    l2 = float(torch.linalg.norm(z, ord=2))
    linf = float(torch.linalg.norm(z, ord=float("inf")))
    print(f"[latent_center_report] D={D} | ||z*||_2={l2:.6f} | ||z*||_inf={linf:.6f}")

    # Top-|z| dims
    topk = min(topk, D)
    vals, idx = torch.topk(z.abs(), k=topk)
    print(f"Top-{topk} dims by |z*|:")
    for r in range(topk):
        d = int(idx[r])
        print(f"  rank {r+1:02d}: dim={d:03d}  z*={float(z[d]):.6f}  |z*|={float(vals[r]):.6f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = plt.subplots()
        ax.bar(np.arange(D), z.numpy())
        ax.set_title("latent_center (z*) values per dimension")
        ax.set_xlabel("dimension")
        ax.set_ylabel("value")
        p = os.path.join(save_dir, "latent_center_bar.png")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return {"z_star": z, "l2": l2, "linf": linf}

def illustrate_one_frame_pred(exp, batch_samples, rollout_steps: int, out_dir: str, dyn_type: str = "memory"):
    assert dyn_type in {"linear", "memory", "recon"}
    if dyn_type == "linear":
        pred_tensor, true_tensor = exp.linear_rollout_one_batch_with_Ab(batch_samples, rollout_steps)
    elif dyn_type == "memory":
        pred_tensor, true_tensor = exp.rollout_one_batch(batch_samples, rollout_steps)    # [B, H, W, T, C]
    else:
        pred_tensor, true_tensor = exp.recon_one_batch(batch_samples)
    print(pred_tensor.shape, true_tensor.shape)
    assert exp.spatial_dim == 2
    out_png = visualize_pred_vs_gt(
        pred_tensor, true_tensor,
        mode="last", batch_index=0,
        save_path=os.path.join(out_dir, 'one_frame_pred'),
        show_error=True, err_type="abs", add_colorbar=True,
        header_time_ratio=0.28,   # ↓ smaller -> closer
        header_cols_ratio=0.34,   # optionally shrink the column-title row too
        time_fontsize=12          # slightly smaller time label
    )
    print("Saved last-step prediction to:", out_png)

def illustrate_long_term_pred(exp, batch_samples, rollout_steps: int, out_dir: str, dyn_type: str = "memory"):
    assert dyn_type in {"linear", "memory", "recon"}
    if dyn_type == "linear":
        pred_tensor, true_tensor = exp.linear_rollout_one_batch_with_Ab(batch_samples, rollout_steps)
    elif dyn_type == "memory":
        pred_tensor, true_tensor = exp.rollout_one_batch(batch_samples, rollout_steps)    # [B, H, W, T, C]
    else:
        pred_tensor, true_tensor = exp.recon_one_batch(batch_samples)
    print(pred_tensor.shape, true_tensor.shape)
    assert exp.spatial_dim == 2
    out_gif = visualize_pred_vs_gt(
        pred_tensor, true_tensor,
        mode="all", batch_index=0,
        save_path=os.path.join(out_dir, 'long_term_seq.gif'),
        as_type="gif", fps=3,
        show_error=True, err_type="abs", add_colorbar=True,
        header_time_ratio=0.28, header_cols_ratio=0.34
    )
    print("Saved GIF to:", out_gif)

def visualize_random_rollout(exp, group: str, batch_size: int, rollout_steps: int,
                             out_dir: str, mode: str = "last", dyn_type: str = "memory"):
    """
    Sample a random batch from the given split and visualize rollout.
    - group: "train" (seen IC), "train_eval", or "test" (new IC)
    - batch_size: arbitrary batch size for visualization (independent of training batch_size)
    - rollout_steps: number of prediction steps (<= T_out in data_y)
    - mode: "last" -> plot last-step panel; "all" -> export GIF/MP4 via your visualize utility
    """
    os.makedirs(out_dir, exist_ok=True)
    batch_samples = exp.sample_batch(group=group, batch_size=batch_size)
    assert exp.spatial_dim == 2
    if mode == "last":
        illustrate_one_frame_pred(exp, batch_samples, rollout_steps, out_dir, dyn_type)
    else:
        illustrate_long_term_pred(exp, batch_samples, rollout_steps, out_dir, dyn_type)

def visualize_rollout_by_index(
    exp,
    group: str,              # "train" | "train_eval" | "test"
    seq_index: int,          # zero-based index in the dataset order
    rollout_steps: int,
    out_dir: str,
    mode: str = "last",      # "last": plot last step; "all": export full GIF
    dyn_type: str = "memory",
):
    """
    Pick the `seq_index`-th sample (by dataset order) from the specified split,
    run a K-step rollout, and visualize the results.

    This bypasses any DataLoader shuffling by addressing the dataset directly.
    """
    import os
    from torch.utils.data import Subset, DataLoader

    os.makedirs(out_dir, exist_ok=True)

    # Ensure the corresponding loader (and hence dataset) exists
    exp._ensure_loader(group)
    loader = {"train": exp.train_loader,
            "train_eval": exp.train_eval_loader,
            "test": exp.test_loader}[group]
    dataset = loader.dataset
    N = len(dataset)
    if not (0 <= seq_index < N):
        raise IndexError(f"seq_index={seq_index} out of bounds (0..{N-1})")

    # Build a temporary DataLoader that yields exactly this single sample (batch_size=1)
    subset = Subset(dataset, [seq_index])
    tmp_loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    batch_samples = next(iter(tmp_loader))

    # Visualize rollout
    if mode == "last":
        illustrate_one_frame_pred(exp, batch_samples, rollout_steps, out_dir, dyn_type)
    else:
        illustrate_long_term_pred(exp, batch_samples, rollout_steps, out_dir, dyn_type)

def save_rollout_comparison(
    exp,
    group: str,              # "train" | "train_eval" | "test"
    rollout_steps: int,
    out_dir: str,
    seq_index: int | None = None,          # zero-based index in the dataset order
    save_linear: bool = True
):
    import os
    from torch.utils.data import Subset, DataLoader
    os.makedirs(out_dir, exist_ok=True)
    if seq_index is None:
        batch_samples = exp.sample_batch(group=group, batch_size=1)
    else:
        # Ensure the corresponding loader (and hence dataset) exists
        exp._ensure_loader(group)
        loader = {"train": exp.train_loader,
                "train_eval": exp.train_eval_loader,
                "test": exp.test_loader}[group]
        dataset = loader.dataset
        N = len(dataset)
        if not (0 <= seq_index < N):
            raise IndexError(f"seq_index={seq_index} out of bounds (0..{N-1})")
        # Build a temporary DataLoader that yields exactly this single sample (batch_size=1)
        subset = Subset(dataset, [seq_index])
        tmp_loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        batch_samples = next(iter(tmp_loader))

    if save_linear:
        pred_tensor, true_tensor, true_tensor_full = exp.linear_rollout_one_batch_with_Ab(batch_samples, rollout_steps, return_gt=True)
        payload = {
            "pred": pred_tensor.detach().cpu().contiguous(),
            "true": true_tensor.detach().cpu().contiguous(),
            "true_full": true_tensor_full.detach().cpu().contiguous(),
        }
        torch.save(payload, os.path.join(out_dir, "phase1"))

    pred_tensor, true_tensor, true_tensor_full = exp.rollout_one_batch(batch_samples, rollout_steps, return_gt=True)
    payload = {
        "pred": pred_tensor.detach().cpu().contiguous(),
        "true": true_tensor.detach().cpu().contiguous(),
        "true_full": true_tensor_full.detach().cpu().contiguous(),
    }
    torch.save(payload, os.path.join(out_dir, "phase2"))

@torch.no_grad()
def evaluate_long_trajs(
    exp, out_dir: str, 
    group: str = "test",
    rollout_steps: int | None = None,
    batch_size: int | None = None,
    loader=None,
    save_pt: bool = True,
    save_png: bool = False,
    vis_cols: int = 8,
    c_vis: int = 0,
    save_time_curve: bool = True, 
    return_run_time: bool = False
):
    import os, json, numpy as np
    import matplotlib.pyplot as plt
    t_eval_ref = None
    curve_sum   = None
    curve_sumsq = None
    curve_count = None

    os.makedirs(out_dir, exist_ok=True)
    exp.switch_to_eval()
    if loader is None:
        loader = exp._build_long_eval_loader(group=group, rollout_steps=rollout_steps, batch_size=batch_size)
    dataset = loader.dataset
    base_dataset = getattr(dataset, "dataset", dataset)
    samples_list = base_dataset.samples  # List[(traj_id, t0)]

    all_mse = []
    run_time = 0.0
    for batch in loader:
        begin = time.time()
        latent_state = exp._encode_cond_batch(batch)          # [B, latent_dim]
        latent_state_ = exp._center_latent(latent_state)
        latent_state_ = exp._whiten_latent(latent_state_)
        if exp.use_projector:
            latent_state_ = exp._project_latent(latent_state_)
        n_cond_m1 = int(exp.n_frames_cond) - 1
        t_vec = batch["t"][0].to(exp.device)                  # [T]
        t_eval = t_vec[n_cond_m1:]                             # [T' = length - (n_cond-1)]
        dyn_states_, _, _ = exp.latent_process(alpha_0=latent_state_, t_eval=t_eval, teacher_forcing=False)
        if exp.use_projector:
            dyn_states_ = exp._lift_latent(dyn_states_)
        dyn_states_ = exp._unwhiten_latent(dyn_states_)
        dyn_states = exp._decenter_latent(dyn_states_)        # [T', B, latent_dim]
        recon_seq = exp._decode_latent(dyn_states)
        end = time.time()
        run_time += (end - begin)
        recon_seq_sp_last = recon_seq.permute(0, 2, 3, 1, 4).contiguous()           # [B, H, W, T', C]
        gt_full = batch["data"].to(exp.device)                                     # [B, T, H, W, C]
        gt_slice = gt_full[:, n_cond_m1:, ...]                                      # [B, T', H, W, C]
        mask_full = batch.get("mask", None)
        if mask_full is not None:
            mask_slice = mask_full.to(exp.device)[:, n_cond_m1:, ...]              # [B, T', H, W, C]
        else:
            mask_slice = None

        if mask_slice is None:
            se = (recon_seq - gt_slice).pow(2)                                      # [B,T',H,W,C]
            mse_b = se.mean(dim=(1,2,3,4))                                          # [B]
            mse_t = se.mean(dim=(0,2,3,4)).detach().cpu()                           # [T']
            mse_bt = se.mean(dim=(2,3,4))                                           # [time-curve] -> [B,T']
            valid_bt = torch.ones_like(mse_bt, dtype=mse_bt.dtype)                  
        else:
            se = (recon_seq - gt_slice).pow(2) * mask_slice
            num_b = se.sum(dim=(1,2,3,4)); den_b = mask_slice.sum(dim=(1,2,3,4)).clamp_min(1e-6)
            mse_b = (num_b / den_b)                                                 # [B]
            num_t = se.sum(dim=(0,2,3,4)); den_t = mask_slice.sum(dim=(0,2,3,4)).clamp_min(1e-6)
            mse_t = (num_t / den_t).detach().cpu()                                  # [T']
            num_bt = se.sum(dim=(2,3,4))                                            # NEW [time-curve] -> [B,T']
            den_bt = mask_slice.sum(dim=(2,3,4)).clamp_min(1e-6)                    # NEW [time-curve]
            mse_bt = (num_bt / den_bt)                                              # NEW [time-curve]
            valid_bt = (den_bt > 0).to(mse_bt.dtype)                                # NEW [time-curve]

        all_mse.extend([float(x) for x in mse_b])

        # -------- NEW [time-curve]: accumulate per-time MSE stats across samples --------
        if save_time_curve:
            mse_bt_cpu   = mse_bt.detach().cpu()
            valid_bt_cpu = valid_bt.detach().cpu()
            if curve_sum is None:
                Tprime = int(mse_bt_cpu.shape[1])
                import torch as _torch
                curve_sum   = _torch.zeros(Tprime)
                curve_sumsq = _torch.zeros(Tprime)
                curve_count = _torch.zeros(Tprime)
                t_eval_ref  = t_eval.detach().cpu()
            else:
                assert mse_bt_cpu.shape[1] == curve_sum.shape[0], "T' mismatch across batches"

            curve_sum   += (mse_bt_cpu * valid_bt_cpu).sum(dim=0)     # sum_b MSE
            curve_sumsq += ((mse_bt_cpu**2) * valid_bt_cpu).sum(dim=0)# sum_b MSE^2
            curve_count += valid_bt_cpu.sum(dim=0)                    # valid sample count
        # ----------------------------------------------------------------


        if save_pt or save_png:
            idx_vec = batch["index"].tolist() 
            for b, ds_idx in enumerate(idx_vec):
                traj_id, t0 = samples_list[int(ds_idx)]
                stem = f"traj{int(traj_id):04d}_t0{int(t0):04d}"

                if save_pt:
                    payload = {
                        "traj_id": int(traj_id),
                        "t0": int(t0),
                        "t": t_eval.detach().cpu(),                                  # [T']
                        "pred": recon_seq_sp_last[b].detach().cpu(),                 # [H,W,T',C]
                        "gt":   gt_slice[b].permute(1,2,0,3).detach().cpu(),         # [H,W,T',C]
                        "mse_all": float(mse_b[b].item()),
                        "mse_t": mse_t,                                              # [T']
                        "normalized": bool(exp.data_processor.cfg.normalize),
                    }
                    torch.save(payload, os.path.join(out_dir, f"{stem}.pt"))

                if save_png:
                    try:
                        Tsel = min(vis_cols, recon_seq.shape[1])
                        idx  = np.linspace(0, recon_seq.shape[1]-1, Tsel, dtype=int)
                        gt_np  = gt_slice[b].detach().cpu().numpy()                  # [T',H,W,C]
                        pr_np  = recon_seq[b].detach().cpu().numpy()                 # [T',H,W,C]
                        err_np = np.abs(gt_np - pr_np)

                        rows = 3
                        fig, axes = plt.subplots(rows, Tsel, figsize=(2.6*Tsel, 2.6*rows), squeeze=False)
                        for j, ti in enumerate(idx):
                            axes[0, j].imshow(gt_np[ti, :, :, c_vis]); axes[0, j].set_title(f"GT t={float(t_eval[ti]):.2f}")
                            axes[1, j].imshow(pr_np[ti, :, :, c_vis]); axes[1, j].set_title("Pred")
                            axes[2, j].imshow(err_np[ti, :, :, c_vis]); axes[2, j].set_title("|Err|")
                            for r in range(rows): axes[r, j].axis('off')
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=160)
                        plt.close(fig)
                    except Exception:
                        pass

        print(f"[{group}] batch_size={recon_seq.shape[0]} | T'={recon_seq.shape[1]} | MSE_mean(batch)={float(mse_b.mean().item()):.6e}")

    import numpy as np
    mean_mse = float(np.mean(all_mse)) if all_mse else float("nan")
    std_mse  = float(np.std(all_mse))  if all_mse else float("nan")
    summary = {"group": group, "n_samples": len(all_mse), "MSE_mean": mean_mse, "MSE_std": std_mse, "run_time": run_time}
    with open(os.path.join(out_dir, f"summary_{group}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"==> [{group}] Long-eval(batched) summary: MSE_mean={mean_mse:.6e} over {len(all_mse)} samples")

    # -------- NEW [time-curve]: summarize and save the error-time curve (mean +/- std) --------
    if save_time_curve and (curve_count is not None) and (curve_count.max().item() > 0):
        mean_curve = (curve_sum / curve_count).numpy()                 # [T']
        var_curve  = (curve_sumsq / curve_count).numpy() - mean_curve**2
        var_curve  = np.clip(var_curve, 0.0, None)
        std_curve  = np.sqrt(var_curve)

        fig, ax = plt.subplots(figsize=(7, 4))
        t_np = t_eval_ref.numpy()
        ax.plot(t_np, mean_curve, label="MSE (mean across traj)")
        ax.fill_between(t_np, mean_curve-std_curve, mean_curve+std_curve, alpha=0.2, label="±1 std")
        ax.set_xlabel("t")
        ax.set_ylabel("MSE")
        ax.set_title(f"MSE vs time — {group}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(out_dir, f"mse_over_time_{group}.png"), dpi=180)
        plt.close(fig)

        # np.savez(os.path.join(out_dir, f"mse_over_time_{group}.npz"),
        #         t=t_np, mean=mean_curve, std=std_curve, count=curve_count.numpy())
        out_json = os.path.join(out_dir, f"mse_over_time_{group}.json")                         # NEW [time-curve-json]
        series = {                                                                              # NEW [time-curve-json]
            "group": group,                                                                     # NEW [time-curve-json]
            "t":    [float(x) for x in t_np.tolist()],                                          # NEW [time-curve-json]
            "mean": [float(x) for x in mean_curve.tolist()],                                    # NEW [time-curve-json]
            "std":  [float(x) for x in std_curve.tolist()],                                     # NEW [time-curve-json]
            "count":[int(x)   for x in curve_count.detach().cpu().numpy().astype(int).tolist()] # NEW [time-curve-json]
        }                                                                                       # NEW [time-curve-json]
        with open(out_json, "w") as f:                                                          # NEW [time-curve-json]
            json.dump(series, f, indent=2)                                                      # NEW [time-curve-json]
    # --------------------------------------------------------------------

    return summary

def evaluate_long_by_indices(exp, out_dir: str, group: str, indices: list[int],
                             rollout_steps: int | None = None,
                             **kwargs):
    from torch.utils.data import Subset, DataLoader
    base_loader = exp._build_long_eval_loader(group=group, rollout_steps=rollout_steps)
    subset = Subset(base_loader.dataset, indices)
    sub_loader = DataLoader(subset, batch_size=len(indices), shuffle=False,
                            num_workers=0, pin_memory=True)
    return evaluate_long_trajs(exp, group=group, out_dir=out_dir, loader=sub_loader, **kwargs)

@torch.no_grad()
def plot_centering_effects(
    exp,
    group: str,
    seq_index: int = 0,
    rollout_steps: int = 32,
    save_dir: str | None = None,
    use_mask: bool = True,
):
    """
    Compare centered vs uncentered rollouts for Phase-I linear dynamics and the trained latent process.
    Produce two figures:
    (1) per-time-step masked MSE curves
    (2) per-time-step latent L2-norm curves

    Args
    ----
    group         : "train" | "train_eval" | "test"
    seq_index     : take the seq_index-th sample (by dataset order)
    rollout_steps : number of predicted steps (<= T_out)
    save_dir      : if provided, figures will be saved here
    use_mask      : use dataset masks when computing MSE (recommended)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # ---------------------- helpers ----------------------
    def _one_sample_from_group(group: str, seq_index: int):
        """Return a batch (B=1) and some handy tensors."""
        from torch.utils.data import Subset, DataLoader
        exp._ensure_loader(group)
        loader = {"train": exp.train_loader,
                "train_eval": exp.train_eval_loader,
                "test": exp.test_loader}[group]
        dataset = loader.dataset
        if not (0 <= seq_index < len(dataset)):
            raise IndexError(f"seq_index={seq_index} out of bounds (0..{len(dataset)-1})")
        tmp = DataLoader(Subset(dataset, [seq_index]), batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        batch = next(iter(tmp))
        return batch, dataset

    def _per_time_mse(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor | None):
        """
        pred, gt: [B, T, H, W, C]; return [T] masked average MSE per time step.
        """
        B, T, H, W, C = pred.shape
        if mask is None or not use_mask:
            se = (pred - gt).pow(2)                       # [B,T,H,W,C]
            num = se.sum(dim=(0, 2, 3, 4))                # [T]
            den = torch.full_like(num, fill_value=B*H*W*C, dtype=pred.dtype)
        else:
            se = (pred - gt).pow(2) * mask                # [B,T,H,W,C]
            num = se.sum(dim=(0, 2, 3, 4))                # [T]
            den = mask.sum(dim=(0, 2, 3, 4)).clamp_min(1e-6)  # [T]
        return (num / den).cpu()                          # [T]

    def _latent_l2_per_time(lat_seq: torch.Tensor):
        """
        lat_seq: [T, B, D] -> return [T] average L2 over B.
        """
        l2 = torch.linalg.norm(lat_seq, dim=-1)           # [T,B]
        return l2.mean(dim=1).cpu()                       # [T]

    def _encode_initial_and_targets(batch):
        """
        Build initial latent a0 from the first nf_cond frames, and assemble ground-truth targets and masks.
        Returns:
        a0         : [B, D]
        gt_seq     : [B, T', H, W, C] where T' = rollout_steps+1 aligned with predictions
        mask_seq   : same shape as gt_seq or None
        t_eval_sub : [T'] relative evaluation times for the rollout
        """
        ground_truth = batch["data"].to(exp.device)    # [B, T, H, W, C]
        masks = batch["mask"].to(exp.device)           # [B, T, H, W, C]
        t_full = batch["t"][0].to(exp.device)          # [T]
        B, T, H, W, C = ground_truth.shape
        n_cond = exp.n_frames_cond - 1
        assert rollout_steps + exp.data_processor.n_frames_cond <= T, \
            f"rollout_steps too long for this sequence (T={T})"

        # ---- encode a0 from the first nf_cond frames through the shared path ----
        a0 = exp._encode_cond_batch(batch)

        # ---- ground-truth subseq aligned with predictions ----
        t_eval_sub = t_full[n_cond:n_cond+rollout_steps+1]  # [T']
        gt_seq = ground_truth[:, n_cond:n_cond+rollout_steps+1, ...]  # [B,T',H,W,C]
        mask_seq = masks[:, n_cond:n_cond+rollout_steps+1, ...]       # [B,T',H,W,C]
        return a0, gt_seq, mask_seq, t_eval_sub

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # ---------------------- fetch one sample ----------------------
    batch, _ = _one_sample_from_group(group, seq_index)
    a0, gt_seq, mask_seq, t_eval_sub = _encode_initial_and_targets(batch)
    B, Tprime, H, W, C = gt_seq.shape
    D = a0.shape[-1]

    # convenience
    zstar = getattr(exp, "latent_center", None)
    has_center = (zstar is not None) and (zstar.numel() == D)

    # ---------------------- rollouts we will compare ----------------------
    curves = {
        "disc_uncentered": None,
        "disc_centered": None,
        "model_uncentered": None,
        "model_centered": None,
    }
    l2_curves = {}   # per-time latent L2 curves

    # ===== DISCRETE: Ad,b linear rollout =====
    if hasattr(exp, "Ad_phase1") and (exp.Ad_phase1 is not None):
        Ad = exp.Ad_phase1.to(exp.device, dtype=a0.dtype)
        b  = getattr(exp, "b_phase1", None)
        b  = (b.to(exp.device, dtype=a0.dtype).view(1, -1) if b is not None else None)

        # --- uncentered: z_{t+1} = A z_t + b ---
        z = torch.empty((Tprime, B, D), device=exp.device, dtype=a0.dtype)
        z[0] = a0
        for t in range(Tprime - 1):
            z[t+1] = z[t] @ Ad.T + (b if b is not None else 0.0)
        l2_curves["disc_uncentered"] = _latent_l2_per_time(z)

        # decode to fields
        pred = exp._decode_latent(z)
        curves["disc_uncentered"] = _per_time_mse(pred, gt_seq, mask_seq)

        # --- centered: y=z-z*, y_{t+1}=A y_t, final add z* back ---
        if has_center:
            zc = zstar.to(exp.device, dtype=a0.dtype).view(1, -1)  # [1,D]
            y = torch.empty((Tprime, B, D), device=exp.device, dtype=a0.dtype)
            y[0] = a0 - zc
            for t in range(Tprime - 1):
                y[t+1] = y[t] @ Ad.T   # no bias if z* solves (I-Ad)z*=b
            z_cent = y + zc
            l2_curves["disc_centered"] = _latent_l2_per_time(z_cent)

            pred_c = exp._decode_latent(z_cent)
            curves["disc_centered"] = _per_time_mse(pred_c, gt_seq, mask_seq)
        else:
            print("[plot_centering_effects] latent_center is None -> skip discrete-centered curve.")
    else:
        print("[plot_centering_effects] Ad_phase1/b_phase1 not found -> skip DISCRETE comparison.")

    # ===== trained discrete latent_process rollout =====
    # uncentered: feed a0 directly (no center/decenter)
    out_uc = exp.latent_process(alpha_0=a0, t_eval=t_eval_sub - t_eval_sub[0], teacher_forcing=False)
    z_uc = out_uc[0] if isinstance(out_uc, tuple) else out_uc   # [T', B, D]
    l2_curves["model_uncentered"] = _latent_l2_per_time(z_uc)
    # decode
    pred_uc = exp._decode_latent(z_uc)
    curves["model_uncentered"] = _per_time_mse(pred_uc, gt_seq, mask_seq)

    # centered: a0-z*, integrate, then add z* back (your train_phase2 style)
    if has_center:
        a0c = a0 - zstar.to(exp.device, dtype=a0.dtype).view(1, -1)
        out_c = exp.latent_process(alpha_0=a0c, t_eval=t_eval_sub - t_eval_sub[0], teacher_forcing=False)
        z_c = (out_c[0] if isinstance(out_c, tuple) else out_c) + zstar.to(exp.device, dtype=a0.dtype).view(1, 1, -1)
        l2_curves["model_centered"] = _latent_l2_per_time(z_c)
        pred_c = exp._decode_latent(z_c)
        curves["model_centered"] = _per_time_mse(pred_c, gt_seq, mask_seq)
    else:
        print("[plot_centering_effects] latent_center is None -> skip model-centered curve.")

    # ---------------------- plot: MSE curves ----------------------
    t_axis = np.arange(Tprime)
    fig1, ax1 = plt.subplots()
    if curves["disc_uncentered"] is not None: ax1.plot(t_axis, curves["disc_uncentered"].numpy(), label="Discrete - Uncentered")
    if curves["disc_centered"]   is not None: ax1.plot(t_axis, curves["disc_centered"].numpy(),   label="Discrete - Centered")
    if curves["model_uncentered"] is not None: ax1.plot(t_axis, curves["model_uncentered"].numpy(), label="Model - Uncentered")
    if curves["model_centered"]   is not None: ax1.plot(t_axis, curves["model_centered"].numpy(),   label="Model - Centered")
    ax1.set_xlabel("time step")
    ax1.set_ylabel("masked MSE" if use_mask else "MSE")
    ax1.set_title(f"Per-step error  (group={group}, idx={seq_index})")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.3)
    if save_dir:
        p1 = os.path.join(save_dir, f"center_vs_nocenter_mse_{group}_idx{seq_index}.png")
        fig1.savefig(p1, dpi=200, bbox_inches="tight")
        print("Saved:", p1)
    plt.close(fig1)

    # ---------------------- plot: latent L2 curves ----------------------
    fig2, ax2 = plt.subplots()
    for k in ["disc_uncentered", "disc_centered", "model_uncentered", "model_centered"]:
        v = l2_curves.get(k, None)
        if v is not None:
            ax2.plot(t_axis, v.numpy(), label=k.replace("_", " "))
    ax2.set_xlabel("time step")
    ax2.set_ylabel("‖latent‖₂ (avg over batch)")
    ax2.set_title(f"Latent L2 per step  (group={group}, idx={seq_index})")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.3)
    if save_dir:
        p2 = os.path.join(save_dir, f"center_vs_nocenter_l2_{group}_idx{seq_index}.png")
        fig2.savefig(p2, dpi=200, bbox_inches="tight")
        print("Saved:", p2)
    plt.close(fig2)

    # ---------------------- quick console summary ----------------------
    def _summ(v):
        if v is None: return None
        return dict(mean=float(v.mean()), last=float(v[-1]), min=float(v.min()), max=float(v.max()))
    summary = {k: _summ(v) for k, v in curves.items()}
    print("[plot_centering_effects] summary (per-step MSE):", summary)
    return summary

# ---------------------------------------------
# Utilities for diagnostics & regularizers
# ---------------------------------------------

@torch.no_grad()
def plot_dyn_energy_stats(
    exp,
    dataloader,
    save_dir: str,
    fname_prefix: str = "dyn_energy",
    center_and_whiten: bool = True,
    use_projector_space: bool = True,
    # ---- style knobs (new) ----
    fig_w: float = 7.2,
    fig_h: float = 4.0,
    line_w: float = 2.4,            # mean-curve line width
    band_alpha: float = 0.18,       # ±std band transparency
    tick_labelsize: int = 14,       # tick numbers size (bigger axis numbers)
    axis_labelsize: int = 15,       # x/y label size
    spine_lw: float = 1.2,          # all four spines width
    major_grid_alpha: float = 0.32, # grid strength
    minor_grid_alpha: float = 0.22,
    use_minor_grid: bool = True,
    color_linear: str = "#1f77b4",  # vivid blue
    color_memory: str = "#d62728",  # vivid red
    dpi: int = 450,
):
    """
    Compute per-timestep L2 norms of linear & memory contributions, aggregate mean±std,
    and plot them with a clean, publication-friendly style.

    - Four spines enabled and thickened.
    - Larger tick labels (axis numbers) and bold axis labels.
    - Dense major + minor grid.
    - Mean curves with ±std shading.

    Saves:
        save_dir/{fname_prefix}.png
        save_dir/{fname_prefix}.json
        save_dir/{fname_prefix}_raw.pt
    """
    import os, json, torch
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib import rcParams

    os.makedirs(save_dir, exist_ok=True)

    # --------------------------- accumulate stats ---------------------------
    lin_list = []   # list length T-1; each element is a list of [B] tensors
    mem_list = []
    T_ref = None

    for batch in dataloader:
        gt = batch["data"].to(exp.device)         # [B, T, H, W, C]
        masks = batch.get("mask", None)
        if masks is not None:
            masks = masks.to(exp.device)
        t_eval = batch["t"][0].to(exp.device)     # [T]
        B, T = gt.shape[0], gt.shape[1]
        if T_ref is None:
            T_ref = T

        # Encode following your training path (already aligned to [nf_cond-1:])
        latent_states, _, _ = exp._encode_and_recon(batch)  # [T', B, D_full]
        z = latent_states
        Tprime = z.size(0)  # timesteps used by latent process

        # Canonicalize to internal latent process space (center/whiten/project)
        z_ = z
        if center_and_whiten:
            z_ = exp._center_latent(z_)
            z_ = exp._whiten_latent(z_)
        if use_projector_space and exp.use_projector:
            z_ = exp._project_latent(z_)     # [T', B, d_internal]

        # One teacher-forced pass to expose memory states if any
        out = exp.latent_process.forward(
            alpha_0=z_[0],
            t_eval=t_eval[exp.n_frames_cond-1:],
            memory_init=None,
            teacher_forcing=True,
            tf_alpha=z_,
            tf_epsilon=0.0,
            tf_mask=None,
        )
        if isinstance(out, tuple):
            _, memory_states, _aux = out
        else:
            memory_states = None

        A = exp.latent_process.A                 # [D_s, D_s]
        zk = z_[:-1]                              # [T'-1, B, D_s]
        lin = torch.matmul(zk, A.T)               # A z_k -> [T'-1, B, D_s]

        if getattr(exp.latent_process, "memory_type", "decoder") == "residual":
            mem = memory_states[:-1]              # [T'-1, B, D_s]
        else:
            Dm = memory_states.shape[-1] if (memory_states is not None) else 0
            mem_flat = memory_states[:-1].reshape(-1, Dm)             # [(T'-1)*B, Dm]
            corr = exp.latent_process.memory_decoder(mem_flat)        # [(T'-1)*B, D_s]
            corr = corr.view(Tprime-1, B, -1)
            gate = getattr(exp.latent_process, "gate", 1.0)
            mem = gate * corr

        # L2 norms per step across feature dim -> [T'-1, B]
        lin_l2 = torch.linalg.norm(lin, ord=2, dim=-1)
        mem_l2 = torch.linalg.norm(mem, ord=2, dim=-1)

        if not lin_list:
            lin_list = [[] for _ in range(Tprime-1)]
            mem_list = [[] for _ in range(Tprime-1)]
        for k in range(Tprime-1):
            lin_list[k].append(lin_l2[k].detach().cpu())
            mem_list[k].append(mem_l2[k].detach().cpu())

    # --------------------------- aggregate mean/std ---------------------------
    lin_mean, lin_std, mem_mean, mem_std = [], [], [], []
    for k in range(len(lin_list)):
        lk = torch.cat(lin_list[k], dim=0).float()  # [N_total]
        mk = torch.cat(mem_list[k], dim=0).float()
        lin_mean.append(float(lk.mean()))
        mem_mean.append(float(mk.mean()))
        lin_std.append(float(lk.std(unbiased=False)))
        mem_std.append(float(mk.std(unbiased=False)))

    # --------------------------- save raw + json ---------------------------
    raw_path = os.path.join(save_dir, f"{fname_prefix}_raw.pt")
    torch.save({
        "lin_l2_per_step": [torch.cat(v, dim=0) for v in lin_list],   # list of [N_total]
        "mem_l2_per_step": [torch.cat(v, dim=0) for v in mem_list],
    }, raw_path)

    json_path = os.path.join(save_dir, f"{fname_prefix}.json")
    with open(json_path, "w") as f:
        json.dump({
            "t_index": list(range(1, len(lin_list) + 1)),
            "linear": {"mean": lin_mean, "std": lin_std},
            "memory": {"mean": mem_mean, "std": mem_std},
            "meta": {
                "use_projector": bool(exp.use_projector and use_projector_space),
                "center_whiten": bool(center_and_whiten),
            }
        }, f, indent=2)

    # --------------------------- plot (publication style) ---------------------------
    rcParams.update({
        "axes.formatter.use_mathtext": True,
        "mathtext.default": "regular",
    })

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x = np.arange(1, len(lin_list) + 1)

    # Show all four spines with consistent width
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(spine_lw)

    # Larger tick labels (axis numbers bigger)
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize, direction="out", length=4.5, width=0.9)
    ax.tick_params(axis="both", which="minor", labelsize=tick_labelsize-1, direction="out", length=3.0, width=0.7)

    # Dense major/minor ticks on x
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    # Minor ticks on y as well
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))

    # Grid (major + optional minor)
    ax.grid(True, which="major", linestyle="--", alpha=major_grid_alpha)
    if use_minor_grid:
        ax.grid(True, which="minor", linestyle=":", alpha=minor_grid_alpha)

    # Helper to draw line + band with fixed color
    def _band(ax, x, mean, std, color, label):
        m = np.asarray(mean); s = np.asarray(std)
        ax.plot(x, m, color=color, linewidth=line_w)
        ax.fill_between(x, m - s, m + s, color=color, alpha=band_alpha, linewidth=0.0)

    _band(ax, x, lin_mean, lin_std, color_linear, label="linear")
    _band(ax, x, mem_mean, mem_std, color_memory, label="memory")

    # Axis labels (bold-ish, larger)
    ax.set_xlabel("time step", fontsize=axis_labelsize, fontweight="bold")
    ax.set_ylabel(r"$\ell_{2}$ norm", fontsize=axis_labelsize, fontweight="bold")

    ax.set_xlim(x[0], x[-1])

    # Legend inside axes with subtle background
    # leg = ax.legend(
    #     loc="best",
    #     frameon=True, fancybox=True,
    #     framealpha=0.95, edgecolor="#444444",
    #     facecolor="white",
    #     fontsize=tick_labelsize-1
    # )

    fig.tight_layout()
    png_path = os.path.join(save_dir, f"{fname_prefix}.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    print(f"[dyn_energy] saved: {png_path}\n- json: {json_path}\n- raw:  {raw_path}")

@torch.no_grad()
def visualize_sample_evolution(
    exp,
    group: str,
    save_dir: str,
    *,
    traj_id: int | None = None,
    t0: int | None = None,
    seq_index: int | None = None,
    steps: int | None = None,
    fname_prefix: str = "latent_evolution",
    # ---------- background options ----------
    bg_mode: str = "none",            # "none" | "discrete_map" | "vector_field" | "landscape"
    grid_res: int = 25,
    field_stride: int = 2,
    field_scale: float = 1.0,
    bg_cmap: str | None = "spiral",
    bg_alpha: float = 0.65,
    # ---- contour overlay on landscape ----
    contour_levels: int = 28,
    contour_color: str = "white",
    contour_lw: float = 0.8,
    contour_alpha: float = 0.55,
    # ---------- extra random samples ----------
    extra_k: int = 3,
    extra_seed: int | None = 123,
    extra_alpha: float = 0.3,        # lighter extras
    # ---------- smoothing (spline) ----------
    smooth_curve: bool = True,
    samples_per_step: int = 12,
    # ---------- markers ----------
    show_markers_main: bool = True,   # keep markers for main
    show_markers_extra: bool = False, # <-- TURNED OFF for extras
    marker_size_main: float = 18.0,
    marker_size_extra: float = 12.0,
    marker_alpha_main: float = 0.9,
    marker_alpha_extra: float = 0.45,
    # ---------- arrows (hollow, sharp) ----------
    arrow_count_main: int = 12,
    arrow_count_extra: int = 8,
    arrow_size_main: float = 14.0,
    arrow_size_extra: float = 11.0,
    # linewidths (main solid; extras thinner)
    lw_main: float = 3.2,
    lw_extra: float = 1.8,            # <-- slightly thinner than before
    lw_arrow_main: float | None = None,   # if None -> 0.8 * lw_main
    lw_arrow_extra: float | None = None,  # if None -> 0.75 * lw_extra
    # ---------- axes & frame ----------
    spine_lw: float = 1.2,
    tick_labelsize: int = 14,
):
    """
    Main trajectory (by traj_id,t0 or seq_index) + several random extras,
    projected to PCA-2D fitted on the main GT latents.

    This version:
    - MAIN curves are SOLID (no gradient).
    - EXTRAS are thinner (lw_extra) and draw NO discrete markers by default.
    - Hollow, sharp arrow heads; four spines; DPI=450.
    """
    import os, numpy as np, matplotlib.pyplot as plt, matplotlib.ticker as mticker
    from matplotlib import rcParams
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch
    import torch
    from torch.utils.data import Subset, DataLoader
    rng = np.random.default_rng(extra_seed)

    # Colors: GT / Linear / Full (sky blue)
    COL_GT   = "#000000"
    COL_LIN  = "#ff7f0e"
    COL_FULL = "#2FA4FF"

    def _get_bg_cmap(name: str | None):
        if name in (None, "", "spiral"):
            colors = [
                "#ecff3a", "#bce86b", "#7dd672", "#38c27f", "#1da1a2",
                "#2d79c7", "#4653b3", "#5b3a9e", "#4a2d84", "#2b1b53"
            ]
            return LinearSegmentedColormap.from_list("spiral", colors, N=256)
        return plt.get_cmap(name)

    # ---------- dataset plumbing ----------
    exp._ensure_loader(group)
    loader = {"train": exp.train_loader,
            "train_eval": exp.train_eval_loader,
            "test": exp.test_loader}[group]
    dataset = loader.dataset

    if seq_index is None:
        if not hasattr(dataset, "samples"):
            raise ValueError("dataset has no 'samples' attribute; please pass seq_index.")
        pos = {(int(tid), int(s0)): i for i, (tid, s0) in enumerate(dataset.samples)}
        key = (int(traj_id), int(t0))
        if key not in pos:
            raise KeyError(f"Sample (traj_id={traj_id}, t0={t0}) not found.")
        seq_index = pos[key]

    os.makedirs(save_dir, exist_ok=True)

    def _fetch_by_index(idx: int):
        tmp = DataLoader(Subset(dataset, [idx]), batch_size=1, shuffle=False, num_workers=0)
        batch = next(iter(tmp))
        t_full = batch["t"][0].to(exp.device)              # [T]
        t_eff  = t_full[exp.n_frames_cond-1:]              # align with encoded seq
        z_gt_full, _, _ = exp._encode_and_recon(batch)     # [T',1,D]
        z_gt_full = z_gt_full[:, 0]
        return batch, t_eff, z_gt_full

    def _to_internal(z_seq: torch.Tensor) -> torch.Tensor:
        z_ = exp._center_latent(z_seq)
        z_ = exp._whiten_latent(z_)
        if exp.use_projector:
            z_ = exp._project_latent(z_)
        return z_

    # ---------- main sample ----------
    batch_main, t_eff_main, z_gt_full = _fetch_by_index(seq_index)
    Tprime_main = z_gt_full.size(0)
    steps_eff = int(steps) if steps is not None else int(Tprime_main)
    steps_eff = max(2, min(steps_eff, int(Tprime_main)))

    z_gt_int_main = _to_internal(z_gt_full)

    # linear-only rollout
    A_int = exp.latent_process.A.to(z_gt_int_main)
    z_lin = [z_gt_int_main[0]]
    for _ in range(1, steps_eff):
        z_lin.append(z_lin[-1] @ A_int.T)
    z_lin_main = torch.stack(z_lin, dim=0)

    # full rollout
    z_full_main, *_ = exp.latent_process(
        alpha_0=z_gt_int_main[0].unsqueeze(0),
        t_eval=t_eff_main[:steps_eff],
        teacher_forcing=False
    )
    z_full_main = z_full_main[:, 0]

    # PCA on main GT ONLY
    X = z_gt_int_main[:steps_eff].float()
    mu = X.mean(dim=0, keepdim=True)
    Xc = X - mu
    C = Xc.T @ Xc / max(1, Xc.size(0) - 1)
    _, vecs = torch.linalg.eigh(C)
    W = vecs[:, -2:]                                        # [Dz, 2]

    def _proj2(Z: torch.Tensor) -> np.ndarray:
        return ((Z.float() - mu) @ W).cpu().numpy()

    P_gt_main   = _proj2(z_gt_int_main[:steps_eff])
    P_lin_main  = _proj2(z_lin_main)
    P_full_main = _proj2(z_full_main)

    # bounds from main
    all_xy = np.vstack([P_gt_main, P_lin_main, P_full_main])
    xmin, xmax = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    ymin, ymax = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    xmid, ymid = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    L = max(xmax - xmin, ymax - ymin); pad = 0.05 * L
    xlim = (xmid - 0.5 * L - pad, xmid + 0.5 * L + pad)
    ylim = (ymid - 0.5 * L - pad, ymid + 0.5 * L + pad)

    # ---------- background prep ----------
    A_lin = None
    G_gen = None

    def _logm_via_scipy(A: torch.Tensor) -> torch.Tensor:
        from scipy.linalg import logm
        A_np = A.detach().to("cpu", dtype=torch.float64).numpy()
        Lm = logm(A_np)
        Lm = np.real_if_close(Lm, tol=1e-7)
        if np.iscomplexobj(Lm):
            Lm = Lm.real
        return torch.from_numpy(Lm).to(device=A.device, dtype=A.dtype)

    if bg_mode in {"discrete_map", "vector_field", "landscape"}:
        A_lin = exp.latent_process.A.detach().to(z_gt_int_main)
        try:
            G_gen = _logm_via_scipy(A_lin) / float(exp.dt_eval)
        except Exception:
            G_gen = None

    bg = {"mode": bg_mode, "xlim": xlim, "ylim": ylim}
    if bg_mode != "none":
        Xg = np.linspace(xlim[0], xlim[1], grid_res)
        Yg = np.linspace(ylim[0], ylim[1], grid_res)
        XX, YY = np.meshgrid(Xg, Yg)
        P_grid = np.stack([XX, YY], axis=-1).reshape(-1, 2)
        W_np = W.cpu().numpy(); mu_np = mu.cpu().numpy().reshape(-1)
        Z_grid = (mu_np[None, :] + P_grid @ W_np.T)
        Z_grid_t = torch.from_numpy(Z_grid).to(z_gt_int_main)

        if bg_mode in {"discrete_map", "vector_field"}:
            if bg_mode == "discrete_map":
                if A_lin is None:
                    U = V = np.zeros_like(XX)
                else:
                    Z_next = (Z_grid_t @ A_lin.T)
                    P_next = _proj2(Z_next)
                    P_curr = _proj2(Z_grid_t)
                    dP = P_next - P_curr
                    U = dP[:, 0].reshape(XX.shape); V = dP[:, 1].reshape(XX.shape)
            else:
                if G_gen is None and A_lin is not None:
                    try:
                        G_gen = _logm_via_scipy(A_lin) / float(exp.dt_eval)
                    except Exception:
                        G_gen = None
                if G_gen is not None:
                    Vz = (Z_grid_t @ G_gen.T)
                    dP = _proj2(Vz)
                else:
                    if A_lin is None:
                        dP = np.zeros_like(P_grid)
                    else:
                        Z_next = (Z_grid_t @ A_lin.T)
                        dP = _proj2(Z_next) - _proj2(Z_grid_t)
                U = dP[:, 0].reshape(XX.shape); V = dP[:, 1].reshape(XX.shape)
            bg.update({"grid_X": XX, "grid_Y": YY, "U": U, "V": V})

        elif bg_mode == "landscape":
            if G_gen is not None:
                Vz = (Z_grid_t @ G_gen.T)
                S = torch.linalg.norm(Vz, dim=-1).cpu().numpy().reshape(XX.shape)
            else:
                if A_lin is None:
                    S = np.zeros_like(XX)
                else:
                    Z_next = (Z_grid_t @ A_lin.T)
                    disp = Z_next - Z_grid_t
                    S = torch.linalg.norm(disp, dim=-1).cpu().numpy().reshape(XX.shape)
            # mild smoothing for nicer rings
            try:
                from scipy.ndimage import gaussian_filter
                S_plot = gaussian_filter(S, sigma=0.8)
            except Exception:
                S_plot = S
            bg.update({"grid_X": XX, "grid_Y": YY, "S": S_plot})

    # ---------- collect extra samples ----------
    extra_seq_indices, P_gt_extra, P_lin_extra, P_full_extra = [], [], [], []
    if hasattr(dataset, "samples") and extra_k > 0:
        pool = np.arange(len(dataset.samples))
        pool = pool[pool != int(seq_index)]
        if len(pool) > 0:
            rng.shuffle(pool)
            chosen = pool[:min(extra_k, len(pool))]
            for idx in chosen:
                try:
                    _, t_eff_e, z_gt_full_e = _fetch_by_index(int(idx))
                    Tprime_e = int(z_gt_full_e.size(0))
                    steps_e = max(2, min(steps_eff, Tprime_e))

                    z_gt_int_e = _to_internal(z_gt_full_e)
                    # linear-only for extra
                    A_int_e = exp.latent_process.A.to(z_gt_int_e)
                    tmp = [z_gt_int_e[0]]
                    for _ in range(1, steps_e):
                        tmp.append(tmp[-1] @ A_int_e.T)
                    z_lin_e = torch.stack(tmp, dim=0)

                    z_full_e, *_ = exp.latent_process(
                        alpha_0=z_gt_int_e[0].unsqueeze(0),
                        t_eval=t_eff_e[:steps_e],
                        teacher_forcing=False
                    )
                    z_full_e = z_full_e[:, 0]

                    P_gt_extra.append(_proj2(z_gt_int_e[:steps_e]))
                    P_lin_extra.append(_proj2(z_lin_e))
                    P_full_extra.append(_proj2(z_full_e))
                    extra_seq_indices.append(int(idx))

                    all_xy = np.vstack([all_xy, P_gt_extra[-1], P_lin_extra[-1], P_full_extra[-1]])
                except Exception:
                    continue

    # update bounds including extras
    xmin, xmax = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    ymin, ymax = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    xmid, ymid = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    L = max(xmax - xmin, ymax - ymin); pad = 0.05 * L
    xlim = (xmid - 0.5 * L - pad, xmid + 0.5 * L + pad)
    ylim = (ymid - 0.5 * L - pad, ymid + 0.5 * L + pad)

    # ---------- smoothing ----------
    def _smooth_polyline(P: np.ndarray, n_per_seg: int) -> np.ndarray:
        """Return smoothed & upsampled polyline P (N,2)."""
        if not smooth_curve or P.shape[0] < 3 or n_per_seg <= 1:
            t = np.arange(P.shape[0], dtype=float)
            ti = np.linspace(0, t[-1], (P.shape[0]-1)*max(1, n_per_seg)+1)
            xi = np.interp(ti, t, P[:,0]); yi = np.interp(ti, t, P[:,1])
            return np.stack([xi, yi], axis=-1)
        try:
            from scipy.interpolate import CubicSpline
            s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
            if s[-1] <= 0: return P.copy()
            s /= s[-1]
            si = np.linspace(0.0, 1.0, (P.shape[0]-1)*n_per_seg + 1)
            csx = CubicSpline(s, P[:,0], bc_type="natural")
            csy = CubicSpline(s, P[:,1], bc_type="natural")
            xi = csx(si); yi = csy(si)
            return np.stack([xi, yi], axis=-1)
        except Exception:
            # Hermite fallback
            N = P.shape[0]
            Tt = np.zeros_like(P)
            Tt[0]  = P[1]   - P[0]
            Tt[-1] = P[-1]  - P[-2]
            if N > 2: Tt[1:-1] = 0.5 * (P[2:] - P[:-2])
            out = [P[0]]
            for i in range(N-1):
                p0, p1 = P[i], P[i+1]; m0, m1 = Tt[i], Tt[i+1]
                ts = np.linspace(0.0, 1.0, n_per_seg+1)
                h00 = (2*ts**3 - 3*ts**2 + 1)[:, None]
                h10 = (ts**3 - 2*ts**2 + ts)[:, None]
                h01 = (-2*ts**3 + 3*ts**2)[:, None]
                h11 = (ts**3 - ts**2)[:, None]
                seg = h00*p0 + h10*m0 + h01*p1 + h11*m1
                if i < N-2: seg = seg[:-1]
                out.append(seg)
            return np.vstack(out)

    # gradient segment drawer (used for EXTRAS only)
    def _line_segments(P: np.ndarray):
        return np.stack([P[:-1], P[1:]], axis=1)

    def _grad_line(ax, P: np.ndarray, c0: str, c1: str, lw: float, alpha: float):
        from matplotlib.colors import LinearSegmentedColormap
        segs = _line_segments(P)
        lc = LineCollection(segs, linewidths=lw, capstyle="round")
        cmap = LinearSegmentedColormap.from_list("tmp", [c0, c1])
        t = np.linspace(0.0, 1.0, max(2, len(segs)))
        lc.set_cmap(cmap); lc.set_array(t); lc.set_alpha(alpha)
        ax.add_collection(lc)
        return lc

    # hollow, sharp arrows
    def _add_arrows(ax, P: np.ndarray, color: str, n: int, ms: float, lw: float, alpha: float):
        N = len(P)
        if N < 2 or n <= 0: return
        idxs = np.linspace(1, N-1, num=min(n, max(1, N-1)), dtype=int)
        idxs = np.unique(np.clip(idxs, 1, N-1))
        for i in idxs:
            x0, y0 = P[i-1]; x1, y1 = P[i]
            arr = FancyArrowPatch(
                (x0, y0), (x1, y1),
                arrowstyle='-|>',
                mutation_scale=ms,
                linewidth=lw,
                facecolor="none",
                edgecolor=color,
                shrinkA=0.0, shrinkB=0.0,
                alpha=alpha,
                joinstyle="miter", capstyle="round",
                zorder=6
            )
            ax.add_patch(arr)

    # smoothed polylines (for drawing)
    P_gt_main_s   = _smooth_polyline(P_gt_main,   samples_per_step)
    P_lin_main_s  = _smooth_polyline(P_lin_main,  samples_per_step)
    P_full_main_s = _smooth_polyline(P_full_main, samples_per_step)

    P_gt_extra_s, P_lin_extra_s, P_full_extra_s = [], [], []
    for Pgt, Plin, Pfull in zip(P_gt_extra, P_lin_extra, P_full_extra):
        P_gt_extra_s.append(_smooth_polyline(Pgt,   samples_per_step))
        P_lin_extra_s.append(_smooth_polyline(Plin, samples_per_step))
        P_full_extra_s.append(_smooth_polyline(Pfull, samples_per_step))

    # arrow widths
    if lw_arrow_main is None: lw_arrow_main = max(0.6, 0.8 * lw_main)
    if lw_arrow_extra is None: lw_arrow_extra = max(0.5, 0.75 * lw_extra)

    # ===================== PLOTTING =====================
    rcParams.update({
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.formatter.use_mathtext": True,
    })
    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    # four spines
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(spine_lw)

    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize,
                direction="out", length=4.5, width=0.9)
    ax.tick_params(axis="both", which="minor", labelsize=tick_labelsize-1,
                direction="out", length=3.0, width=0.7)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))

    # limits BEFORE background
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    # -------- background --------
    if bg_mode == "discrete_map":
        U = bg["U"]; V = bg["V"]; XX = bg["grid_X"]; YY = bg["grid_Y"]
        ax.quiver(XX[::field_stride, ::field_stride],
                YY[::field_stride, ::field_stride],
                (U*field_scale)[::field_stride, ::field_stride],
                (V*field_scale)[::field_stride, ::field_stride],
                width=0.003, alpha=0.65, minlength=0.0, zorder=1)
    elif bg_mode == "vector_field":
        U = bg["U"]; V = bg["V"]; XX = bg["grid_X"]; YY = bg["grid_Y"]
        ax.quiver(XX[::field_stride, ::field_stride],
                YY[::field_stride, ::field_stride],
                (U*field_scale)[::field_stride, ::field_stride],
                (V*field_scale)[::field_stride, ::field_stride],
                width=0.003, alpha=0.70, minlength=0.0, zorder=1)
    elif bg_mode == "landscape":
        S = bg["S"]
        ax.imshow(
            S,
            extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
            origin="lower",
            cmap=_get_bg_cmap(bg_cmap),
            alpha=float(max(0.0, min(1.0, bg_alpha))),
            interpolation="bilinear",
            zorder=0,
            aspect="auto"
        )
        # overlay contour rings
        cx = np.linspace(xlim[0], xlim[1], S.shape[1])
        cy = np.linspace(ylim[0], ylim[1], S.shape[0])
        CCX, CCY = np.meshgrid(cx, cy)
        ax.contour(
            CCX, CCY, S,
            levels=int(contour_levels),
            colors=contour_color,
            linewidths=contour_lw,
            alpha=contour_alpha,
            zorder=1
        )

    # ------- MAIN (SOLID colors, no gradient) -------
    ax.plot(P_gt_main_s[:,0],   P_gt_main_s[:,1],   color=COL_GT,   lw=lw_main,  zorder=5)
    ax.plot(P_lin_main_s[:,0],  P_lin_main_s[:,1],  color=COL_LIN,  lw=lw_main,  zorder=5)
    ax.plot(P_full_main_s[:,0], P_full_main_s[:,1], color=COL_FULL, lw=lw_main,  zorder=5)
    _add_arrows(ax, P_gt_main_s,   COL_GT,   arrow_count_main,  arrow_size_main,  lw_arrow_main, 1.0)
    _add_arrows(ax, P_lin_main_s,  COL_LIN,  arrow_count_main,  arrow_size_main,  lw_arrow_main, 1.0)
    _add_arrows(ax, P_full_main_s, COL_FULL, arrow_count_main,  arrow_size_main,  lw_arrow_main, 1.0)

    # start markers (main) + optional per-step markers
    ax.scatter(P_gt_main[0,0],   P_gt_main[0,1],   s=36, color=COL_GT,   zorder=7)
    ax.scatter(P_lin_main[0,0],  P_lin_main[0,1],  s=36, color=COL_LIN,  zorder=7)
    ax.scatter(P_full_main[0,0], P_full_main[0,1], s=36, color=COL_FULL, zorder=7)
    if show_markers_main:
        ax.scatter(P_gt_main[:,0],   P_gt_main[:,1],   s=marker_size_main,  color=COL_GT,   alpha=marker_alpha_main,  zorder=8)
        ax.scatter(P_lin_main[:,0],  P_lin_main[:,1],  s=marker_size_main,  color=COL_LIN,  alpha=marker_alpha_main,  zorder=8)
        ax.scatter(P_full_main[:,0], P_full_main[:,1], s=marker_size_main,  color=COL_FULL, alpha=marker_alpha_main,  zorder=8)

    # ------- EXTRAS (gradient, thinner, NO markers by default) -------
    for Pgt, Plin, Pfull, Pgt_s, Plin_s, Pfull_s in zip(
        P_gt_extra, P_lin_extra, P_full_extra, P_gt_extra_s, P_lin_extra_s, P_full_extra_s
    ):
        _grad_line(ax, Pgt_s,   "#D9D9D9", COL_GT,   lw_extra, extra_alpha)
        _grad_line(ax, Plin_s,  "#FFDAB8", COL_LIN,  lw_extra, extra_alpha)
        _grad_line(ax, Pfull_s, "#D7EEFF", COL_FULL, lw_extra, extra_alpha)
        _add_arrows(ax, Pgt_s,   COL_GT,   arrow_count_extra, arrow_size_extra, lw_arrow_extra, extra_alpha)
        _add_arrows(ax, Plin_s,  COL_LIN,  arrow_count_extra, arrow_size_extra, lw_arrow_extra, extra_alpha)
        _add_arrows(ax, Pfull_s, COL_FULL, arrow_count_extra, arrow_size_extra, lw_arrow_extra, extra_alpha)

        if show_markers_extra:
            ax.scatter(Pgt[:,0],   Pgt[:,1],   s=marker_size_extra,  color=COL_GT,   alpha=marker_alpha_extra,  zorder=7)
            ax.scatter(Plin[:,0],  Plin[:,1],  s=marker_size_extra,  color=COL_LIN,  alpha=marker_alpha_extra,  zorder=7)
            ax.scatter(Pfull[:,0], Pfull[:,1], s=marker_size_extra,  color=COL_FULL, alpha=marker_alpha_extra,  zorder=7)

    # optional crosshair
    ax.axhline(0, lw=0.8, alpha=0.5); ax.axvline(0, lw=0.8, alpha=0.5)

    # save (dpi=450, no pad_inches)
    png_path = os.path.join(save_dir, f"{fname_prefix}.png")
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    plt.close(fig)

    # ---------- save data ----------
    data_path = os.path.join(save_dir, f"{fname_prefix}_data.pt")
    torch.save({
        "z_gt_int_main": z_gt_int_main[:steps_eff].detach().cpu(),
        "z_lin_int_main": z_lin_main.detach().cpu(),
        "z_full_int_main": z_full_main.detach().cpu(),
        "P_gt_main": torch.from_numpy(P_gt_main),
        "P_lin_main": torch.from_numpy(P_lin_main),
        "P_full_main": torch.from_numpy(P_full_main),
        "W": W.detach().cpu(),
        "mu": mu.detach().cpu(),
        "A_lin": (A_lin.detach().cpu() if A_lin is not None else None),
        "G_gen": (G_gen.detach().cpu() if G_gen is not None else None),
        "bg": bg,
        "extras": [
            {"seq_index": int(i),
            "P_gt": torch.from_numpy(p0),
            "P_lin": torch.from_numpy(p1),
            "P_full": torch.from_numpy(p2)}
            for i, p0, p1, p2 in zip(extra_seq_indices, P_gt_extra, P_lin_extra, P_full_extra)
        ],
        "meta": {
            "group": group,
            "seq_index_main": int(seq_index),
            "extra_seq_indices": [int(i) for i in extra_seq_indices],
            "steps_main": int(steps_eff),
            "bg_mode": bg_mode,
            "grid_res": int(grid_res),
            "field_stride": int(field_stride),
            "dt_eval": float(exp.dt_eval),
            "bg_cmap": bg_cmap,
            "bg_alpha": float(bg_alpha),
            "colors": {"gt": COL_GT, "lin": COL_LIN, "full": COL_FULL},
            "style": {
                "lw_main": float(lw_main), "lw_extra": float(lw_extra),
                "lw_arrow_main": float(lw_arrow_main), "lw_arrow_extra": float(lw_arrow_extra),
                "arrow_count_main": int(arrow_count_main),
                "arrow_count_extra": int(arrow_count_extra),
                "arrow_size_main": float(arrow_size_main),
                "arrow_size_extra": float(arrow_size_extra),
                "spine_lw": float(spine_lw),
                "tick_labelsize": int(tick_labelsize),
                "extra_alpha": float(extra_alpha),
                "smooth_curve": bool(smooth_curve),
                "samples_per_step": int(samples_per_step),
                "show_markers_main": bool(show_markers_main),
                "show_markers_extra": bool(show_markers_extra),
                "marker_size_main": float(marker_size_main),
                "marker_size_extra": float(marker_size_extra),
                "marker_alpha_main": float(marker_alpha_main),
                "marker_alpha_extra": float(marker_alpha_extra),
                "contour_levels": int(contour_levels),
                "contour_color": str(contour_color),
                "contour_lw": float(contour_lw),
                "contour_alpha": float(contour_alpha),
            }
        }
    }, data_path)

    print(f"[evolution] saved: {png_path}\n- data: {data_path}")

@torch.no_grad()
def plot_lowdim_time_series(
    exp,
    time_scale: float = 4.0,
    *,
    traj_id: int | None = None,
    t0: int | None = None,
    use_center: bool = True,
    use_whiten: bool = True,
    use_projector: bool = True,        # requires U_proj
    steps: int | None = None,
    save_dir: str = "./vis",
    fname_prefix: str = "lowdim_timeseries",
    # --- visual knobs ---
    lw: float = 3.0,                   # unified line width
    alpha: float = 1.0,                # unified alpha
    legend: bool = True,
    legend_max: int = 20,
    legend_loc: str = "upper left",    # inside-axes legend location
    downsample: int = 1,               # temporal stride on time axis
    # --- axes & layout ---
    hide_y_ticks: bool = True,         # remove y ticks/labels
    x_tick_step: float | None = None,  # major x tick step; None -> auto
    fig_w: float = 7.2,
    fig_h: float = 3.6,
    tick_labelsize: int = 13,
    # --- readability helpers (viz only) ---
    standardize: bool = False,         # per-dim z-score (only affects viz)
    smooth_window: int = 1,            # odd >=1; 1 = no smoothing
    # --- NEW: dimension selection ---
    max_dims: int | None = None,       # keep first K dims (after projection)
    dim_indices: list[int] | None = None,  # explicit dim indices to plot
    topk_by_var: int | None = None,    # choose K dims with largest variance (after viz ops)
    # --- output ---
    dpi: int = 450
):
    """
    Overlay time series of low-dimensional latent coordinates (after U_proj) on one axes.

    Downsampling:
        - 'downsample' is a stride on the time axis: keep every s-th frame (s>=1).

    Dimension limiting (priority):
        1) dim_indices: use exactly these dims (validated & de-duplicated)
        2) topk_by_var: pick K dims with largest variance on the displayed data
        3) max_dims:    keep first K dims (0..K-1)
        4) otherwise:   plot all dims

    Saves:
        - {fname_prefix}.png
        - {fname_prefix}_data.pt
        - {fname_prefix}_data.npz
        - {fname_prefix}_data.csv
    """
    import os, json, numpy as np, matplotlib.pyplot as plt, matplotlib.ticker as mticker
    from matplotlib import rcParams
    import torch, colorsys

    assert use_projector, "This function visualizes post-projection time series; set use_projector=True."
    assert getattr(exp, "use_projector", False) and (exp.U_proj is not None), \
        "U_proj not set. Train or load a projector first."

    os.makedirs(save_dir, exist_ok=True)

    # ------------------ 1) fetch one fixed sample ------------------
    loader = exp.sample_from_fix(traj_id, t0, steps)
    batch = next(iter(loader))
    t_full = batch["t"][0].to(exp.device)                # [T]
    t_eff  = t_full[exp.n_frames_cond - 1:] * time_scale

    # ------------------ 2) encode -> (center, whiten) -> project ------------------
    z_full, _, _ = exp._encode_and_recon(batch)          # [T',1,D]
    z_full = z_full[:, 0]                                 # [T',D]
    Tprime = int(z_full.size(0))
    if steps is None:
        steps = Tprime
    steps = max(2, min(int(steps), Tprime))

    z = z_full[:steps]
    if use_center:  z = exp._center_latent(z)
    if use_whiten:  z = exp._whiten_latent(z)
    z = exp._project_latent(z) if use_projector else z   # [T', d]
    z = z.detach().cpu().float()                          # [T', d]

    # ------------------ 3) time downsample / standardize / smooth ------------------
    s = max(1, int(downsample))
    z_plot = z[::s]                                       # [T_ds, d]
    t_plot = t_eff[:steps].detach().cpu().numpy()[::s]    # [T_ds]

    if standardize:
        mu = z_plot.mean(0, keepdim=True)
        sd = z_plot.std(0, keepdim=True).clamp_min(1e-8)
        z_plot = (z_plot - mu) / sd

    def _movavg(x: np.ndarray, w: int) -> np.ndarray:
        if w <= 1: return x
        if w % 2 == 0: w += 1
        pad = w // 2
        xpad = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
        kernel = np.ones((w,), dtype=np.float64) / float(w)
        return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), axis=0, arr=xpad)

    if int(smooth_window) > 1:
        z_plot = torch.from_numpy(_movavg(z_plot.numpy(), int(smooth_window))).float()

    Tds, d_all = z_plot.shape

    # ------------------ 4) choose which dims to plot ------------------
    # Priority: dim_indices > topk_by_var > max_dims > all
    if dim_indices is not None:
        idx_np = np.array([int(i) for i in dim_indices], dtype=int)
        idx_np = idx_np[(idx_np >= 0) & (idx_np < d_all)]
        # de-duplicate while keeping order
        _, first_pos = np.unique(idx_np, return_index=True)
        idx_np = idx_np[np.sort(first_pos)]
        if idx_np.size == 0:
            raise ValueError("dim_indices filtered to empty set (out of range?).")
    elif topk_by_var is not None:
        k = int(max(1, min(topk_by_var, d_all)))
        var = z_plot.var(dim=0) if Tds > 1 else torch.zeros(d_all)
        idx_np = torch.topk(var, k=k).indices.cpu().numpy()
        idx_np.sort()  # keep ascending order for labels
    elif max_dims is not None:
        k = int(max(1, min(max_dims, d_all)))
        idx_np = np.arange(k, dtype=int)
    else:
        idx_np = np.arange(d_all, dtype=int)

    z_plot = z_plot[:, idx_np]   # [T_ds, d_sel]
    d = z_plot.shape[1]

    # ------------------ 5) vivid color cycle (HSV with golden-ratio jumps) ------------------
    def bright_cycle(n: int):
        out = []
        phi = 0.6180339887498949  # golden ratio conjugate
        h = 0.0
        for _ in range(n):
            h = (h + phi) % 1.0
            out.append(colorsys.hsv_to_rgb(h, 0.95, 1.0))  # high saturation & brightness
        return out

    colors = bright_cycle(d)

    # ------------------ 6) figure style ------------------
    rcParams.update({
        "axes.formatter.use_mathtext": True,
        "font.size": tick_labelsize,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # full frame with consistent linewidth (show all four spines)
    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)

    # ticks & grids
    ax.set_xlim(float(t_plot[0]), float(t_plot[-1]))
    if x_tick_step is not None and x_tick_step > 0:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(base=float(x_tick_step)))
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, prune=None, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(4))

    ax.tick_params(axis="x", which="both", labelsize=tick_labelsize)
    ax.tick_params(axis="y", which="both", labelsize=tick_labelsize)

    if hide_y_ticks:
        ax.tick_params(axis="y", which="both", length=0)
        ax.set_yticklabels([])

    # dense grids (no y=0 baseline)
    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.55, alpha=0.25)

    # unified lines
    labels = []
    for j in range(d):
        ax.plot(
            t_plot, z_plot[:, j].numpy(),
            color=colors[j], lw=lw, alpha=alpha, solid_capstyle="round",
            label=(f"obs {int(idx_np[j])+1}" if (legend and d <= legend_max) else None)
        )
        labels.append(int(idx_np[j]))

    # legend INSIDE the axes
    if legend and d <= legend_max:
        ax.legend(
            loc=legend_loc, frameon=True, framealpha=0.85,
            facecolor="white", edgecolor="none",
            ncol=min(len(ax.get_legend_handles_labels()[1]), 4),
            fontsize=tick_labelsize - 1
        )

    fig.tight_layout()
    out_png = os.path.join(save_dir, f"{fname_prefix}.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    # ------------------ 7) save data (exact reproduction) ------------------
    meta = {
        "traj_id": int(traj_id) if traj_id is not None else None,
        "t0": int(t0) if t0 is not None else None,
        "steps": int(steps),
        "downsample": int(s),
        "use_center": bool(use_center),
        "use_whiten": bool(use_whiten),
        "use_projector": bool(use_projector),
        "standardize": bool(standardize),
        "smooth_window": int(smooth_window),
        "dpi": int(dpi),
        "dt_eval": float(getattr(exp, "dt_eval", 1.0)),
        "time_scale": float(time_scale),
        "hide_y_ticks": bool(hide_y_ticks),
        "x_tick_step": (None if x_tick_step is None else float(x_tick_step)),
        "fig_w": float(fig_w),
        "fig_h": float(fig_h),
        "tick_labelsize": int(tick_labelsize),
        "lw": float(lw),
        "alpha": float(alpha),
        "legend_loc": str(legend_loc),
        # NEW: record selection strategy & indices
        "max_dims": (None if max_dims is None else int(max_dims)),
        "topk_by_var": (None if topk_by_var is None else int(topk_by_var)),
        "dim_indices": [int(x) for x in labels],   # actual plotted original dim ids (after selection)
    }

    torch_path = os.path.join(save_dir, f"{fname_prefix}_data.pt")
    torch.save({"t": torch.from_numpy(np.asarray(t_plot)),
                "z_plot": z_plot.detach().cpu(),   # [T_ds, d_selected]
                "meta": meta}, torch_path)

    npz_path = os.path.join(save_dir, f"{fname_prefix}_data.npz")
    np.savez_compressed(
        npz_path,
        t=np.asarray(t_plot),
        z_plot=z_plot.numpy(),
        meta_json=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    )

    csv_path = os.path.join(save_dir, f"{fname_prefix}_data.csv")
    header = ",".join(["t"] + [f"dim_{k}" for k in labels])
    table  = np.concatenate([np.asarray(t_plot)[:, None], z_plot.numpy()], axis=1)
    np.savetxt(csv_path, table, delimiter=",", header=header, comments="", fmt="%.10g")

    print(f"[lowdim] saved figure: {out_png}")
    print(f"[lowdim] saved data  : {torch_path}")
    print(f"[lowdim] saved data  : {npz_path}")
    print(f"[lowdim] saved data  : {csv_path}")
