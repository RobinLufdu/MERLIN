import torch
import torch.nn.functional as F

"""
https://github.com/thuml/Neural-Solver-Library/blob/main/utils/loss.py
"""
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        bs = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(bs, -1) - y.view(bs, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        # x, y: [B, ...] tensors
        bs = x.size()[0]
        diff_norms = torch.norm(x.reshape(bs, -1) - y.reshape(bs, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(bs, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def masked_mse_loss(
    data1: torch.Tensor,
    data2: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    # data: [B, T, H, W, C], mask: [B, T, H, W, C] or None
    mse = F.mse_loss(data1, data2)
    if mask is None:
        return mse, mse

    sqerr = (data1 - data2).pow(2) * mask
    sqerr_sum = sqerr.sum(dim=(2, 3))
    denom = mask.sum(dim=(2, 3)).clamp(min=eps)
    mse_wzmask = (sqerr_sum / denom).mean()
    return mse, mse_wzmask


def fft_mse_on_frames(
    x_hat: torch.Tensor,   # [B, T, H, W, C]
    x_true: torch.Tensor,  # [B, T, H, W, C]
    *,
    use_log_mag: bool = True,
    hf_power: float = 0.0,
) -> torch.Tensor:
    """
    Frequency-domain MSE on (log-)magnitude spectra of 2D rFFT per frame.
    """
    B, T, H, W, C = x_hat.shape

    def flat_hw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 4, 2, 3).contiguous().view(B * T * C, H, W)

    Xh = flat_hw(x_hat)
    Xt = flat_hw(x_true)
    Xh = Xh - Xh.mean(dim=(1, 2), keepdim=True)
    Xt = Xt - Xt.mean(dim=(1, 2), keepdim=True)

    Fh = torch.fft.rfft2(Xh, norm="ortho")
    Ft = torch.fft.rfft2(Xt, norm="ortho")
    W_r = Fh.size(-1)

    Mh = torch.abs(Fh)
    Mt = torch.abs(Ft)
    if use_log_mag:
        Mh = torch.log1p(Mh)
        Mt = torch.log1p(Mt)

    if hf_power > 0.0:
        fy = torch.fft.fftfreq(H, d=1.0).to(Xh.device, Xh.dtype)
        fx = torch.fft.rfftfreq(W, d=1.0).to(Xh.device, Xh.dtype)
        gy = fy[:, None]
        gx = fx[None, :]
        r = torch.sqrt(gx * gx + gy * gy)
        r = r / (r.max() + 1e-12)
        w = r.pow(hf_power)
    else:
        w = Xh.new_ones((H, W_r))

    w = w.clone()
    w[0, 0] = 0.0
    Mh = Mh * w
    Mt = Mt * w

    def normalize(a: torch.Tensor) -> torch.Tensor:
        scale = a.mean(dim=(1, 2), keepdim=True)
        return a / (scale + 1e-6)

    return F.mse_loss(normalize(Mh), normalize(Mt))


def multiscale_spatial_mse(
    x_hat: torch.Tensor,  # [B, T, H, W, C]
    x_true: torch.Tensor, # [B, T, H, W, C]
    *,
    pool_scales: tuple[int, ...] = (2, 4),
) -> torch.Tensor:
    """
    Multi-scale MSE via an average-pooling pyramid.
    """
    if not pool_scales:
        return x_hat.new_zeros(())

    B, T, H, W, C = x_hat.shape
    xh = x_hat.permute(0, 1, 4, 2, 3).contiguous().view(B * T * C, 1, H, W)
    xt = x_true.permute(0, 1, 4, 2, 3).contiguous().view(B * T * C, 1, H, W)

    loss = x_hat.new_zeros(())
    n = 0
    for s in pool_scales:
        if H // s < 1 or W // s < 1:
            continue
        pool = torch.nn.AvgPool2d(kernel_size=s, stride=s, ceil_mode=False, count_include_pad=False)
        loss = loss + F.mse_loss(pool(xh), pool(xt))
        n += 1

    if n == 0:
        return x_hat.new_zeros(())
    return loss / n


def multistep_latent_consistency(
    latent_states: torch.Tensor,  # [T, B, D]
    A: torch.Tensor,              # [D, D]
    b: torch.Tensor | None = None,
    H: int = 4,
    gamma: float = 1.0,
) -> torch.Tensor:
    T, _, _ = latent_states.shape
    if H <= 0 or T < 2:
        return latent_states.new_tensor(0.0)
    H = min(H, T - 1)

    A_T = A.transpose(-1, -2)
    w_sum = 0.0
    loss_acc = latent_states.new_tensor(0.0)
    z_pred = latent_states[:-1]

    for h in range(1, H + 1):
        z_pred = z_pred @ A_T + (b if b is not None else 0.0)
        T_h = T - h
        z_use = z_pred[:T_h]
        target = latent_states[h:]
        mse_h = (z_use - target).pow(2).mean()
        w = gamma ** h
        loss_acc = loss_acc + w * mse_h
        w_sum += w
        if h < H:
            z_pred = z_pred[:-1]

    return loss_acc / max(w_sum, 1e-12)
