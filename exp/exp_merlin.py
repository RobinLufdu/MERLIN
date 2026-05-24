import os, json
from pprint import pformat
import torch
import torch.nn as nn
from utilities.utils import set_requires_grad
import torch.nn.functional as F

from data.data_process import PDEDataProcessor
from exp.exp_basic import Exp_Basic, ExpConfigs
from utilities import loader as loader_utils
from utilities.losses import (
    LpLoss,
    fft_mse_on_frames,
    masked_mse_loss,
    multiscale_spatial_mse,
    multistep_latent_consistency,
)
from utilities.data_proc import delay_stack_last_channel
from utilities.data_proc import index_points, mask_to_spatial_indices

from config import MERLINParamBundle
from config import build_encoder, build_decoder, build_latent


class Phase1LinearOperator(nn.Module):
    """Global affine one-step latent map used only in Phase-I joint-GD ablations.

    Convention matches the ridge solver in this file: for z in R^D,
        z_next = z @ A.T + b
    where A has shape [D, D] and b has shape [D].
    """
    def __init__(self, latent_dim: int, use_bias: bool = True, init: str = "zeros"):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.use_bias = bool(use_bias)
        self.A = nn.Parameter(torch.zeros(self.latent_dim, self.latent_dim))
        if self.use_bias:
            self.b = nn.Parameter(torch.zeros(self.latent_dim))
        else:
            self.register_parameter("b", None)
        self.reset_parameters(init=init)

    def reset_parameters(self, init: str = "zeros"):
        init = str(init).lower()
        with torch.no_grad():
            if init == "identity":
                self.A.zero_()
                self.A.add_(torch.eye(self.latent_dim, device=self.A.device, dtype=self.A.dtype))
            elif init == "kaiming":
                nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
            else:  # zeros
                self.A.zero_()
            if self.b is not None:
                self.b.zero_()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.A.T + (self.b if self.b is not None else 0.0)

    @torch.no_grad()
    def load_from_solution(self, A: torch.Tensor, b: torch.Tensor | None = None):
        self.A.copy_(A.to(device=self.A.device, dtype=self.A.dtype))
        if self.b is not None:
            if b is None:
                self.b.zero_()
            else:
                self.b.copy_(b.to(device=self.b.device, dtype=self.b.dtype))

    @torch.no_grad()
    def export(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        A = self.A.detach().clone()
        b = self.b.detach().clone() if self.b is not None else None
        return A, b



class Exp_MERLIN(Exp_Basic):
    def __init__(self, args, exp_cfg: ExpConfigs, model_cfg: MERLINParamBundle, data_processor: PDEDataProcessor):
        super(Exp_MERLIN, self).__init__(args, exp_cfg, model_cfg, data_processor)

        if hasattr(model_cfg, "as_model_kwargs"):
            self.model_cfg = model_cfg.as_model_kwargs(include_meta=True)
        elif isinstance(model_cfg, dict):
            self.model_cfg = model_cfg
        else:
            raise TypeError("model_cfg must be a MERLINParamBundle or a dict loaded from model_cfg.json")
        assert self.model_cfg["n_frames_cond"] == data_processor.n_frames_cond

        self.is_2d = (len(self.shapelist) == 2)
        self.state_dim  = self.model_cfg["state_dim"]
        self.latent_dim = self.model_cfg["latent_dim"]
        self.code_dim   = self.model_cfg["code_dim"]
        self.memory_type = self.model_cfg["latent_process_discrete"].get("memory_type", "leaky")
        
        # Optional Phase-I joint-GD ablation with a global latent affine map.
        self.phase1_linear = None
        self.optim_phase1_lin = None
        self.scheduler_phase1_lin = None
        
        # dataloader, initialized in Exp_Basic
        # teacher forcing params
        self.tf_epsilon, self.epsilon = exp_cfg.tf_epsilon, exp_cfg.epsilon
        self.dt_eval = exp_cfg.dt_eval

        self.enc_mode = self.cfg.enc_mode
        self.loss_with_mask = self.cfg.loss_with_mask

        # loss weights
        self.lambda_dyn, self.lambda_pred = self.cfg.lambda_dyn, self.cfg.lambda_pred
        self.lambda_corr = self.cfg.lambda_corr
        self.lambda_resid = getattr(self.cfg, "lambda_residual", 0.0)

        self.use_diag_whiten = getattr(self.cfg, "use_diag_whiten", True)
        self.whiten_scale = None  # [D]

        ################# Params for Low-dimensional Projectors #################
        self.U_proj = None           # [D, d] or None
        self.use_projector = False  
        self.latent_dim_y = None     # d

        # load model
        self.load_model()
        self.log_param_table()

    
    def build_dataloader(self, group: str):
        sample_map = {
            "train": self.train_sample_idx,
            "train_eval": self.train_eval_sample_idx,
            "test": self.test_sample_idx,
        }
        dataloader = self.data_processor.get_dataloader(group=group, samples=sample_map[group])
        if group == "train":
            self.train_loader = dataloader
        elif group == "train_eval":
            self.train_eval_loader = dataloader
        elif group == "test":
            self.test_loader = dataloader

    
    def load_model(self):
        if self.enc_mode == "galerkin_transformer":
            encoder_cfg = self.model_cfg["encoder"]
        elif self.enc_mode == "set_transformer":
            encoder_cfg = self.model_cfg["set_encoder"]
        else:
            raise ValueError(f"Unknown enc_mode: {self.enc_mode}")

        self.encoder = build_encoder(model_cfg=encoder_cfg, enc_mode=self.enc_mode).to(self.device)
        self.decoder = build_decoder(model_cfg=self.model_cfg["fourier_decoder"]).to(self.device)
        self.latent_process = build_latent(model_cfg=self.model_cfg["latent_process_discrete"]).to(self.device)

    
    def count_parameters(self) -> dict:
        """Return a dict with per-module and total trainable parameter counts."""
        def ntrainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        counts = {
            "encoder": ntrainable(self.encoder) if hasattr(self, "encoder") else 0,
            "latent_process": ntrainable(self.latent_process) if hasattr(self, "latent_process") else 0,
            "decoder": ntrainable(self.decoder) if hasattr(self, "decoder") else 0,
        }
        counts["total"] = sum(counts.values())
        return counts


    def log_param_table(self, title: str = "Trainable parameters"):
        c = self.count_parameters()
        lines = [
            f"{title}:",
            f"  encoder       : {c['encoder']:,}",
            f"  latent_process: {c['latent_process']:,}",
            f"  decoder       : {c['decoder']:,}",
            f"  ---------------------------",
            f"  TOTAL         : {c['total']:,}",
        ]
        msg = "\n".join(lines)
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    
    def switch_to_train(self):
        self.encoder.train()
        self.decoder.train()
        self.latent_process.train()

    
    def switch_to_eval(self):
        self.encoder.eval()
        self.decoder.eval()
        self.latent_process.eval()
        

    def init_optim(self):
        self.optim_enc = torch.optim.Adam([{'params': self.encoder.parameters(), 'lr': self.lr}])
        self.optim_dec = torch.optim.Adam([{'params': self.decoder.parameters(), 'lr': self.lr}])
        self.optim_dyn = torch.optim.Adam([{'params': self.latent_process.parameters(), 'lr': self.lr, 'weight_decay': self.cfg.weight_decay}])

        if self.cfg.scheduler == 'OneCycleLR':
            self.scheduler_enc = torch.optim.lr_scheduler.OneCycleLR(self.optim_enc, max_lr=self.cfg.lr, epochs=self.cfg.epochs,
                                                            steps_per_epoch=len(self.train_loader),
                                                            pct_start=self.cfg.pct_start)
            self.scheduler_dec = torch.optim.lr_scheduler.OneCycleLR(self.optim_dec, max_lr=self.cfg.lr, epochs=self.cfg.epochs,
                                                            steps_per_epoch=len(self.train_loader),
                                                            pct_start=self.cfg.pct_start)
            self.scheduler_dyn = torch.optim.lr_scheduler.OneCycleLR(self.optim_dyn, max_lr=self.cfg.lr, epochs=self.cfg.epochs,
                                                            steps_per_epoch=len(self.train_loader),
                                                            pct_start=self.cfg.pct_start)
        elif self.cfg.scheduler == 'CosineAnnealingLR':
            self.scheduler_enc = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_enc, T_max=self.cfg.epochs)
            self.scheduler_dec = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_dec, T_max=self.cfg.epochs)
            self.scheduler_dyn = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_dyn, T_max=self.cfg.epochs)
        elif self.cfg.scheduler == 'StepLR':
            self.scheduler_enc = torch.optim.lr_scheduler.StepLR(self.optim_enc, step_size=self.cfg.step_size, gamma=self.cfg.gamma)
            self.scheduler_dec = torch.optim.lr_scheduler.StepLR(self.optim_dec, step_size=self.cfg.step_size, gamma=self.cfg.gamma)
            self.scheduler_dyn = torch.optim.lr_scheduler.StepLR(self.optim_dyn, step_size=self.cfg.step_size, gamma=self.cfg.gamma)
    

    def _init_phase1_joint_linear(self, use_bias: bool = True, init: str = "zeros"):
        """Jointly optimize the global linear operator with the function autoencoders."""
        self.phase1_linear = Phase1LinearOperator(self.latent_dim, use_bias=use_bias, init=init).to(self.device)

        lr_lin = self.cfg.lr if (getattr(self.cfg, "lr_phase1_linear", None) is None) else self.cfg.lr_phase1_linear
        wd_lin = getattr(self.cfg, "wd_phase1_linear", 0.0)
        self.optim_phase1_lin = torch.optim.Adam(
            [{'params': self.phase1_linear.parameters(), 'lr': lr_lin, 'weight_decay': wd_lin}]
        )

        if self.cfg.scheduler == 'OneCycleLR':
            self.scheduler_phase1_lin = torch.optim.lr_scheduler.OneCycleLR(
                self.optim_phase1_lin, max_lr=lr_lin, epochs=self.cfg.epochs,
                steps_per_epoch=len(self.train_loader), pct_start=self.cfg.pct_start
            )
        elif self.cfg.scheduler == 'CosineAnnealingLR':
            self.scheduler_phase1_lin = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim_phase1_lin, T_max=self.cfg.epochs
            )
        elif self.cfg.scheduler == 'StepLR':
            self.scheduler_phase1_lin = torch.optim.lr_scheduler.StepLR(
                self.optim_phase1_lin, step_size=self.cfg.step_size, gamma=self.cfg.gamma
            )


    @torch.no_grad()
    def _get_phase1_joint_linear_Ab(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.phase1_linear is None:
            raise RuntimeError("Phase-I joint linear operator has not been initialized.")
        return self.phase1_linear.export()


    def _build_encoder_inputs(
        self,
        ground_truth: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Build the point-set encoder inputs.

        Returns
        -------
        data_in : [B*T_eff, S_or_N, C_in + coord_dim]
        pos_in  : [B*T_eff, S_or_N, coord_dim]
        bs      : batch size
        t_eff   : number of delay-stacked latent frames
        """
        ground_truth = ground_truth.to(self.device)    # [B, T, (*spatial_dims), C]
        if masks is not None:
            masks = masks.to(self.device)

        bs, seq_len = ground_truth.shape[0], ground_truth.shape[1]
        t_eff = seq_len - self.n_frames_cond + 1
        if t_eff <= 0:
            raise ValueError(
                f"Need at least n_frames_cond={self.n_frames_cond} frames, got seq_len={seq_len}."
            )

        delay_data = delay_stack_last_channel(x=ground_truth, d=self.n_frames_cond)
        data_in = delay_data.flatten(0, 1)
        data_in = data_in.reshape(data_in.shape[0], -1, data_in.shape[-1])

        pos_base = self.pos_feat.reshape(-1, self.pos_feat.shape[-1]).to(self.device)
        pos_in = pos_base.unsqueeze(0).expand(data_in.shape[0], -1, -1)

        if self.is_2d:
            mask_index = mask_to_spatial_indices(mask=masks)
            mask_index = mask_index.unsqueeze(1).expand(-1, t_eff, -1).flatten(0, 1)
            data_in = index_points(data_in, mask_index)
            pos_in = index_points(pos_in, mask_index)

        return data_in, pos_in, bs, t_eff


    def _encode_latent_from_field(
        self,
        ground_truth: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a field sequence into latent states [T_eff, B, D]."""
        data_in, pos_in, bs, t_eff = self._build_encoder_inputs(ground_truth, masks)
        latent_token = self.encoder(data_in, pos_in)
        latent_states = latent_token.reshape((bs, t_eff, -1))
        return latent_states.permute(1, 0, 2)


    def _encode_latent_sequence(self, batch) -> torch.Tensor:
        """Encode the full batch sequence."""
        ground_truth = batch["data"].to(self.device)
        masks = batch["mask"].to(self.device) if self.is_2d else None
        return self._encode_latent_from_field(ground_truth, masks)
    

    def _decode_latent(self, latent_seqs: torch.Tensor):
        """
        Decode a latent sequence a_seq [T',B,D] into field tensor aligned per your decoder.
        Returns [B,T',H,W,C] (or your state_dim layout).
        """
        T1, B, D = latent_seqs.shape
        latent_feats_ = latent_seqs.permute(1, 0, 2).flatten(0, 1)    # [B*T', latent_dim]
        grid_dim = self.pos_feat.shape[-1]
        grid = self.pos_feat.reshape(-1, grid_dim).to(self.device)    # [N_pt, grid_dim]
        recon_field_ = self.decoder(grid=grid, latent_feat=latent_feats_)    # [B*T', N_pts, out_dim]
        return recon_field_.reshape(B, T1, *self.shapelist, self.state_dim)
    

    def _encode_and_recon(self, batch, return_recon: bool = False):
        ground_truth = batch["data"].to(self.device)    # [B, T, (*spatial_dims), n_ch]
        masks = batch["mask"].to(self.device) if self.is_2d else None           # [B, T, H, W, n_ch]
        latent_states = self._encode_latent_from_field(ground_truth, masks)

        recon_encdec = self._decode_latent(latent_seqs=latent_states)    # [B, t, H, W, c]
        if self.is_2d:
            recon_loss, recon_loss_wz_mask = masked_mse_loss(recon_encdec, ground_truth[:, self.n_frames_cond-1:, ...],
                                                             mask=masks[:, self.n_frames_cond-1:, ...])
        else:
            recon_loss, recon_loss_wz_mask = masked_mse_loss(recon_encdec, ground_truth[:, self.n_frames_cond-1:, ...]) 
        if return_recon:
            # Return both losses and tensors needed for spectral/multiscale penalties
            x_rec   = recon_encdec
            x_gt    = ground_truth[:, self.n_frames_cond - 1:, ...]
            m_rec   = masks[:, self.n_frames_cond - 1:, ...] if self.is_2d else None
            return latent_states, recon_loss, recon_loss_wz_mask, x_rec, x_gt, m_rec
        else:
            return latent_states, recon_loss, recon_loss_wz_mask

    
    @torch.no_grad()    # for evaluation
    def _encode_cond_batch(self, batch):
        ground_truth = batch["data"].to(self.device)    # [B, T, H, W, n_ch]
        masks = batch["mask"].to(self.device) if self.is_2d else None           # [B, T, H, W, n_ch]
        masks_cond = masks[:, :self.n_frames_cond, ...] if masks is not None else None
        latent_states = self._encode_latent_from_field(
            ground_truth[:, :self.n_frames_cond, ...],
            masks_cond,
        )
        return latent_states[0]


    @staticmethod
    @torch.no_grad()
    def solve_Ab_ridge(Z0: torch.Tensor, Zp: torch.Tensor, ridge: float = 1e-3, add_bias: bool = True):
        """
        Closed-form ridge regression for one-step latent linear dynamics:
        min_{A,b} ||Zp - (A Z0 + b)||_F^2 + ridge * ||[A b]||_F^2
        Inputs:
        Z0: [N, D]  flattened (t,b) -> N samples at time t
        Zp: [N, D]  flattened (t,b) -> N samples at time t+1
        Returns:
        A*: [D, D], b*: [D] or None
        """
        N, D = Z0.shape
        if add_bias:
            X = torch.cat([Z0, torch.ones(N, 1, device=Z0.device, dtype=Z0.dtype)], dim=1)  # [N, D+1]
            p = D + 1
        else:
            X = Z0; p = D
        Y = Zp                                          # [N, D]
        G = X.T @ X + ridge * torch.eye(p, device=X.device, dtype=X.dtype)   # [p, p]
        YXt = Y.T @ X                                   # [D, p]
        Theta = torch.linalg.solve(G, YXt.T).T          # [D, p]
        A = Theta[:, :D]                                # [D, D]
        b = Theta[:, D] if add_bias else None           # [D]
        return A, b
    

    @staticmethod
    @torch.no_grad()
    def ema_update_stats(Sxx: torch.Tensor, Syx: torch.Tensor,
                        Z0: torch.Tensor, Zp: torch.Tensor,
                        ema_beta: float = 0.95, add_bias: bool = True):
        """
        EMA sufficient statistics across batches:
        Sxx ≈ E[X^T X],  Syx ≈ E[Y^T X], where X=[Z0; 1] (if add_bias)
        In-place updates with EMA.
        """
        N, D = Z0.shape
        if add_bias:
            X = torch.cat([Z0, torch.ones(N, 1, device=Z0.device, dtype=Z0.dtype)], dim=1)  # [N, D+1]
        else:
            X = Z0
        Y = Zp
        Sxx.mul_(ema_beta).add_(X.T @ X, alpha=(1 - ema_beta))
        Syx.mul_(ema_beta).add_(Y.T @ X, alpha=(1 - ema_beta))
        return Sxx, Syx


    @staticmethod
    @torch.no_grad()
    def solve_global_A_from_stats(Sxx: torch.Tensor, Syx: torch.Tensor,
                                ridge: float = 1e-3, D: int | None = None, add_bias: bool = True):
        """
        Solve global Ad_phase1 (+ b_phase1) from EMA sufficient statistics.
        """
        p = Sxx.size(0)
        I = torch.eye(p, device=Sxx.device, dtype=Sxx.dtype)
        Theta = torch.linalg.solve(Sxx + ridge * I, Syx.T).T  # [D, p]
        if D is None:
            D = Theta.size(0)
        Ad_phase1 = Theta[:, :D]
        b_phase1 = Theta[:, D] if add_bias else None
        return Ad_phase1, b_phase1
    
    
    @torch.no_grad()
    def solve_global_A_fullpass(self, dataloader, ridge: float = 5e-3, use_bias: bool = True):
        enc_was_train = self.encoder.training
        dec_was_train = self.decoder.training
        self.encoder.eval(); self.decoder.eval()
        Z0_list, Zp_list = [], []
        for batch in dataloader:
            latent_states = self._encode_latent_sequence(batch)  # [T',B,D]
            T1, B, D = latent_states.shape
            if T1 <= 1:
                continue
            Z0_list.append(latent_states[:-1].reshape(-1, D))
            Zp_list.append(latent_states[ 1:].reshape(-1, D))

        if len(Z0_list) == 0:
            raise RuntimeError("[solve_global_A_fullpass] No (Z0,Zp) pairs collected. Check dataloader or n_frames_cond.")
        Z0 = torch.cat(Z0_list, dim=0).to(self.device)
        Zp = torch.cat(Zp_list, dim=0).to(self.device)
        A_full, b_full = self.solve_Ab_ridge(Z0, Zp, ridge=ridge, add_bias=use_bias)

        if enc_was_train: self.encoder.train()
        if dec_was_train: self.decoder.train()
        return A_full.detach(), (b_full.detach() if b_full is not None else None)


    def train_phase1_linear(self, 
                            epochs: int,
                            ridge: float = 5e-3,
                            ema_beta: float = 0.97,
                            use_bias: bool = True,
                            use_pred_loss: bool = True,
                            lambda_pred: float = 0.1,
                            lambda_dyn: float = 0.05,
                            log_every: int | None = 5,
                            eval_every: int | None = 20,
                            verbose: str = None,
                            ms_consistency_enable: bool = False,         # Add latent multi-step consistency.
                            freq_ms_enable: bool = False,                # Add spectral and multi-scale reconstruction penalties.
                            global_A_mode: str | None = None,            # "fullpass" or "ema"; only used in ridge mode.
                            ):
        """
        Phase-I: train encoder/decoder together with a latent one-step linear backbone.

        Modes:
        - ridge:
          Fit a batch-local affine map (A*, b*) with closed-form ridge regression.
          Gradients from the latent and decoded prediction losses update only the
          encoder/decoder. A global operator is exported at epoch end using either
          a full pass over the train loader or EMA sufficient statistics.
        - joint_gd:
          Optimize one trainable global affine map jointly with the encoder/decoder.

        The exported Phase-I operator is stored in self.Ad_phase1/self.b_phase1 and
        saved to Ad_phase1.pt for Phase-II initialization.
        """

        mode = (global_A_mode or getattr(self.cfg, "global_A_mode", "fullpass")).lower()
        assert mode in {"fullpass", "ema"}, f"global_A_mode must be 'fullpass' or 'ema', got {mode}"
        phase1_mode = str(getattr(self.cfg, "phase1_linear_mode", "ridge")).lower()
        assert phase1_mode in {"ridge", "joint_gd"}, f"Unsupported phase1_linear_mode={phase1_mode}"

        self.setup_logger()
        self.save_repro_artifacts()
        self.log_param_table()
        assert self.data_processor.mode == "interpolation", "Mismatched dataloaders"
        self.build_dataloader(group="train")
        if eval_every is not None:
            self.build_dataloader(group="test")
            self.build_dataloader(group="train_eval")
        self._save_split_and_samples()

        self.encoder.train()
        self.decoder.train()
        set_requires_grad(self.latent_process, False) 
        self.init_optim()
        if phase1_mode == "joint_gd":
            self._init_phase1_joint_linear(use_bias=use_bias, init="zeros")
            self.phase1_linear.train()
            self.logger.info(
                f"[Phase-I] using joint-GD affine backbone | lr={self.optim_phase1_lin.param_groups[0]['lr']:.3e} | "
                f"wd={self.optim_phase1_lin.param_groups[0]['weight_decay']:.3e}"
            )
        else:
            self.logger.info(f"[Phase-I] using closed-form ridge estimated backbone | global_A_mode={mode} | ridge={ridge:.3e}")

        if verbose is not None:
            self.logger.info(f"{verbose}")

        if (phase1_mode == "ridge") and (mode == "ema"):
            D = self.latent_dim
            p = D + (1 if use_bias else 0)
            Sxx = torch.zeros(p, p, device=self.device)
            Syx = torch.zeros(D, p, device=self.device)
        else:
            Sxx = Syx = None

        num_epochs = epochs if epochs is not None else self.cfg.epochs
        best_rec, best_metrics = float("inf"), None

        for epoch in range(1, num_epochs + 1):
            for it, batch in enumerate(self.train_loader):
                masks = batch["mask"].to(self.device) if self.is_2d else None  # [B, T, H, W, n_ch], none for 1d
                # 1) Encode the sequence and reconstruct the encoded frames.
                latent_states, recon_loss, recon_loss_wzmask, x_rec, x_gt, _ = self._encode_and_recon(batch, return_recon=True)
                T, B, D = latent_states.shape

                # 2) Build all one-step latent pairs z_t -> z_{t+1}.
                Z0 = latent_states[:-1].reshape(-1, D)  # [N, D]
                Zp = latent_states[ 1:].reshape(-1, D)  # [N, D]
                
                # 3) Obtain the one-step affine map for this batch.
                if phase1_mode == "ridge":
                    with torch.no_grad():
                        A_use, b_use = self.solve_Ab_ridge(Z0, Zp, ridge=ridge, add_bias=use_bias)
                    Zp_hat = (Z0 @ A_use.T) + (b_use if (use_bias and b_use is not None) else 0.0)
                else:
                    A_use = self.phase1_linear.A
                    b_use = self.phase1_linear.b if use_bias else None
                    Zp_hat = self.phase1_linear(Z0)

                # 4) Latent one-step consistency: z_{t+1} ~= A z_t + b.
                dyn_loss = F.mse_loss(Zp_hat, Zp)

                # 5) Optional decoded one-step prediction loss.
                if use_pred_loss:
                    a1_hat = Zp_hat.view(T - 1, B, D)                  # [T-1,B,D]
                    x1_hat = self._decode_latent(a1_hat)               # [B,T-1,H,W,C]
                    x1_true = batch["data"][:, self.n_frames_cond:, ...].to(self.device)
                    if self.is_2d:
                        pred_loss, pred_loss_wzmask = masked_mse_loss(x1_hat, x1_true, masks[:, self.n_frames_cond:, ...])
                    else: 
                        pred_loss, pred_loss_wzmask = masked_mse_loss(x1_hat, x1_true)
                else:
                    pred_loss, pred_loss_wzmask = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)

                # 6) Optional latent multi-step consistency under the same affine map.
                lambda_lt_pred = self.cfg.lambda_lt_pred if (self.cfg.lambda_lt_pred is not None) and ms_consistency_enable else 0.0
                if ms_consistency_enable and (lambda_lt_pred > 0.0):
                    ms_loss = multistep_latent_consistency(
                        latent_states,
                        A_use,
                        b_use if (use_bias and b_use is not None) else None,
                        H=self.cfg.rollout_steps,
                        gamma=self.cfg.gamma_decay,
                    )
                else:
                    ms_loss = torch.tensor(0.0, device=self.device)

                # 7) Optional frequency and multi-scale reconstruction penalties.
                lambda_freq = self.cfg.lambda_freq if (self.cfg.lambda_freq is not None) and freq_ms_enable else 0.0
                if freq_ms_enable and (lambda_freq > 0.0):
                    xf_hat, xf_true = x_rec, x_gt  # use reconstruction frames
                    fft_loss = fft_mse_on_frames(
                        xf_hat, xf_true,
                        use_log_mag=True,
                        hf_power=self.cfg.freq_hf_power
                    )
                    ms_loss_img = multiscale_spatial_mse(
                        xf_hat, xf_true,
                        pool_scales=self.cfg.ms_pool_scales
                    )
                    freq_ms_loss = fft_loss + ms_loss_img
                else:
                    freq_ms_loss = torch.tensor(0.0, device=self.device)

                # 8) Combine losses and update trainable Phase-I modules.
                loss = recon_loss_wzmask + lambda_dyn * dyn_loss + lambda_pred * pred_loss_wzmask
                loss += lambda_lt_pred * ms_loss
                loss += lambda_freq * freq_ms_loss

                self.optim_enc.zero_grad()
                self.optim_dec.zero_grad()
                if phase1_mode == "joint_gd":
                    self.optim_phase1_lin.zero_grad()
                loss.backward()
                if self.cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.cfg.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.cfg.max_grad_norm)
                    if phase1_mode == "joint_gd":
                        torch.nn.utils.clip_grad_norm_(self.phase1_linear.parameters(), self.cfg.max_grad_norm)
                self.optim_enc.step()
                self.optim_dec.step()
                if phase1_mode == "joint_gd":
                    self.optim_phase1_lin.step()
                if self.cfg.scheduler == "OneCycleLR":
                    self.scheduler_enc.step()
                    self.scheduler_dec.step()
                    if phase1_mode == "joint_gd":
                        self.scheduler_phase1_lin.step()
                if (phase1_mode == "ridge") and (mode == "ema"):
                    # Update global-map statistics.
                    self.ema_update_stats(Sxx, Syx, Z0, Zp, ema_beta=ema_beta, add_bias=use_bias)
    
                if log_every and ((epoch * len(self.train_loader) + it) % log_every == 0):
                    extra = ""
                    if phase1_mode == "joint_gd":
                        with torch.no_grad():
                            A_now, _ = self._get_phase1_joint_linear_Ab()
                            extra = f" | rho(A_joint) {self._spectral_radius(A_now):.6f}"
                    self.logger.info(
                        f"[Phase-I/{phase1_mode}] epoch {epoch:03d} it {it:04d} | "
                        f"rec {recon_loss.item():.8f} | dyn {dyn_loss.item():.8f} | pred {pred_loss.item():.8f} | "
                        f"rec(mask) {recon_loss_wzmask.item():.8f} | pred(mask) {pred_loss_wzmask.item():.8f} | "
                        f"dyn(long) {ms_loss.item():.8f} | freq_ms_loss {freq_ms_loss.item():.8f}{extra}"
                     )

            # Epoch end: compute global Ad_phase1 (+ b_phase1) for Phase-II init/logging
            with torch.no_grad():
                if phase1_mode == "joint_gd":
                    A_glob, b_glob = self._get_phase1_joint_linear_Ab()
                elif mode == "fullpass":
                    A_glob, b_glob = self.solve_global_A_fullpass(self.train_loader, ridge=ridge, use_bias=use_bias)
                else:
                    A_glob, b_glob = self.solve_global_A_from_stats(Sxx, Syx, ridge=ridge, D=D, add_bias=use_bias)
                self.Ad_phase1 = A_glob.clone()
                self.b_phase1 = b_glob.clone() if b_glob is not None else None
                out_dir = os.path.join(self.cfg.out_dir, f"{self.run_id}")
                torch.save({'Ad_phase1': self.Ad_phase1, 'b_phase1': self.b_phase1}, os.path.join(out_dir, "Ad_phase1.pt"))
            if self.cfg.scheduler == 'CosineAnnealingLR' or self.cfg.scheduler == 'StepLR':
                self.scheduler_enc.step()
                self.scheduler_dec.step()
                if phase1_mode == "joint_gd":
                    self.scheduler_phase1_lin.step()

            ###################### Evaluation ######################
            if eval_every is not None and epoch % eval_every == 0:
                if hasattr(self, "train_eval_loader") and self.train_eval_loader is not None:
                    metrics_tr = self.evaluate_phase1(self.train_eval_loader, use_global_A=True)
                    self.logger.info(
                        f"[P1/Eval-TrainEval/{phase1_mode}] rec(mask)={metrics_tr['rec_masked']:.8f} | "
                        f"dyn={metrics_tr['dyn_mse']:.8f} | pred(mask)={metrics_tr['pred_masked']:.8f} | "
                        f"diag={metrics_tr['diag']}"
                    )
                    if metrics_tr["rec_masked"] < best_rec:
                        best_rec = metrics_tr["rec_masked"]
                        best_metrics = {"split": "train_eval", "phase1_linear_mode": phase1_mode, **metrics_tr}
                        self.save_phase1_checkpoint(epoch=epoch, metrics=best_metrics, pth_name="phase1_best_rec.pth")
                if hasattr(self, "test_loader") and self.test_loader is not None:
                    metrics_ts = self.evaluate_phase1(self.test_loader, use_global_A=True)
                    self.logger.info(
                        f"[P1/Eval-Test/{phase1_mode}] rec(mask)={metrics_ts['rec_masked']:.8f} | "
                        f"dyn={metrics_ts['dyn_mse']:.8f} | pred(mask)={metrics_ts['pred_masked']:.8f} | "
                        f"diag={metrics_ts['diag']}"
                    )
        self.logger.info(f"[Phase-I/{phase1_mode}] finished. Exported discrete one-step operator saved to self.Ad_phase1.")
        self.save_phase1_checkpoint(epoch=None, metrics=best_metrics, pth_name="phase1_final.pth")

    
    @torch.no_grad()
    def evaluate_phase1(self,
                        dataloader=None,
                        *,
                        use_global_A: bool = True,   # use Ad_phase1 if available; else fall back to per-batch closed-form
                        use_bias: bool = True,
                        use_pred_loss: bool = True,  # include one-step decoded loss
                        ridge: float = 5e-3):
        """
        Phase-I evaluation.
        Metrics:
        - rec_wzmask: masked recon MSE on frames [nf_cond-1 : T-1]
        - dyn_mse   : latent linear consistency MSE (Z_{t+1} vs A Z_t + b)
        - pred_wzmask (optional): one-step decoded masked MSE using A,b
        - (optional) spectral stats of Ad_phase1 if used

        If use_global_A=True and self.Ad_phase1 exists, we use that fixed A (+b) for all batches;
        otherwise we solve closed-form A*, b* per batch.
        """
        self.encoder.eval()
        self.decoder.eval()

        if dataloader is None:
            if hasattr(self, "test_loader") and self.test_loader is not None:
                dataloader = self.test_loader
            elif hasattr(self, "train_eval_loader") and self.train_eval_loader is not None:
                dataloader = self.train_eval_loader
            else:
                try:
                    self.build_dataloader(group="test")
                    dataloader = self.test_loader
                except Exception:
                    self.build_dataloader(group="train_eval")
                    dataloader = self.train_eval_loader

        tot_rec_m = 0.0   # masked recon
        tot_dyn   = 0.0   # latent linear consistency
        tot_pred_m = 0.0  # masked one-step decoded
        nsamples  = 0

        # cache global A if requested and available
        Ad_fix = None
        b_fix = None
        if use_global_A and hasattr(self, "Ad_phase1") and (self.Ad_phase1 is not None):
            Ad_fix = self.Ad_phase1.to(self.device, dtype=torch.float32)
            b_fix = (self.b_phase1.to(self.device, dtype=torch.float32)
                    if getattr(self, "b_phase1", None) is not None else None)
        elif use_global_A and (getattr(self, "phase1_linear", None) is not None):
            Ad_fix, b_fix = self._get_phase1_joint_linear_Ab()
            Ad_fix = Ad_fix.to(self.device, dtype=torch.float32)
            b_fix = (b_fix.to(self.device, dtype=torch.float32) if b_fix is not None else None)

        for batch in dataloader:
            masks = batch["mask"].to(self.device) if self.is_2d else None           # [B, T, H, W, n_ch]
            # 1) encode & recon: latent_states [t_eff,B,D], recon loss on [nf_cond-1:]
            latent_states, recon_loss, recon_loss_wzmask = self._encode_and_recon(batch)  # recon_loss_wzmask is masked MSE
            T_eff, B, D = latent_states.shape
            nsamples += B

            # 2) closed-form A,b (per-batch) OR fixed Ad_phase1,b_phase1
            Z0 = latent_states[:-1].reshape(-1, D)  # [N, D], N=(T_eff-1)*B
            Zp = latent_states[ 1:].reshape(-1, D)  # [N, D]
            if Ad_fix is None:
                A_use, b_use = self.solve_Ab_ridge(Z0, Zp, ridge=ridge, add_bias=use_bias)
            else:
                A_use, b_use = Ad_fix, b_fix

            # 3) latent linear consistency loss: MSE(Zp_hat, Zp)
            Zp_hat = (Z0 @ A_use.T) + (b_use if (use_bias and b_use is not None) else 0.0)
            dyn_mse = F.mse_loss(Zp_hat, Zp)

            # 4) optional: one-step decoded loss in observation space
            if use_pred_loss:
                a1_hat = Zp_hat.view(T_eff - 1, B, D)                # [T_eff-1, B, D]
                x1_hat = self._decode_latent(a1_hat)                 # [B, T_eff-1, H, W, C]
                x1_true = batch["data"][:, self.n_frames_cond:, ...].to(self.device)  # next frames [nf_cond : T-1]
                if self.is_2d:
                    pred_mse, pred_mse_wzmask = masked_mse_loss(x1_hat, x1_true, masks[:, self.n_frames_cond:, ...])
                else:
                    pred_mse, pred_mse_wzmask = masked_mse_loss(x1_hat, x1_true)
            else:
                pred_mse_wzmask = torch.tensor(0.0, device=self.device)

            # accumulate (weight by batch size for fair averaging)
            tot_rec_m += float(recon_loss_wzmask)  * B
            tot_dyn   += float(dyn_mse)            * B
            tot_pred_m += float(pred_mse_wzmask)   * B

        # finalize averages
        rec_masked = tot_rec_m / max(1, nsamples)
        dyn_avg    = tot_dyn   / max(1, nsamples)
        pred_masked= tot_pred_m/ max(1, nsamples)

        # spectral diagnostics
        diag = {}
        if Ad_fix is not None:
            try:
                eigvals = torch.linalg.eigvals(Ad_fix).detach().cpu()
                diag["rho(Ad_phase1)"] = float(eigvals.abs().max())       # discrete spectral radius
            except Exception:
                pass

        return {
            "rec_masked": rec_masked,
            "dyn_mse": dyn_avg,
            "pred_masked": pred_masked if use_pred_loss else None,
            "use_global_A": (Ad_fix is not None),
            "diag": diag
        }


    @torch.no_grad()
    def _phase1_ckpt_payload(self, epoch: int | None, metrics: dict | None = None):
        """
        Pack a portable Phase-I checkpoint.
        """
        def sd_cpu(module: nn.Module):
            return {k: v.detach().cpu() for k, v in module.state_dict().items()}

        payload = {
            "epoch": epoch,
            "run_id": getattr(self, "run_id", None),
            "model_cfg": getattr(self, "model_cfg", None),
            "encoder_state": sd_cpu(self.encoder),
            "decoder_state": sd_cpu(self.decoder),
            # Stash Phase-I linear stats if available
            "Ad_phase1": (self.Ad_phase1.detach().cpu()
                    if hasattr(self, "Ad_phase1") and self.Ad_phase1 is not None else None),
            "b_phase1": (self.b_phase1.detach().cpu()
                    if hasattr(self, "b_phase1") and self.b_phase1 is not None else None),
            "metrics": metrics,
        }
        return payload


    def save_phase1_checkpoint(self, epoch: int | None, metrics: dict | None, pth_name: str):
        out_dir = os.path.join(self.cfg.out_dir, f"{self.run_id}")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, pth_name)
        torch.save(self._phase1_ckpt_payload(epoch, metrics), path)
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.info(f"[Phase-I] checkpoint saved: {path}")


    def load_phase1_ckpt(
        self,
        path: str,
        *,
        restore_modules: bool = True,
        init_linear: bool = True,
        eps_eye: float = 1e-9,
        strict: bool = True,
    ):
        """
        Load a Phase-I checkpoint and optionally:
        1) restore encoder/decoder weights,
        2) initialize the latent linear dynamics from Ad (=Ad_phase1),
        3) compute the fixed-point center z* from (I - Ad) z* = b_phase1 for de-biasing.

        Returns
        -------
        info : dict
            {
            "Ad": torch.Tensor | None,
            "b":  torch.Tensor | None,
            "z_star": torch.Tensor | None,
            "dt": float | None,
            "rho_Ad": float
            }
        """
        ckpt = torch.load(path, map_location="cpu")
        if restore_modules:
            enc_state = ckpt.get("encoder_state", None)
            dec_state = ckpt.get("decoder_state", None)
            if enc_state is not None:
                self.encoder.load_state_dict(enc_state, strict=strict)
            if dec_state is not None:
                self.decoder.load_state_dict(dec_state, strict=strict)

        Ad = None
        b = None
        z_star = None
        rho = float("nan")
        Ad_cpu = ckpt.get("Ad_phase1", None)
        if Ad_cpu is None:
            raise KeyError("[load_phase1_ckpt] 'Ad_phase1' is missing in the checkpoint.")
        Ad = Ad_cpu.to(self.device, dtype=torch.float32)
        b_cpu = ckpt.get("b_phase1", None)
        if b_cpu is not None:
            b = b_cpu.to(self.device, dtype=torch.float32).view(-1)
        else:
            b = None
        # Always cache Phase-I operator for diagnostics / ROM analysis
        self.Ad_phase1 = Ad.detach().clone()
        self.b_phase1 = b.detach().clone() if b is not None else None

        # ---- Optionally initialize current model's latent linear dynamics from Ad. ----
        if init_linear:
            if Ad.size(0) != self.latent_dim:
                raise ValueError(
                    f"[load_phase1_ckpt] Ad dim {Ad.size(0)} != latent_dim {self.latent_dim}"
                )
            with torch.no_grad():
                self.latent_process.A.copy_(Ad)

        # ---- Optionally compute bias center. ----
        z_star = None
        if b is not None and b.numel() == Ad.size(0):
            D = Ad.size(0)
            I = torch.eye(D, device=self.device, dtype=Ad.dtype)
            M = I - Ad
            M_reg = M + eps_eye * I
            try:
                z_star = torch.linalg.solve(M_reg, b)
            except RuntimeError:
                z_star = torch.linalg.pinv(M_reg) @ b

            if z_star.numel() == self.latent_dim:
                self.latent_center = z_star.detach().clone()
        else:
            z_star = None

        # Spectral radius of Ad
        try:
            rho = torch.max(torch.abs(torch.linalg.eigvals(Ad))).item()
        except Exception:
            rho = float("nan")

        if hasattr(self, "logger") and self.logger is not None:
            self.logger.info(
                f"[load_phase1_ckpt] from {path} | "
                f"restore_modules={restore_modules} | init_linear={init_linear} | "
                f"bias={'yes' if b is not None else 'no'} | "
                f"center={'yes' if z_star is not None else 'no'} | "
                f"rho(Ad)={rho:.4f}"
            )
        return {"Ad": Ad, "b": b, "z_star": z_star, "rho_Ad": rho}
    

    def _center_latent(self, z: torch.Tensor) -> torch.Tensor:
        zc = getattr(self, "latent_center", None)
        if zc is None:
            return z
        assert z.shape[-1] == zc.numel(), f"center dim {zc.numel()} != z last dim {z.shape[-1]}"
        if z.ndim == 1:
            return z - zc.to(z)
        view_shape = [1] * (z.ndim - 1) + [-1]
        return z - zc.to(z).view(*view_shape)


    def _decenter_latent(self, z: torch.Tensor) -> torch.Tensor:
        zc = getattr(self, "latent_center", None)
        if zc is None:
            return z
        assert z.shape[-1] == zc.numel(), f"center dim {zc.numel()} != z last dim {z.shape[-1]}"
        if z.ndim == 1:
            return z + zc.to(z)
        view_shape = [1] * (z.ndim - 1) + [-1]
        return z + zc.to(z).view(*view_shape)
    

    @torch.no_grad()
    def set_whitening_scale(self, s: torch.Tensor | None):
        """Register per-dim whitening scales."""
        if s is None:
            self.whiten_scale = None
            return
        s = s.view(-1).to(device=self.device, dtype=torch.float32)
        assert s.numel() == self.latent_dim
        self.whiten_scale = torch.clamp(s, min=1e-3)


    def _whiten_latent(self, centered: torch.Tensor) -> torch.Tensor:
        """w = S^{-1}(z - z*); shape-preserving."""
        s = getattr(self, "whiten_scale", None)
        if (not self.use_diag_whiten) or (s is None):
            return centered
        view = [1]*(centered.ndim - 1) + [-1]
        return centered / s.view(*view)


    def _unwhiten_latent(self, w: torch.Tensor) -> torch.Tensor:
        """(z - z*) = S w; shape-preserving."""
        s = getattr(self, "whiten_scale", None)
        if (not self.use_diag_whiten) or (s is None):
            return w
        view = [1]*(w.ndim - 1) + [-1]
        return w * s.view(*view)


    @torch.no_grad()
    def fit_diag_whitening_from_phase1(self, group: str = "train_eval", max_batches: int | None = 16):
        """Estimate per-dim RMS of centered latents from Phase-I encoder outputs."""
        # choose dataloader
        self._ensure_loader(group)
        loader = {"train": self.train_loader, "train_eval": self.train_eval_loader, "test": self.test_loader}[group]
        if loader is None:
            raise RuntimeError(f"[fit_diag_whitening_from_phase1] loader for {group} is None")
        self.encoder.eval()
        m2 = torch.zeros(self.latent_dim, device=self.device, dtype=torch.float64)
        n = 0
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            lat, _, _ = self._encode_and_recon(batch)                # [T',B,D]
            y = self._center_latent(lat).reshape(-1, self.latent_dim).to(torch.float64)
            m2 += (y*y).sum(dim=0)
            n += y.shape[0]
        if n == 0:
            raise RuntimeError("[fit_diag_whitening_from_phase1] no samples")
        mean_sq = (m2 / n).to(torch.float32)
        s = torch.sqrt(torch.clamp(mean_sq, min=1e-8)).clamp_min(1e-3)
        self.set_whitening_scale(s)
        return s


    @torch.no_grad()
    def _reinit_linear_in_whiten_space(self):
        """Apply Aw = S^{-1} Ad S and re-init latent_process linear skeleton."""
        if (not self.use_diag_whiten) or (self.whiten_scale is None):
            return
        if not hasattr(self, "Ad_phase1") or self.Ad_phase1 is None:
            return
        Ad = self.Ad_phase1.to(self.device, dtype=torch.float32)  # [D,D]
        s  = self.whiten_scale.to(self.device, dtype=Ad.dtype)    # [D]
        # Aw = S^{-1} Ad S  (column scale by s, then row divide by s)
        Aw = (Ad * s.view(1, -1)) / s.view(-1, 1)
        self.latent_process.A.copy_(Aw)
        self.Ad_phase1_whiten = Aw.detach().clone()

    
    """
    ROM helpers
    """
    def _get_U(self) -> torch.Tensor:
        """Return orthonormal U [D,d] on correct device/dtype."""
        if self.U_proj is None:
            raise RuntimeError("U_proj is None. Train or load projector first.")
        p0 = next(self.latent_process.parameters(), None)
        dtype = p0.dtype if p0 is not None else torch.float32
        return self.U_proj.to(device=self.device, dtype=dtype)


    def _project_latent(self, w: torch.Tensor) -> torch.Tensor:
        """w [..., D] -> y [..., d] via U."""
        U = self._get_U()
        return torch.matmul(w, U)  # broadcast matmul: (...,D) x (D,d) -> (...,d)


    def _lift_latent(self, y: torch.Tensor) -> torch.Tensor:
        """y [..., d] -> w [..., D] via U^T."""
        U = self._get_U()
        return torch.matmul(y, U.t())


    @torch.no_grad()
    def set_projector(self, U: torch.Tensor):
        """Register an orthonormal projector U [D,d]."""
        U = U.to(device=self.device, dtype=torch.float32)
        Q, _ = torch.linalg.qr(U)
        d = U.size(1)
        self.U_proj = Q[:, :d].contiguous()
        self.use_projector = True
        self.latent_dim_y = d


    def save_projector(self, path: str):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"U_proj": (self.U_proj.detach().cpu() if self.U_proj is not None else None)}, path)


    def load_projector(self, path: str):
        payload = torch.load(path, map_location="cpu")
        U = payload.get("U_proj", None)
        if U is None:
            raise KeyError(f"'U_proj' missing in {path}")
        self.set_projector(U)


    def train_projector_from_phase1_ckpt(self, phase1_path: str, d: int,
                                         epochs: int = 3, lr: float = 1e-3, lr_dec: float = 0.0,
                                         lambda_dyn: float = 0.1, lambda_ortho: float = 1e-2,
                                         log_every: int | None = None, eval_every: int | None = None,
                                         stiefel: bool = True
                                         ):
        self.setup_logger()
        self.logger.info(f"Training low dimensional projectors based on pretrained encoder-decoder models!")
        self.save_repro_artifacts()
        print(f"stiefel = {stiefel}")

        # load model from phase1
        # load encoder-decoder, and linear parameter
        phase1_info = self.load_phase1_ckpt(path=phase1_path)
        assert self.Ad_phase1 is not None
        Ad = self.Ad_phase1
        train_dec = (self.cfg.lr_dec != 0)
        self.logger.info(f"finetune decoder = {train_dec}")
        set_requires_grad(self.encoder, False)
        set_requires_grad(self.decoder, True) if train_dec else set_requires_grad(self.decoder, False)    ##################

        # ---- estimate whitening scales (Phase-I encoder) & re-init Aw ----
        if self.use_diag_whiten:
            self.fit_diag_whitening_from_phase1(group="train", max_batches=16)
            self._reinit_linear_in_whiten_space()
            self.logger.info(f"whiten vector: {self.whiten_scale}")
        if self.whiten_scale is not None:
            s = self.whiten_scale.to(self.device)
            Aw = (Ad.to(self.device, torch.float32) * s.view(1,-1)) / s.view(-1,1)
            # bw = None
            # if getattr(self, "b_phase1", None) is not None:
            #     bw = (self.b_phase1.to(self.device, torch.float32) / (s + 1e-12))
        else:
            Aw = Ad.to(self.device, torch.float32)
            # bw = (self.b_phase1.to(self.device, torch.float32) if getattr(self, "b_phase1", None) is not None else None)
        
        assert self.data_processor.mode == "interpolation", f"Mismatched dataloaders"
        self.build_dataloader(group="train")
        self._save_split_and_samples()

        ######## Build Optimizer ########
        D = self.latent_dim
        assert d <= D, f"d={d} must be <= latent_dim={D}"
        U_param = torch.nn.Parameter(torch.randn(D, d, device=self.device) / (D**0.5))
        optim_proj = torch.optim.Adam([U_param], lr=lr)
        optim_dec = torch.optim.Adam([{'params': self.decoder.parameters(), 'lr': lr_dec}]) if lr_dec != 0 else None

        loss_tr_min, loss_ts_min = float('inf'), float('inf')
        loss_dict = {
            "recon": [], "dyn": [], "total": []
        }
        for epoch in range(1, epochs + 1):
            for i, batch in enumerate(self.train_loader):
                latent_states, _, _ = self._encode_and_recon(batch)     # [t, B, latent_dim]
                latent_states_ = self._center_latent(latent_states)
                latent_states_ = self._whiten_latent(latent_states_)

                Q, _ = torch.linalg.qr(U_param)                         # [D,D]
                U = Q[:, :d] if stiefel else U_param                    # [D,d]
                eff_states_ = torch.matmul(latent_states_, U)           # [T',B,d]
                latent_hat_ = torch.matmul(eff_states_, U.t())          # [T',B,D]
                latent_hat = self._decenter_latent(self._unwhiten_latent(latent_hat_))
                x_hat = self._decode_latent(latent_hat)                 # [B,T',H,W,C]
                x_gt = batch["data"][:, self.n_frames_cond-1:, ...].to(self.device)
                m_gt = batch["mask"][:, self.n_frames_cond-1:, ...].to(self.device)
                _, rec_loss = masked_mse_loss(x_hat, x_gt, m_gt)        # masked MSE

                A_eff = U.t() @ Aw @ U                                  # [d,d]
                eff_next_ = torch.matmul(eff_states_[:-1], A_eff.t())   # [T'-1,B,d]
                dyn_loss = torch.nn.functional.mse_loss(eff_states_[1:], eff_next_)
                ortho_loss = torch.norm(U.t() @ U - torch.eye(d, device=U.device))**2

                loss = rec_loss + lambda_dyn * dyn_loss + lambda_ortho * ortho_loss
                rec_val = float(rec_loss.detach().cpu())
                dyn_val = float(dyn_loss.detach().cpu())
                loss_val = float(loss.detach().cpu())
                loss_dict["recon"].append(rec_val)
                loss_dict["dyn"].append(dyn_val)
                loss_dict["total"].append(loss_val)
                
                optim_proj.zero_grad()
                optim_dec.zero_grad() if train_dec else None
                loss.backward()
                optim_proj.step()
                optim_dec.step() if train_dec else None

                if log_every is not None and (epoch * len(self.train_loader) + i) % log_every == 0:
                    self.logger.info(f"[Projector] epoch {epoch:03d}/{epochs} | rec(mask)={float(rec_loss):.6f} | dyn={float(dyn_loss):.6f} | ortho={float(ortho_loss):.6f}")
        
        with torch.no_grad():
            Q, _ = torch.linalg.qr(U_param.data)
            self.U_proj = Q[:, :d].contiguous() if stiefel else U_param
            self.use_projector = True
            self.latent_dim_y = d
        A_eff = (self.U_proj.t().to(Aw) @ Aw @ self.U_proj.to(Aw)).detach().cpu()
        out_dir = os.path.join(self.cfg.out_dir, f"{self.run_id}")
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"U_proj": self.U_proj.detach().cpu(),
                    "A_eff": A_eff}, os.path.join(out_dir, f"U_proj_d{d}.pt"))
        with open(os.path.join(out_dir, "loss.json"), "w", encoding="utf-8") as f:
            json.dump(loss_dict, f, ensure_ascii=False, indent=2)
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.info(f"[Projector] saved to: {os.path.join(out_dir, f'U_proj_d{d}.pt')}")

        ##### rebuild latent process #####
        Ay = self.U_proj.t() @ Aw @ self.U_proj
        from copy import deepcopy
        lp_cfg = deepcopy(self.model_cfg["latent_process_discrete"])
        assert d % self.state_dim == 0
        lp_cfg["code_dim"] = d // self.state_dim
        self.model_cfg["latent_process_discrete"] = lp_cfg
        self.latent_process = build_latent(model_cfg=lp_cfg).to(self.device)
        with torch.no_grad():
            self.latent_process.A.copy_(Ay)
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.info(f"[Projector] latent_process rebuilt at dim d={d}. Now ready for Phase-2.")


    def _phase2_param_groups(self, lr_mem=1e-3, lr_lin=0.0, wd_mem=0.0, wd_lin=1e-4,):
        lp = self.latent_process
        mem_params, lin_params = [], []

        # ---- linear skeleton ----
        lin_params += [lp.A]
        # ---- memory correction----
        for n in ["memory_encoder", "memory_decoder"]:
            if hasattr(lp, n):
                mem_params += list(getattr(lp, n).parameters())
        if hasattr(lp, "memory"):
            mem_params += list(lp.memory.parameters())
        if hasattr(lp, "_raw_gate"):
            mem_params += [lp._raw_gate]
        if hasattr(lp, "res_lstm"):
            mem_params += list(lp.res_lstm.parameters())

        groups = []
        if mem_params:
            groups.append({"name": "memory", "params": mem_params, "lr": lr_mem, "weight_decay": wd_mem})
        if lin_params:
            groups.append({"name": "linear", "params": lin_params, "lr": lr_lin, "weight_decay": wd_lin})
        return groups


    def init_optim_phase2(self, lr_mem=None, lr_lin=None, lr_dec=None):
        if lr_mem is None:
            lr_mem = getattr(self.cfg, "lr_dyn_mem", self.lr)
        if lr_lin is None:
            lr_lin = getattr(self.cfg, "lr_dyn_lin", 0.0)
        if lr_dec is None:
            lr_dec = getattr(self.cfg, "lr_dec", 0.0)

        groups = self._phase2_param_groups(
            lr_mem=lr_mem, lr_lin=lr_lin,
            wd_mem=0.0, wd_lin=0.0,
        )
        self.optim_dyn = torch.optim.Adam(groups) if groups else None
        self.optim_dec = torch.optim.Adam([{'params': self.decoder.parameters(), 'lr': lr_dec}]) if lr_dec != 0 else None

        # ------- scheduler -------
        self.scheduler_dyn = None
        self.scheduler_enc, self.scheduler_dec = None, None
        if self.optim_dyn is not None:
            if self.cfg.scheduler == 'CosineAnnealingLR':
                self.scheduler_dyn = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_dyn, T_max=self.cfg.epochs)
            elif self.cfg.scheduler == 'StepLR':
                self.scheduler_dyn = torch.optim.lr_scheduler.StepLR(
                    self.optim_dyn, step_size=self.cfg.step_size, gamma=self.cfg.gamma
                )
        if lr_dec != 0:
            if self.cfg.scheduler == 'CosineAnnealingLR':
                self.scheduler_dec = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_dec, T_max=self.cfg.epochs)
            elif self.cfg.scheduler == 'StepLR':
                self.scheduler_dec = torch.optim.lr_scheduler.StepLR(self.optim_dec, step_size=self.cfg.step_size, gamma=self.cfg.gamma)


    def train_phase2(self, phase1_path: str, log_every: int | None = None, eval_every: int | None = None, 
                     verbose: str = None):
        self.setup_logger()
        self.save_repro_artifacts()
        self.log_param_table()
        criterion = nn.MSELoss()

        # load model from phase1
        # load encoder-decoder, and linear parameter
        if not self.use_projector:
            phase1_info = self.load_phase1_ckpt(path=phase1_path)
        train_dec = (self.cfg.lr_dec != 0)
        self.logger.info(f"finetune decoder = {train_dec}")
        set_requires_grad(self.encoder, False)
        set_requires_grad(self.decoder, True) if train_dec else set_requires_grad(self.decoder, False)
        set_requires_grad(self.latent_process, True)
        
        # ---- estimate whitening scales (Phase-I encoder) & re-init Aw ----
        if self.use_diag_whiten and not self.use_projector:
            self.fit_diag_whitening_from_phase1(group="train", max_batches=16)
            self._reinit_linear_in_whiten_space()
        self.logger.info(f"whiten vector: {self.whiten_scale}")
        
        assert self.data_processor.mode == "interpolation", f"Mismatched dataloaders"
        self.build_dataloader(group="train")
        self.init_optim_phase2()
        if eval_every is not None:
            self.build_dataloader(group="test")
            self.build_dataloader(group="train_eval")
        self._save_split_and_samples()
        if verbose is not None:
            self.logger.info(f"{verbose}")

        ######## Initial Evaluation (Linear Backbone) ########
        if self.train_eval_loader is not None:
            self.logger.info("--------Begin Evaluation on Train--------")
            train_eval_errs = self.evaluate(dataloader=self.train_eval_loader)
            self.logger.info("Evaluation on train:\n%s", pformat(train_eval_errs, width=100, compact=False))
        # out-of-domain evaluation
        if self.test_loader is not None:
            self.logger.info("--------Begin Evaluation on Test--------")
            test_errs = self.evaluate(dataloader=self.test_loader)
            self.logger.info("Evaluation on test:\n%s", pformat(test_errs, width=100, compact=False))

        tf_epsilon = self.tf_epsilon
        loss_tr_min, loss_ts_min = float('inf'), float('inf')
        for epoch in range(1, self.cfg.epochs + 1):
            self.latent_process.train()
            for i, batch in enumerate(self.train_loader):
                ground_truth = batch["data"].to(self.device)    # [B, T, H, W, n_ch]
                masks = batch["mask"].to(self.device) if self.is_2d else None  # [B, T, H, W, n_ch], none for 1d
                t_eval = batch['t'][0].to(self.device)          # [T]
                bs, train_len = ground_truth.shape[0], ground_truth.shape[1]
                assert train_len == self.n_frames_train

                latent_states = self._encode_latent_from_field(ground_truth, masks)
                latent_states_ = self._center_latent(latent_states)
                latent_states_ = self._whiten_latent(latent_states_)

                if self.use_projector:
                    latent_states_ = self._project_latent(latent_states_)
                    dyn_y, memory_states, aux = self.latent_process(
                        alpha_0=latent_states_[0], t_eval=t_eval[self.n_frames_cond-1:],
                        memory_init=None, teacher_forcing=True,
                        tf_alpha=latent_states_, tf_epsilon=tf_epsilon, tf_mask=None
                    )                                                          # [T',B,d]
                    dyn_loss = criterion(dyn_y, latent_states_.detach())
                    dyn_states_ = self._lift_latent(dyn_y)                     # lift back to [T',B,D]
                else:
                    dyn_states_, memory_states, aux = self.latent_process(
                        alpha_0=latent_states_[0], t_eval=t_eval[self.n_frames_cond-1:],
                        memory_init=None,
                        teacher_forcing=True, tf_alpha=latent_states_, tf_epsilon=tf_epsilon, tf_mask=None
                    )    # [t, B, latent_dim]
                    dyn_loss = criterion(dyn_states_, latent_states_.detach())

                corr = aux["phi_dec_l2"]
                if self.latent_process.memory_type != "residual":
                    T_eff, B, D = latent_states_.shape
                    A = self.latent_process.A
                    res_gt = latent_states_[1:] - (latent_states_[:-1] @ A.T)
                    mem_flatten = memory_states[:-1].reshape(-1, self.latent_process.memory_dim)
                    res_flatten = self.latent_process.memory_decoder(mem_flatten)
                    res_pred = (self.latent_process.gate * res_flatten).view(T_eff-1, B, D)
                    residual_loss = F.mse_loss(res_pred, res_gt)
                else:
                    A = self.latent_process.A
                    res_gt = latent_states_[1:] - (latent_states_[:-1] @ A.T)
                    res_pred = memory_states[:-1]
                    residual_loss = F.mse_loss(res_pred, res_gt)
                dyn_states_ = self._unwhiten_latent(dyn_states_)   
                dyn_states = self._decenter_latent(dyn_states_)

                pred_field = self._decode_latent(latent_seqs=dyn_states)    # [B, t, H, W, C]
                if self.is_2d:
                    pred_loss, pred_loss_wzmask = masked_mse_loss(
                        pred_field, ground_truth[:, self.n_frames_cond-1:, ...], masks[:, self.n_frames_cond-1:, ...]
                    )
                else:
                    pred_loss, pred_loss_wzmask = masked_mse_loss(pred_field, ground_truth[:, self.n_frames_cond-1:, ...])

                loss = dyn_loss + self.lambda_pred * pred_loss_wzmask
                if self.lambda_corr is not None and corr is not None:
                    loss += self.lambda_corr * corr
                loss += self.lambda_resid * residual_loss

                self.optim_dyn.zero_grad()
                if train_dec:
                    self.optim_dec.zero_grad()
                loss.backward()
                if self.cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.latent_process.parameters(), self.cfg.max_grad_norm)    #######
                    if train_dec:
                        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.cfg.max_grad_norm)
                self.optim_dyn.step()
                if train_dec:
                    self.optim_dec.step()

                if log_every is not None and (epoch * len(self.train_loader) + i) % log_every == 0:
                    msg = [
                        f"Epoch {epoch:04d}/{self.cfg.epochs} | iteration {i+1:03d}",
                        f"| pred {pred_loss.item():.8f}",
                        f"| pred(mask) {pred_loss_wzmask.item():.8f}",
                        f"| dyn {dyn_loss.item():.8f}",
                    ]
                    msg += [f"| residual {residual_loss.item():.8f}"]
                    msg += [f"| correction ratio mean {aux['mem_ratio_mean']:.8f}"]
                    msg += [f"| correction ratio p90 {aux['mem_ratio_p90']:.8f}"]
                    msg += [f"| memory_corr {corr.item():.8f}"]
                    msg += [f"| epsilon {tf_epsilon:.4f}"]
                    self.logger.info(" ".join(msg))
                if (epoch * len(self.train_loader) + i + 1) % self.cfg.update_every == 0:
                    tf_epsilon = max(tf_epsilon * self.epsilon, self.cfg.tf_epsilon_min)

                if eval_every is not None and (epoch * len(self.train_loader) + i + 1) % eval_every == 0:
                    if self.train_eval_loader is not None:
                        self.logger.info("--------Begin Evaluation on Train--------")
                        train_eval_errs = self.evaluate(dataloader=self.train_eval_loader)
                        self.logger.info("Evaluation on train:\n%s", pformat(train_eval_errs, width=100, compact=False))
                        losses_tr = train_eval_errs["mse_losses"]
                        if loss_tr_min > losses_tr["loss"]:
                            loss_tr_min = losses_tr["loss"]
                            self.save_model(epoch, train_eval_errs, pth_name="model_tr_best.pth")
                    # out-of-domain evaluation
                    if self.test_loader is not None:
                        self.logger.info("--------Begin Evaluation on Test--------")
                        test_errs = self.evaluate(dataloader=self.test_loader)
                        self.logger.info("Evaluation on test:\n%s", pformat(test_errs, width=100, compact=False))
                        losses_ts = test_errs["mse_losses"]
                        if loss_ts_min > losses_ts["loss"]:
                            loss_ts_min = losses_ts["loss"]
                            self.save_model(epoch, test_errs, pth_name="model_ts_best.pth")
            if self.cfg.scheduler == 'CosineAnnealingLR' or self.cfg.scheduler == 'StepLR':
                self.scheduler_dyn.step()
                if train_dec:
                    self.scheduler_dec.step()
        self.logger.info("Training Finished! Saving model...")
        self.save_model(epoch=None, losses=None, pth_name="final_model.pth")


    @torch.no_grad()
    def evaluate(self, dataloader, model_pth: str | None = None):
        if model_pth is not None:
            _ = self.load_from_ckpt(ckpt_path=model_pth, device=self.device)
        
        err_dict = {}
        rel_err, mse_err = 0.0, 0.0
        rel_err_in_t, rel_err_out_t, mse_err_in_t, mse_err_out_t = 0.0, 0.0, 0.0, 0.0
        rmse_err, rmse_err_in_t, rmse_err_out_t = 0.0, 0.0, 0.0  
        rel_criterion = LpLoss(size_average=False)
        loss, loss_out_t, loss_in_t = 0.0, 0.0, 0.0
        num_samples = int(0)

        for batch in dataloader:
            ground_truth = batch["data"].to(self.device)    # [B, T, H, W, n_ch] (T = n_frames_train + n_frames_out)
            t_eval = batch['t'][0][self.n_frames_cond-1:].to(self.device)
            masks = batch["mask"].to(self.device) if self.is_2d else None  # [B, T, H, W, n_ch], none for 1d
            bs, length = ground_truth.shape[0], ground_truth.shape[1]
            num_samples += bs
            
            with torch.no_grad():
                latent_state = self._encode_cond_batch(batch)
                latent_state_ = self._center_latent(latent_state)
                latent_state_ = self._whiten_latent(latent_state_)
                if self.use_projector:
                    latent_state_ = self._project_latent(latent_state_)
                dyn_states_, _, _ = self.latent_process(alpha_0=latent_state_, t_eval=t_eval, teacher_forcing=False)    # [T, B, latent_dim]
                if self.use_projector:
                    dyn_states_ = self._lift_latent(dyn_states_)
                dyn_states_ = self._unwhiten_latent(dyn_states_)
                dyn_states = self._decenter_latent(dyn_states_)
                recon_seq = self._decode_latent(dyn_states)

                # compute losses
                # recon_seq, ground_truth: [B, T, H, W, s]
                n_cond = self.n_frames_cond - 1
                ground_truth_ = ground_truth[:, n_cond:, ...]
                masks_ = masks[:, n_cond:, ...] if self.is_2d else None
                pred_in_t_, pred_out_t_ = recon_seq[:, :self.n_frames_train-n_cond, ...], recon_seq[:, self.n_frames_train-n_cond:, ...]
                gt_in_t_, gt_out_t_ = ground_truth_[:, :self.n_frames_train-n_cond, ...], ground_truth_[:, self.n_frames_train-n_cond:, ...]

                rel_err += rel_criterion(recon_seq.reshape(bs, -1), ground_truth_.reshape(bs, -1)).item()
                rel_err_in_t += rel_criterion(pred_in_t_.reshape(bs, -1), gt_in_t_.reshape(bs, -1)).item()
                rel_err_out_t += rel_criterion(pred_out_t_.reshape(bs, -1), gt_out_t_.reshape(bs, -1)).item()
                mse = masked_mse_loss(recon_seq, ground_truth_)[0]
                mse_in_t = masked_mse_loss(pred_in_t_, gt_in_t_)[0]
                mse_out_t = masked_mse_loss(pred_out_t_, gt_out_t_)[0]
                mse_err += mse * bs
                mse_err_in_t += mse_in_t * bs
                mse_err_out_t += mse_out_t * bs
                rmse_err += torch.sqrt(mse) * bs
                rmse_err_in_t += torch.sqrt(mse_in_t) * bs
                rmse_err_out_t += torch.sqrt(mse_out_t) * bs

        rel_err = rel_err / num_samples
        mse_err = mse_err / num_samples
        rmse_err = rmse_err / num_samples
        rel_err_in_t, rel_err_out_t = rel_err_in_t / num_samples, rel_err_out_t / num_samples
        mse_err_in_t, mse_err_out_t = mse_err_in_t / num_samples, mse_err_out_t / num_samples
        rmse_err_in_t, rmse_err_out_t = rmse_err_in_t / num_samples, rmse_err_out_t / num_samples

        rel_losses = {
            "loss": rel_err, "loss_in_t": rel_err_in_t, "loss_out_t": rel_err_out_t
        }
        mse_losses = {
            "loss": mse_err, "loss_in_t": mse_err_in_t, "loss_out_t": mse_err_out_t
        }
        rmse_losses = {
            "loss": rmse_err, "loss_in_t": rmse_err_in_t, "loss_out_t": rmse_err_out_t
        }
        err_dict.update({"rel_losses": rel_losses})
        err_dict.update({"mse_losses": mse_losses})
        err_dict.update({"rmse_losses": rmse_losses})
        return err_dict


    def save_model(self, epoch: int | None = None, losses: dict | None = None, pth_name: str = "model_tr.pth"):
        save_dict = {
            "args": vars(self.args) if self.args is not None else None,
            "epoch": epoch,
            "encoder": self.encoder.state_dict(),
            "latent_process": self.latent_process.state_dict(),
            "decoder": self.decoder.state_dict(),
            "losses": dict(losses) if isinstance(losses, dict) else losses
        }
        lc = getattr(self, "latent_center", None)
        if lc is not None:
            save_dict["latent_center"] = lc.detach().cpu().view(-1)
        ws = getattr(self, "whiten_scale", None)
        if ws is not None:
            save_dict["whiten_scale"] = ws.detach().cpu().view(-1)
        U = getattr(self, "U_proj", None)
        if U is not None:
            save_dict["U_proj"] = U.detach().cpu()
        out_dir = os.path.join(self.cfg.out_dir, f"{self.run_id}")
        torch.save(save_dict, os.path.join(out_dir, f'{pth_name}'))


    def load_from_ckpt(self, ckpt_path: str, device: str | None = None):
        target_device = device if device is not None else self.device
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")

        self.encoder.load_state_dict(ckpt["encoder"])
        self.latent_process.load_state_dict(ckpt["latent_process"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.encoder.to(target_device)
        self.latent_process.to(target_device)
        self.decoder.to(target_device)

        lc = ckpt.get("latent_center", None)
        if lc is not None:
            p0 = next(self.latent_process.parameters(), None)
            dtype = p0.dtype if p0 is not None else torch.float32
            lc = lc.detach().view(-1).cpu()
            if lc.numel() == self.latent_dim:
                self.latent_center = lc.to(device=target_device, dtype=dtype)
            else:
                self.latent_center = None
                if hasattr(self, "logger") and self.logger is not None:
                    self.logger.warning(
                        f"[load_from_ckpt] latent_center dim {lc.numel()} != latent_dim {self.latent_dim}; dropped."
                    )
        else:
            self.latent_center = None

        ws = ckpt.get("whiten_scale", None)
        if ws is not None:
            p0 = next(self.latent_process.parameters(), None)
            dtype = p0.dtype if p0 is not None else torch.float32
            ws = ws.detach().view(-1).cpu()
            valid = (
                ws.numel() == self.latent_dim
                and torch.isfinite(ws).all().item()
                and (ws > 0).all().item()
            )
            if valid:
                self.set_whitening_scale(ws.to(device=target_device, dtype=dtype))
            else:
                self.set_whitening_scale(None)
                if hasattr(self, "logger") and self.logger is not None:
                    self.logger.warning("whiten_scale invalid; ignored.")
        else:
            self.set_whitening_scale(None)

        U = ckpt.get("U_proj", None)
        if U is not None:
            try:
                self.set_projector(U.detach().cpu())
            except Exception as e:
                if hasattr(self, "logger") and self.logger is not None:
                    self.logger.warning(f"U_proj load failed; ignoring. err={e}")

        return {
            "epoch": ckpt.get("epoch", -1),
            "losses": ckpt.get("losses", None),
            "args": ckpt.get("args", None),
        }


    def _ensure_loader(self, group: str) -> None:
        loader_utils.ensure_loader(self, group)
    
    def sample_batch(self, group: str, batch_size: int, replace: bool = False):
        return loader_utils.sample_batch(self, group, batch_size, replace=replace)

    def sample_from_fix(self, traj_id: int, t0: int, rollout_steps: int | None = None):
        return loader_utils.sample_from_fix(self, traj_id, t0, rollout_steps=rollout_steps)
    
    def make_subset_from_saved(self, group: str, saved_json_path: str, batch_size: int | None = None):
        return loader_utils.make_subset_from_saved(self, group, saved_json_path, batch_size=batch_size)


    def linear_rollout_one_batch_with_Ab(self, batch_samples, rollout_steps: int, return_gt: bool = False):
        self.switch_to_eval()
        device = self.device
        assert hasattr(self, "Ad_phase1") and hasattr (self, "b_phase1"), f"load phase I checkpoint first"
        A, b = self.Ad_phase1, self.b_phase1
        with torch.no_grad():
            n_cond = self.n_frames_cond - 1
            ground_truth = batch_samples["data"].to(self.device)    # [B, T, H, W, n_ch] (T = n_frames_train + n_frames_out)
            masks = batch_samples["mask"].to(self.device)           # [B, T, H, W, n_ch]
            assert rollout_steps + self.data_processor.n_frames_cond <= ground_truth.shape[1]
            bs, length, H, W, _ = ground_truth.shape

            latent_state = self._encode_cond_batch(batch_samples)
            dyn_states = torch.empty((rollout_steps+1, bs, self.latent_dim), device=device, dtype=latent_state.dtype)
            dyn_states[0] = latent_state
            if b is not None:
                b = b.view(1, self.latent_dim).to(device=device, dtype=latent_state.dtype)
            for t in range(rollout_steps):
                dyn_states[t + 1] = dyn_states[t] @ A.T + (b if b is not None else 0.0)
            recon_seq = self._decode_latent(dyn_states)
            recon_seq_ = recon_seq.permute(0, 2, 3, 1, 4)

            # recon_seq, ground_truth: [B, T, H, W, s]
            ground_truth_ = ground_truth[:, n_cond:n_cond+rollout_steps+1, ...].permute(0, 2, 3, 1, 4)
            masks_ = masks[:, n_cond:n_cond+rollout_steps+1, ...].permute(0, 2, 3, 1, 4)

            pred_err, pred_err_wzmask = masked_mse_loss(
                recon_seq,
                ground_truth[:, n_cond:n_cond+rollout_steps+1, ...],
                masks[:, n_cond:n_cond+rollout_steps+1, ...],
            )
            print(f"pred loss = {pred_err}, pred loss (mask) = {pred_err_wzmask}")
        if return_gt:
            return recon_seq_, ground_truth_, ground_truth.permute(0, 2, 3, 1, 4)
        return recon_seq_, ground_truth_    # [B, ..., rollout_steps+1, C]


    def rollout_one_batch(self, batch_samples, rollout_steps: int, return_gt: bool = False):
        # batch_samples, containing data_tensor: [B, T, C, H, W], using first n_frames_cond to generate initial state,
        # record prediciton at n_cond_frames, n_cond_frames + 1, ..., n_cond_frames + rollout_steps (< T)
        self.switch_to_eval()
        with torch.no_grad():
            n_cond = self.n_frames_cond - 1
            ground_truth = batch_samples["data"].to(self.device)    # [B, T, H, W, n_ch] (T = n_frames_train + n_frames_out)
            t_eval = batch_samples['t'][0][n_cond:].to(self.device)
            sample_idx = batch_samples["index"].to(self.device)     # [B,]
            masks = batch_samples["mask"].to(self.device) if self.is_2d else None           # [B, T, H, W, n_ch]
            assert rollout_steps + self.data_processor.n_frames_cond <= ground_truth.shape[1]
            bs = ground_truth.shape[0]

            latent_state = self._encode_cond_batch(batch_samples)
            latent_state_ = self._center_latent(latent_state)
            latent_state_ = self._whiten_latent(latent_state_)
            if self.use_projector:
                latent_state_ = self._project_latent(latent_state_)

            # print(latent_state.shape, latent_state_.shape)
            dyn_states_, _, _ = self.latent_process(alpha_0=latent_state_, t_eval=t_eval[:rollout_steps+1], teacher_forcing=False)    # [T, B, latent_dim]
            if self.use_projector:
                dyn_states_ = self._lift_latent(dyn_states_)
            dyn_states_ = self._unwhiten_latent(dyn_states_)
            dyn_states = self._decenter_latent(dyn_states_)
            recon_seq = self._decode_latent(dyn_states)
            recon_seq_ = recon_seq.permute(0, 2, 3, 1, 4) if self.is_2d else recon_seq.permute(0, 2, 1, 3)

            # compute losses
            # recon_seq, ground_truth: [B, T, H, W, s]
            if self.is_2d:
                ground_truth_ = ground_truth[:, n_cond:n_cond+rollout_steps+1, ...].permute(0, 2, 3, 1, 4)
                masks_ = masks[:, n_cond:n_cond+rollout_steps+1, ...].permute(0, 2, 3, 1, 4)
            else:
                ground_truth_ = ground_truth[:, n_cond:n_cond+rollout_steps+1, ...].permute(0, 2, 1, 3)

            pred_err = masked_mse_loss(recon_seq, ground_truth[:, n_cond:n_cond+rollout_steps+1, ...])[0]
            if self.is_2d:
                pred_err_wzmask = masked_mse_loss(
                    recon_seq, ground_truth[:, n_cond:n_cond+rollout_steps+1, ...], masks[:, n_cond:n_cond+rollout_steps+1, ...]
                )[1]
            else:
                pred_err_wzmask = masked_mse_loss(
                    recon_seq, ground_truth[:, n_cond:n_cond+rollout_steps+1, ...]
                )[1]
            print(f"pred loss = {pred_err}, pred loss (mask) = {pred_err_wzmask}")

        if return_gt:
            if self.is_2d:
                return recon_seq_, ground_truth_, ground_truth.permute(0, 2, 3, 1, 4)
            else:
                return recon_seq_, ground_truth_, ground_truth.permute(0, 2, 1, 3)
        return recon_seq_, ground_truth_    # [B, ..., rollout_steps+1, C]


    def save_rollout_tensors(
        self,
        out_dir: str,
        traj_id: int, t0: int, rollout_steps: int | None = None,
        phase: str = "phase1",
    ):
        os.makedirs(out_dir, exist_ok=True)
        self.switch_to_eval()
        loader = self.sample_from_fix(traj_id, t0, rollout_steps)
        for batch in loader:
            masks = batch["mask"].permute(0, 2, 3, 1, 4)    # [1, ..., T, C]
            if phase == "phase1":
                recon_seq_, ground_truth_, ground_truth = self.linear_rollout_one_batch_with_Ab(
                    batch, rollout_steps, return_gt=True
                )
            else:
                recon_seq_, ground_truth_, ground_truth = self.rollout_one_batch(
                    batch, rollout_steps, return_gt=True
                )
            print(f"recon_seq_: {recon_seq_.shape}, ground_truth_: {ground_truth_.shape}, ground_truth: {ground_truth.shape}")
            payload = {
                "traj_id": int(traj_id),
                "t0": int(t0),
                "pred": recon_seq_.squeeze(0).detach().cpu(),                      # [H, W, rollout_steps+1, C]
                "gt": ground_truth_.squeeze(0).detach().cpu(),                     # [H, W, rollout_steps+1, C]
                "gt_full": ground_truth.squeeze(0).detach().cpu(),                 # [H, W, T', C]
                "mask_full": masks.squeeze(0).detach().cpu(),                      # [H, W, T', C]
            }
            torch.save(payload, os.path.join(out_dir, f"traj_id{traj_id}_t0_{t0}.pt"))
    

    @torch.no_grad()
    def save_all_latents(self, group: str, out_dir: str,
                        mode: str = "aggregate",   # "per_sample" or "aggregate"
                        fname: str = "latents_all.pt"):
        """
        Save alpha_t for all samples in the selected split to disk.
        - per_sample: one .pt file per sample with {'alpha': [T',D], 't': [T'], 'index': (traj_id,t0)}
        - aggregate: one large .pt file containing the collected payload list
        """
        assert mode in {"per_sample", "aggregate"}
        self._ensure_loader(group)
        loader = {"train": self.train_loader,
                "train_eval": self.train_eval_loader,
                "test": self.test_loader}[group]
        dataset = loader.dataset
        os.makedirs(out_dir, exist_ok=True)

        # Keep the sampling order aligned with the dataset order.
        from torch.utils.data import DataLoader
        tmp_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        agg = []  # Collect payloads when using aggregate mode.
        for k, batch in enumerate(tmp_loader):
            lat, _, _ = self._encode_and_recon(batch)   # [T',1,D]
            lat = lat[:, 0].detach().cpu()              # [T',D]
            # Save the aligned time axis from batch['t'][0], cropped to [nf_cond-1:].
            t_full = batch['t'][0].detach().cpu()       # [T]
            n_cond = self.n_frames_cond - 1
            t_eff = t_full[n_cond:]                     # [T']
            # Record index metadata when the dataset exposes samples.
            idx_meta = None
            if hasattr(dataset, "samples"):
                try:
                    idx_meta = dataset.samples[k]
                except Exception:
                    idx_meta = (int(k), 0)

            payload = {"alpha": lat, "t": t_eff, "index": idx_meta, "group": group}
            if mode == "per_sample":
                torch.save(payload, os.path.join(out_dir, f"latent_{k:06d}.pt"))
            else:
                agg.append(payload)

        if mode == "aggregate":
            torch.save(agg, os.path.join(out_dir, fname))
        print(f"[save_all_latents] done. group={group}, N={len(dataset)}, mode={mode}, out={out_dir}")


    @staticmethod
    @torch.no_grad()
    def _spectral_radius(A: torch.Tensor) -> float:
        """Return spectral radius max|lambda_i(A)| for a real square matrix."""
        eig = torch.linalg.eigvals(A)  # complex
        return float(eig.abs().max().item())
