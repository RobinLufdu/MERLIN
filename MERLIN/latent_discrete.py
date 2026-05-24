from __future__ import annotations
import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn

from MERLIN.network import MLP


# ------------------------------ Helpers ------------------------------
def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)

def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean() if x.numel() > 0 else torch.tensor(0.0, device=x.device, dtype=x.dtype)

def _percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    # q in [0,100]
    k = int(math.ceil(q / 100.0 * max(1, x.numel()))) - 1
    if x.numel() == 0: return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    k = max(0, min(k, x.numel()-1))
    vals, _ = torch.sort(x.view(-1))
    return vals[k]


# ------------------------------ Memory blocks ------------------------------
class LeakyMemory(nn.Module):
    """
    m_{t+1} = gamma ⊙ m_t + psi_enc(alpha_t)
    gamma in (0,1) (vector).
    """
    def __init__(self, D_alpha: int, D_m: int, enc_hidden: int = 128, enc_layers: int = 2,
                 nl: str = "swish", scalar_gamma: bool = False, ln: bool = True):
        super().__init__()
        self.D_alpha, self.D_m = D_alpha, D_m
        self.scalar_gamma = bool(scalar_gamma)
        self.encoder = MLP(in_dim=D_alpha, hidden_dim=enc_hidden, out_dim=D_m,
                           num_layers=max(1, enc_layers), nl=nl,
                           last_zero_init=False, use_layernorm=ln)
        # raw_gamma -> gamma = sigmoid(raw_gamma) ∈ (0,1)
        g_shape = (1,) if self.scalar_gamma else (D_m,)
        self._raw_gamma = nn.Parameter(torch.zeros(*g_shape))

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self._raw_gamma)

    @torch.no_grad()
    def init_from_tau(self, tau_in_steps: float):
        """
        Set gamma ≈ exp(-1/tau) (discrete leakage). tau is in steps.
        """
        gamma0 = math.exp(-1.0 / max(1e-6, tau_in_steps))
        g = torch.full_like(self._raw_gamma, fill_value=gamma0)
        self._raw_gamma.copy_(_logit(g))

    def forward(self, alpha_t: torch.Tensor, m_t: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(alpha_t)            # [B, D_m]
        g = self.gamma                         # scalar or [D_m]
        return g * m_t + enc                   # [B, D_m]

    # stats (tau in steps) per dim if vector gamma; scalar otherwise
    @torch.no_grad()
    def tau_stats(self) -> Dict[str, float]:
        g = self.gamma.detach()
        g_vec = g if g.numel() > 1 else g.repeat(self.D_m)
        # tau = -1 / ln(g)
        eps = 1e-8
        tau = -1.0 / torch.log(g_vec.clamp(min=eps))
        out = dict(
            tau_steps_mean=float(_safe_mean(tau).item()),
            tau_steps_p50=float(_percentile(tau, 50).item()),
            tau_steps_p90=float(_percentile(tau, 90).item()),
        )
        return out


class GRUMemory(nn.Module):
    """
    Multi-layer GRU memory using nn.GRU. Top-layer hidden H[-1] is the memory vector m_t.
    step() consumes an already-encoded input x_t (dim = D_m).
    """
    def __init__(self, D_m: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.D_m = D_m
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size=D_m, hidden_size=D_m, num_layers=num_layers,
            batch_first=True, dropout=(dropout if num_layers > 1 else 0.0)
        )

    def init_state(self, B: int, device, dtype) -> torch.Tensor:  # (L,B,D_m)
        return torch.zeros(self.num_layers, B, self.D_m, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, H_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_t: (B, D_m) encoded input; H_prev: (L,B,D_m)
        Returns: m_next (B,D_m) from top layer, and H_next (L,B,D_m)
        """
        x = x_t.unsqueeze(1)                      # (B,1,D_m)
        out, H_next = self.rnn(x, H_prev)         # H_next: (L,B,D_m)
        m_next = H_next[-1]                       # top layer as memory vector
        return m_next, H_next

    @torch.no_grad()
    def tau_stats(self) -> Dict[str, float]:
        return {}  # cannot read gates from nn.GRU directly


class LSTMMemory(nn.Module): 
    """
    Multi-layer LSTM memory using nn.LSTM. Top-layer hidden H[-1] is the memory vector m_t.
    step() consumes an already-encoded input x_t (dim = D_m).
    """
    def __init__(self, D_m: int, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.D_m = D_m
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size=D_m, hidden_size=D_m, num_layers=num_layers,
            batch_first=True, dropout=(dropout if num_layers > 1 else 0.0)
        )

    def init_state(self, B: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:  # (H,C)
        H = torch.zeros(self.num_layers, B, self.D_m, device=device, dtype=dtype)
        C = torch.zeros(self.num_layers, B, self.D_m, device=device, dtype=dtype)
        return (H, C)

    def step(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x_t: (B,D_m); state=(H,C) each (L,B,D_m)
        Returns: m_next (B,D_m) from top hidden, and new (H,C).
        """
        x = x_t.unsqueeze(1)                          # (B,1,D_m)
        out, (H_next, C_next) = self.rnn(x, state)    # H_next: (L,B,D_m)
        m_next = H_next[-1]
        return m_next, (H_next, C_next)

    @torch.no_grad()
    def tau_stats(self) -> Dict[str, float]:
        return {}


# ============================== Residual LSTM over a sliding window ==============================
class ResidualLSTMWindow(nn.Module):
    """
    LSTM over a fixed-length context window. Returns a D-dimensional residual vector.

    Input:
        x_seq: (B, d, Din), where
               Din = D if augment=False
               Din = 2D if augment=True (concatenate [alpha_k || augment_feature_k])

    Design:
        - Run an LSTM over the window (chronological: oldest -> newest).
        - Take the last hidden state's linear projection as the residual.
        - The head is zero-initialized so the system starts as pure linear A alpha_t.
    """
    def __init__(self, D: int, augment: bool,
                 hidden: int = 256, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        in_dim = D * (2 if augment else 1)
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=(dropout if layers > 1 else 0.0)
        )
        self.head = nn.Linear(hidden, D)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:  # (B, d, Din)
        out, _ = self.lstm(x_seq)       # (B, d, H)
        h_last = out[:, -1, :]          # (B, H)
        return self.head(h_last)        # (B, D)


# ------------------------------ Discrete latent process ------------------------------
class LatentProcessDiscrete(nn.Module):
    """
    Discrete-time latent process with several residual mechanisms:

      1) Memory backends ('leaky' / 'gru' / 'lstm'):
           alpha_{t+1} = A alpha_t + gate ⊙ phi_dec(m_t)           (m_t updated from alpha_t)
        memory_type:
            - 'leaky' : m_{t} = gamma ⊙ m_{t-1} + psi_enc(alpha_t)
            - 'gru'   : multi-layer GRU over x_t = phi_enc(alpha_t); m_t is top-layer hidden
            - 'lstm'  : multi-layer LSTM over x_t = phi_enc(alpha_t); m_t is top-layer hidden

      2) Residual LSTM over a context window ('residlstm'):
           alpha_{t+1} = A alpha_t + ResidualLSTMWindow([alpha_{t-d+1}, ..., alpha_t])
         with optional feature augmentation along the CHANNEL dimension:
           - augment=False: use alpha_k only
           - augment=True, variant='history': concat [alpha_k || A alpha_k]
           - augment=True, variant='current': concat [alpha_k || A alpha_t] (same A alpha_t used for all k in the window)

    All branches share the same linear backbone A.
    """
    def __init__(self,
                 state_dim: int, code_dim: int,
                 memory_dim: Optional[int] = None,
                 memory_type: str = "leaky",           # {'leaky','gru','lstm','residual'}
                 # dec/enc MLPs for leaky backend
                 enc_hidden_dim: int = 128,
                 enc_layers: int = 2,
                 dec_hidden_dim: int = 128,
                 dec_layers: int = 2,
                 nl: str = "swish",
                 # RNN config (for gru/lstm)
                 rnn_layers: int = 2, 
                 rnn_dropout: float = 0.0, 
                 # gate
                 gate_per_dim: bool = True,
                 # init options
                 init_tau_steps: Optional[float] = None,  # for leaky: initialize gamma from tau,
                 use_layer_norm: bool = True,
                 # ---- residual LSTM window configs ----
                 context_window: int = 4,                    # d
                 window_pad: str = "repeat",                 # {'repeat','zero'} padding for the initial window
                 augment: bool = False,                      # channel-wise augmentation
                 augment_variant: str = "history",           # {'history','current'}
                 rnn_hidden: int = 256,
                 ):
        super().__init__()
        assert memory_type in {"leaky", "gru", "lstm", "residual"}
        assert window_pad in {"repeat", "zero"}
        assert augment_variant in {"history", "current"}

        self.state_dim = state_dim
        self.code_dim = code_dim
        self.D_alpha = state_dim * code_dim
        self.D_m = int(memory_dim) if memory_dim is not None else self.D_alpha
        self.memory_type = memory_type

        # Linear transition A (free)
        self.A = nn.Parameter(torch.zeros(self.D_alpha, self.D_alpha))

        if memory_type in {"leaky", "gru", "lstm"}:
            # Decoder: phi_dec(m) -> R^{D_alpha}. Last layer zero-init for small residuals.
            self.memory_decoder = MLP(in_dim=self.D_m, hidden_dim=dec_hidden_dim, out_dim=self.D_alpha,
                                    num_layers=max(1, dec_layers), nl=nl, last_zero_init=True, use_layernorm=use_layer_norm)

            # Gate g in (0,1): scalar or per-dim
            g_shape = (1,) if not gate_per_dim else (self.D_alpha,)
            self._raw_gate = nn.Parameter(torch.zeros(*g_shape))  # sigmoid->(0,1)

            # Shared encoder for RNN inputs: x_t = phi_enc(alpha_t) in R^{D_m}
            self.memory_encoder = MLP(in_dim=self.D_alpha, hidden_dim=enc_hidden_dim, out_dim=self.D_m,
                                    num_layers=max(1, enc_layers), nl=nl, last_zero_init=False, use_layernorm=use_layer_norm)

            # Memory backend
            if memory_type == "leaky":
                self.memory = LeakyMemory(self.D_alpha, self.D_m, enc_hidden=enc_hidden_dim,
                                        enc_layers=enc_layers, nl=nl, scalar_gamma=False, ln=use_layer_norm)
                if init_tau_steps is not None:
                    self.memory.init_from_tau(init_tau_steps)
            elif memory_type == "gru":
                self.memory = GRUMemory(D_m=self.D_m, num_layers=rnn_layers, dropout=rnn_dropout)
            else:  # lstm
                self.memory = LSTMMemory(D_m=self.D_m, num_layers=rnn_layers, dropout=rnn_dropout)
        
        if memory_type == "residual":
            assert context_window >= 1
            self.ctx = int(context_window)
            self.window_pad = window_pad
            self.augment = augment
            self.augment_variant = augment_variant
            self.res_lstm = ResidualLSTMWindow(
                D=self.D_alpha, augment=self.augment, hidden=rnn_hidden, layers=rnn_layers, dropout=rnn_dropout
            )

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self._raw_gate)

    @property
    def memory_dim(self) -> int:
        return self.D_m if self.memory_type in {"leaky", "gru", "lstm"} else self.D_alpha

    # ---- core step (one-step update) ----
    def _step_leaky(self, alpha_t: torch.Tensor, m_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # update memory first, then use the UPDATED memory for residual
        m_t = self.memory(alpha_t, m_prev)             # m_t = gamma ⊙ m_{t-1} + psi_enc(alpha_t)
        f_lin = alpha_t @ self.A.T                     # [B, D_alpha]
        f_mem = self.gate * self.memory_decoder(m_t)
        alpha_next = f_lin + f_mem
        return alpha_next, m_t

    def _step_gru_discrete(self, alpha_t: torch.Tensor, H_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t = self.memory_encoder(alpha_t)            # [B, D_m]
        m_t, H_next = self.memory.step(x_t, H_prev)   # m_t is UPDATED memory
        f_lin = alpha_t @ self.A.T
        f_mem = self.gate * self.memory_decoder(m_t)
        alpha_next = f_lin + f_mem
        return alpha_next, m_t, H_next

    def _step_lstm_discrete(self, alpha_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_t = self.memory_encoder(alpha_t)                 # [B, D_m]
        m_t, state_next = self.memory.step(x_t, state)     # m_t is UPDATED memory (top hidden)
        f_lin = alpha_t @ self.A.T
        f_mem = self.gate * self.memory_decoder(m_t)
        alpha_next = f_lin + f_mem
        return alpha_next, m_t, state_next

    @torch.no_grad()
    def _make_init_window(self, alpha0: torch.Tensor) -> torch.Tensor:
        """
        Build the initial window for t=0 given alpha_0.
        Returns (B, d, D_alpha), chronological (oldest -> newest).
        """
        B, D = alpha0.shape
        device, dtype = alpha0.device, alpha0.dtype
        if self.window_pad == "repeat":
            win = alpha0.unsqueeze(1).expand(B, self.ctx, D).clone()
        else:
            win = torch.zeros(B, self.ctx, D, device=device, dtype=dtype)
            win[:, -1, :] = alpha0
        return win

    def _append_step(self, win: torch.Tensor, alpha_t: torch.Tensor) -> torch.Tensor:
        """Slide window forward by 1 and append the new alpha_t. Shapes: win (B,d,D), alpha_t (B,D)."""
        return torch.cat([win[:, 1:, :], alpha_t.unsqueeze(1)], dim=1)

    def _build_lstm_input_from_window(self, win: torch.Tensor) -> torch.Tensor:
        """
        Channel-wise augmentation builder.

        Args:
            win: (B, d, D_alpha) chronological [alpha_{t-d+1}, ..., alpha_t]
        Returns:
            x_seq: (B, d, Din) where Din = D_alpha or 2*D_alpha depending on 'augment'.
        """
        if not self.augment:
            return win
        B, d, D = win.shape
        # History-based augmentation: concat [alpha_k || A alpha_k]
        if self.augment_variant == "history":
            Aalpha_hist = torch.einsum("bld,fd->blf", win, self.A)  # (B, d, D)
            return torch.cat([win, Aalpha_hist], dim=-1)
        # Current-based augmentation: concat [alpha_k || A alpha_t] for all k in the window
        alpha_t = win[:, -1, :]                    # (B, D)
        Aalpha_t = alpha_t @ self.A.T             # (B, D)
        tail = Aalpha_t.unsqueeze(1).expand(B, d, D).contiguous()
        return torch.cat([win, tail], dim=-1)

    # ---- rollout utilities ----
    def _rollout_discrete(self,
                          alpha_0: torch.Tensor,                 # [B, D_alpha]
                          T: int,
                          memory_init: Optional[torch.Tensor] = None,
                          teacher_forcing: bool = False,
                          tf_alpha: Optional[torch.Tensor] = None,   # [T, B, D_alpha]
                          tf_mask: Optional[torch.Tensor] = None,    # [T-1] True=end segment
                          tf_detach_alpha_starts: bool = True,
                          detach_memory_between_segments: bool = False
                          ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        B, D = alpha_0.size()
        device, dtype = alpha_0.device, alpha_0.dtype
        seg_mask = torch.zeros(T - 1, dtype=torch.bool, device=device) if T > 1 else torch.zeros(0, device=device, dtype=torch.bool)
        if tf_mask is not None:
            assert tf_mask.shape == seg_mask.shape
            seg_mask = tf_mask.to(device=device, dtype=torch.bool)

        # Init memory states (m_{-1})
        if self.memory_type == "leaky":
            m_prev = torch.zeros(B, self.D_m, device=device, dtype=dtype) if memory_init is None else memory_init.to(device=device, dtype=dtype)
            H = C = None
        elif self.memory_type == "gru":
            H = self.memory.init_state(B, device, dtype)
            if memory_init is not None:
                H[-1] = memory_init.to(device=device, dtype=dtype)  # put into top layer
            m_prev = None
            C = None
        else:  # lstm
            H, C = self.memory.init_state(B, device, dtype)
            if memory_init is not None:
                H[-1] = memory_init.to(device=device, dtype=dtype)
            m_prev = None

        alpha_seq = torch.zeros(T, B, D, device=device, dtype=dtype)
        mem_seq   = torch.zeros(T, B, self.D_m, device=device, dtype=dtype)

        # Initial alpha (maybe TF at t=0)
        alpha_t = alpha_0
        if teacher_forcing and tf_alpha is not None:
            alpha_t = tf_alpha[0].detach() if tf_detach_alpha_starts else tf_alpha[0]
        alpha_seq[0] = alpha_t

        # Rollout with "update memory first, then residual"
        for t in range(T):
            if self.memory_type == "leaky":
                # m_t from (m_{t-1}, alpha_t)
                if t == 0:
                    # m_prev already holds m_{-1}
                    pass
                alpha_next, m_t = self._step_leaky(alpha_t, m_prev)   # step returns (alpha_{t+1}, m_t)
                mem_seq[t] = m_t                                      # record m_t
                m_prev = m_t
            elif self.memory_type == "gru":
                x_t = self.memory_encoder(alpha_t)
                m_t, H = self.memory.step(x_t, H)                     # updated memory
                mem_seq[t] = m_t 
                alpha_next = alpha_t @ self.A.T + self.gate * self.memory_decoder(m_t)
            else:
                x_t = self.memory_encoder(alpha_t)
                m_t, (H, C) = self.memory.step(x_t, (H, C))           # updated memory (top hidden)
                mem_seq[t] = m_t
                alpha_next = alpha_t @ self.A.T + self.gate * self.memory_decoder(m_t)

            # produce alpha_{t+1} only if t < T-1
            if t < T - 1:
                alpha_seq[t + 1] = alpha_next

                # teacher forcing boundary
                if seg_mask[t] and teacher_forcing and (tf_alpha is not None):
                    alpha_t = tf_alpha[t + 1].detach() if tf_detach_alpha_starts else tf_alpha[t + 1]
                    if detach_memory_between_segments:
                        if self.memory_type == "leaky":
                            m_prev = m_prev.detach()
                        elif self.memory_type == "gru":
                            H = H.detach()
                        else:
                            H, C = H.detach(), C.detach()
                else:
                    alpha_t = alpha_next

        # Build aux
        aux: Dict[str, torch.Tensor | float] = {}
        dec_flat = self.memory_decoder(mem_seq.reshape(-1, self.D_m))
        aux["phi_dec_l2"] = (dec_flat.pow(2).sum(dim=-1)).mean()

        # energy ratio uses m_t (mem_seq[:-1]) for steps producing alpha_{t+1}
        f_lin = (alpha_seq[:-1] @ self.A.T)
        f_mem = (self.gate * self.memory_decoder(mem_seq[:-1].reshape(-1, self.D_m))).view(T-1, B, D)
        e_lin = (f_lin ** 2).sum(dim=-1)
        e_mem = (f_mem ** 2).sum(dim=-1)
        r = e_mem / (e_lin + e_mem + 1e-8)
        aux["mem_ratio_mean"] = float(r.mean().item())
        aux["mem_ratio_p90"] = float(_percentile(r, 90).item())

        if self.memory_type == "leaky":
            aux.update(self.memory.tau_stats())

        gate_vec = self.gate.detach()
        aux["gate_mean"] = float(gate_vec.mean().item())
        return alpha_seq, mem_seq, aux

    def _rollout_residlstm(self,
                           alpha_0: torch.Tensor, t_eval: torch.Tensor,
                           teacher_forcing: bool, tf_alpha: Optional[torch.Tensor],
                           tf_mask: Optional[torch.Tensor], tf_detach_alpha_starts: bool
                           ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Rollout using: alpha_{t+1} = A alpha_t + ResidualLSTMWindow(window_t),
        where window_t = [alpha_{t-d+1}, ..., alpha_t] (chronological).
        Returns:
            alpha_seq: (T, B, D)
            mem_seq  : (T, B, D) storing the per-step residual (last step left as zeros)
            aux      : dict of monitoring scalars
        """
        device, dtype = alpha_0.device, alpha_0.dtype
        B, D = alpha_0.shape
        T = t_eval.numel()

        # Build TF segment mask (if any)
        seg_mask = torch.zeros(T - 1, dtype=torch.bool, device=device) if T > 1 else torch.zeros(0, device=device, dtype=torch.bool)
        if tf_mask is not None:
            assert tf_mask.shape == seg_mask.shape
            seg_mask = tf_mask.to(device=device, dtype=torch.bool)

        alpha_seq = torch.zeros(T, B, D, device=device, dtype=dtype)
        resid_seq = torch.zeros(T, B, D, device=device, dtype=dtype)  # store residuals used at each step

        # Initialize alpha_0 and the initial window
        alpha_t = alpha_0
        win = self._make_init_window(alpha_t)
        alpha_seq[0] = alpha_t

        # Optional teacher-forced start
        if teacher_forcing and (tf_alpha is not None):
            alpha_t = tf_alpha[0].detach() if tf_detach_alpha_starts else tf_alpha[0]
            alpha_seq[0] = alpha_t
            if self.window_pad == "repeat":
                win = alpha_t.unsqueeze(1).expand(B, self.ctx, D).clone()
            else:
                win = torch.zeros(B, self.ctx, D, device=device, dtype=dtype)
                win[:, -1, :] = alpha_t

        # Rollout
        for t in range(T - 1):
            # Build channel-augmented LSTM input from the current window
            x_seq = self._build_lstm_input_from_window(win)           # (B, d, Din)

            # Compute residual and linear part
            resid_t = self.res_lstm(x_seq)                            # (B, D)
            Aalpha_t = (win[:, -1, :] @ self.A.T)                     # (B, D)

            # Update alpha
            alpha_next = Aalpha_t + resid_t
            alpha_seq[t + 1] = alpha_next
            resid_seq[t] = resid_t

            # Teacher forcing boundary?
            if bool(seg_mask[t] and teacher_forcing and (tf_alpha is not None)):
                alpha_t = tf_alpha[t + 1].detach() if tf_detach_alpha_starts else tf_alpha[t + 1]
            else:
                alpha_t = alpha_next

            # Slide the window
            win = self._append_step(win, alpha_t)

        # Aux metrics
        with torch.no_grad():
            lin = (alpha_seq[:-1] @ self.A.T)                # (T-1, B, D)
            nonlin = alpha_seq[1:] - lin                     # (T-1, B, D)
            e_lin = (lin ** 2).sum(dim=-1)
            e_nl  = (nonlin ** 2).sum(dim=-1)
            ratio = e_nl / (e_lin + e_nl + 1e-8)
            aux = dict(
                mem_ratio_mean = float(ratio.mean().item()),
                mem_ratio_p90  = float(torch.quantile(ratio.flatten(), 0.9).item()),
                phi_dec_l2     = e_nl.mean(),
            )
        return alpha_seq, resid_seq, aux
    
    # Public API
    def forward(self,
                alpha_0: torch.Tensor,              # [B, D_alpha]
                t_eval: torch.Tensor,               # [T]
                memory_init: Optional[torch.Tensor] = None,
                teacher_forcing: bool = False,
                tf_alpha: Optional[torch.Tensor] = None,   # [T, B, D_alpha]
                tf_epsilon: float = 0.0,
                tf_mask: Optional[torch.Tensor] = None,
                tf_detach_alpha_starts: bool = True,
                detach_memory_between_segments: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        T = t_eval.numel()
        if (tf_mask is None) and teacher_forcing and (T >= 2) and (tf_epsilon > 1e-8):
            with torch.no_grad():
                tf_mask = (torch.rand(T - 1, device=alpha_0.device) < float(tf_epsilon))
                tf_mask[-1] = False
        
        if self.memory_type == "residual":
            return self._rollout_residlstm(
                alpha_0=alpha_0, t_eval=t_eval,
                teacher_forcing=teacher_forcing, tf_alpha=tf_alpha,
                tf_mask=tf_mask, tf_detach_alpha_starts=tf_detach_alpha_starts
            )

        return self._rollout_discrete(
            alpha_0=alpha_0,
            T=T,
            memory_init=memory_init,
            teacher_forcing=teacher_forcing,
            tf_alpha=tf_alpha,
            tf_mask=tf_mask,
            tf_detach_alpha_starts=tf_detach_alpha_starts,
            detach_memory_between_segments=detach_memory_between_segments,
        )
