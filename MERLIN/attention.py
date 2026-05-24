import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, orthogonal_
from einops import rearrange, repeat


# New position encoding module
# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)


class LinearAttention(nn.Module):
    """
    Two linearized / kernelized attention variants ("Choose a Transformer: Fourier or Galerkin"):
    - Galerkin  : InstanceNorm1d on K and V
    - Fourier   : InstanceNorm1d on Q and K

    Goal: avoid explicit softmax(QK^T) with O(N^2) cost by using associativity to get O(N * d^2).
    """
    def __init__(self,
                 dim,                       # token feature dim (per point)
                 attn_type,                 # 'fourier' or 'galerkin'
                 heads=8,
                 dim_head=64,               # per-head dim d
                 dropout=0.,
                 init_params=True,          # custom init for to_qkv
                 relative_emb=False,        # rotary pos-emb (RoPE) on q,k
                 scale=1.,                  # spatial scale used by RoPE
                 init_method='orthogonal',  # qkv weight init method
                 init_gain=None,            # init gain (default=1/d)
                 relative_emb_dim=2,        # 1D or 2D rotary
                 min_freq=1/64,             # base freq (match grid res)
                 cat_pos=False,             # if no RoPE, concat absolute pos to q,k,v
                 pos_dim=2,                 # abs pos dim (2 in 2D)
                 norm_type='layer',         # 'instance' normalization / 'layer' normalization
                 ):
        super().__init__()

        inner_dim = dim_head * heads        # H * d
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type
        self.heads = heads
        self.dim_head = dim_head
        self.norm_type = norm_type

        # Single linear to produce q,k,v then split on the last dim
        # x: [B, N, dim] -> [B, N, 3*H*d]
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Which tensors to normalize depends on the variant
        if norm_type not in ['instance', 'layer']:
            raise ValueError(f'Unknown norm_type {norm_type}')
        norm_cls = nn.LayerNorm if norm_type == 'layer' else lambda dim: nn.InstanceNorm1d(dim, affine=False)
        if attn_type == 'galerkin':
            # Galerkin: normalize K and V
            self.k_norm = norm_cls(dim_head)
            self.v_norm = norm_cls(dim_head)
        elif attn_type == 'fourier':
            # Fourier: normalize Q and K
            self.q_norm = norm_cls(dim_head)
            self.k_norm = norm_cls(dim_head)
        else:
            raise Exception(f'Unknown attention type {attn_type}')

        # Output projection: concat heads (H*d) -> dim
        # If cat_pos=True we also pass concatenated pos through the proj
        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim * heads, dim),
                nn.Dropout(dropout)
            )

        # Init gain & tiny diagonal bias (stabilizes early training)
        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain

        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        # Rotary positional embedding (relative)
        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            assert not cat_pos  # mutually exclusive
            # Each coordinate consumes dim_head // relative_emb_dim channels
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim,
                                             min_freq=min_freq, scale=scale)
    
    def _init_params(self):
        # Choose the weight init function for linear layers
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        # Iterate over parameters of the to_qkv linear layer
        for param in self.to_qkv.parameters():
            # Skip bias (1-D tensors) — we only touch the weight matrix
            if param.ndim <= 1:
                continue

            # Shapes:
            # to_qkv.weight has shape [3 * H * d, in_dim], where:
            #   H = number of heads, d = dim_head, in_dim = input feature dim
            H, d = self.heads, self.dim_head
            in_dim = param.size(-1)              # columns of the weight matrix

            # We will add a tiny "skinny identity" on a rectangular block [d, in_dim].
            # Only min(d, in_dim) diagonal entries exist in that rectangle.
            diag_len = min(d, in_dim)
            ar = torch.arange(diag_len, device=param.device)  # indices 0..diag_len-1

            # Loop over heads and initialize the per-head block
            for h in range(H):
                if self.attn_type == 'fourier':
                    # For Fourier: initialize the V block for head h
                    # Row range for V block: rows [(2H + h) * d : (2H + h + 1) * d]
                    start = (2 * H + h) * d
                    end   = start + d

                    # Initialize that [d, in_dim] slice
                    init_fn(param[start:end, :], gain=self.init_gain)

                    # Add skinny identity: for i in [0..diag_len-1], bump (row=start+i, col=i)
                    # This nudges the block toward an identity-like mapping without shape mismatch
                    param.data[start:start + diag_len, ar] += self.diagonal_weight

                else:  # 'galerkin'
                    # For Galerkin: initialize the Q block for head h
                    # Row range for Q block: rows [h * d : (h + 1) * d]
                    start = h * d
                    end   = start + d

                    # Initialize that [d, in_dim] slice
                    init_fn(param[start:end, :], gain=self.init_gain)

                    # Add the same skinny identity on the Q block
                    param.data[start:start + diag_len, ar] += self.diagonal_weight

    def norm_wrt_domain(self, x, norm_fn):
        """
        Apply InstanceNorm1d over the 'domain' axis (tokens):
        x: [B, H, N, d]  (H: heads, N: tokens/points, d: head dim)
        InstanceNorm1d expects [N_like, C, L] with C=channel.
        We fold heads into batch, treat channels=d, length=N.
        """
        if self.norm_type == 'layer':
            return norm_fn(x)
        b = x.shape[0]
        x_bn = rearrange(x, 'b h n d -> (b h) d n')  # (B*H, d, N)
        x_bn = norm_fn(x_bn)
        return rearrange(x_bn, '(b h) d n -> b h n d', b=b)

    def forward(self, x, pos=None, not_assoc=False):
        """
        x   : [B, N, dim]   token features per point
        pos : [B, N, 2] or [B, N, 1] absolute coords
              (required if relative_emb=True; also used if cat_pos=True)
        return: [B, N, dim] ([B, N, dim] -> [B, N, dim])
        """
        # 1) q,k,v and split into heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)                   # 3 * [B, N, H*d]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # each: [B,H,N,d]

        # pos is required when using relative positional embedding
        if pos is None and self.relative_emb:
            raise Exception('Must pass coordinates when relative position embedding is enabled')

        # 2) InstanceNorm for stability (variant-specific)
        if self.attn_type == 'galerkin':
            k = self.norm_wrt_domain(k, self.k_norm)
            v = self.norm_wrt_domain(v, self.v_norm)
        else:  # fourier
            q = self.norm_wrt_domain(q, self.q_norm)
            k = self.norm_wrt_domain(k, self.k_norm)

        # 3) Position handling: relative (RoPE) or absolute concatenation
        if self.relative_emb:
            if self.relative_emb_dim == 2:
                # 2D RoPE: build freq embeddings for x and y, then apply to q,k
                freqs_x = self.emb_module.forward(pos[..., 0], x.device)  # [B, N, d/2]
                freqs_y = self.emb_module.forward(pos[..., 1], x.device)  # [B, N, d/2]
                freqs_x = repeat(freqs_x, 'b n d -> b h n d', h=q.shape[1])  # [B,H,N,d/2]
                freqs_y = repeat(freqs_y, 'b n d -> b h n d', h=q.shape[1])  # [B,H,N,d/2]
                q = apply_2d_rotary_pos_emb(q, freqs_x, freqs_y)
                k = apply_2d_rotary_pos_emb(k, freqs_x, freqs_y)
            elif self.relative_emb_dim == 1:
                # 1D RoPE
                assert pos.shape[-1] == 1
                freqs = self.emb_module.forward(pos[..., 0], x.device)     # [B, N, d]
                freqs = repeat(freqs, 'b n d -> b h n d', h=q.shape[1])     # [B,H,N,d]
                q = apply_rotary_pos_emb(q, freqs)
                k = apply_rotary_pos_emb(k, freqs)
            else:
                raise Exception('relative_emb_dim > 2 is not supported')
        elif self.cat_pos:
            # basis enrichment using coordinates
            # Concatenate absolute coordinates to q,k,v if no relative embedding
            assert pos.size(-1) == self.pos_dim
            pos_heads = pos.unsqueeze(1).repeat([1, self.heads, 1, 1])  # [B,H,N,pos_dim]
            q, k, v = [torch.cat([pos_heads, t], dim=-1) for t in (q, k, v)]  # head dim becomes d+pos_dim

        # 4) Linear attention via associativity
        if not_assoc:
            # Non-associative path: A = q @ k^T (B,H,N,N) then out = A @ v
            # Cost: O(B*H*N*N*d) — only preferable if N << d
            score = torch.matmul(q, k.transpose(-1, -2))              # [B,H,N,N]
            out = torch.matmul(score, v) * (1. / q.shape[2])          # normalize by N
        else:
            # Associative path (default): B = k^T @ v (B,H,d,d) then out = q @ B
            # Cost: O(B*H*N*d*d) — much cheaper when N >> d (common)
            dots = torch.matmul(k.transpose(-1, -2), v)               # [B,H,d,d]
            out  = torch.matmul(q, dots) * (1. / q.shape[2])          # [B,H,N,d]

        # 5) Merge heads and project out
        out = rearrange(out, 'b h n d -> b n (h d)')                   # [B, N, H*d]
        return self.to_out(out)                                        # [B, N, dim]



# -------------------------------------------
# Cross Linear Attention (Q from latents, K/V from spatial features)
# -------------------------------------------

class CrossLinearAttention(nn.Module):
    """
    Linearized cross-attention with associativity:
      Q from latents  [B, T, dim_q]
      K,V from spatial features [B, N, dim_kv]
    Compute: dots = K^T @ V  -> [B,H,d,d],  out = Q @ dots / N  (no explicit O(N^2) softmax).
    Variants:
      - 'galerkin' : InstanceNorm1d on K and V
      - 'fourier'  : InstanceNorm1d on Q and K
    Positional handling:
      - Relative (RoPE, 1D/2D): acts on Q,K
      - Absolute concat (cat_pos): concatenates positions to Q,K,V (mutually exclusive with RoPE)
    """

    def __init__(self,
                 dim_q: int, dim_kv: int,
                 heads: int = 8, dim_head: int = 64,
                 attn_type: str = 'galerkin',      # 'galerkin' | 'fourier'
                 dropout: float = 0.,
                 relative_emb: bool = True,
                 relative_emb_dim: int = 2,        # 1 for 1D; 2 for 2D
                 min_freq: float = 1/64,
                 scale: float = 1.,
                 cat_pos: bool = False,
                 pos_dim: int = 2,
                 # init controls (migrated from the older version)
                 init_params: bool = True,
                 init_method: str = 'orthogonal',  # 'xavier' | 'orthogonal'
                 init_gain: float | None = None,
                 norm_type: str = 'instance'):
        super().__init__()
        assert attn_type in ['galerkin', 'fourier']
        if relative_emb:
            assert not cat_pos, "relative_emb and cat_pos are mutually exclusive"
            assert dim_head % relative_emb_dim == 0, "dim_head must be divisible by relative_emb_dim"

        self.heads = heads
        self.dim_head = dim_head
        self.attn_type = attn_type
        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        self.cat_pos = cat_pos
        self.pos_dim = pos_dim
        self.norm_type = norm_type

        inner = heads * dim_head

        # Separate projections for cross-attention
        self.to_q = nn.Linear(dim_q,  inner, bias=False)
        self.to_k = nn.Linear(dim_kv, inner, bias=False)
        self.to_v = nn.Linear(dim_kv, inner, bias=False)

        # Variant-specific normalization
        if norm_type not in ['instance', 'layer']:
            raise ValueError(f'Unknown norm_type {norm_type}')
        norm_cls = nn.LayerNorm if norm_type == 'layer' else lambda dim: nn.InstanceNorm1d(dim, affine=False)
        if attn_type == 'galerkin':
            # Normalize K and V.
            self.k_norm = norm_cls(dim_head)
            self.v_norm = norm_cls(dim_head)
        else:  # 'fourier'
            # Normalize Q and K.
            self.q_norm = norm_cls(dim_head)
            self.k_norm = norm_cls(dim_head)

        # Output projection: heads*dim_head (+ pos_dim*heads if cat_pos) -> dim_q
        out_in = inner + (pos_dim * heads if cat_pos else 0)
        self.to_out = nn.Sequential(
            nn.Linear(out_in, dim_q),
            nn.Dropout(dropout)
        )

        # Relative positional embedding (RoPE)
        if relative_emb:
            self.emb_module = RotaryEmbedding(dim_head // relative_emb_dim,
                                              min_freq=min_freq, scale=scale)

        # Initialization controls (migrated skinny-diagonal init)
        self.init_method = init_method
        self.init_gain = (1. / dim_head) if init_gain is None else init_gain
        self.diagonal_weight = self.init_gain
        if init_params:
            self._init_params()

    # ---------- initialization (migrated & adapted) ----------
    def _init_params(self):
        """
        Per-head block initialization for to_q / to_k / to_v, with a "skinny identity"
        added on the min(d, in_dim) diagonal to encourage early stability.
        This mirrors the spirit of the older implementation but adapts
        to separate q/k/v projections and possibly rectangular weight matrices.
        """
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise ValueError('Unknown initialization')

        H, d = self.heads, self.dim_head

        def init_linear(linear: nn.Linear):
            # linear.weight: [H*d, in_dim]
            w = linear.weight
            if w.ndim <= 1:
                return
            in_dim = w.size(-1)
            diag_len = min(d, in_dim)
            ar = torch.arange(diag_len, device=w.device)
            for h in range(H):
                s = h * d
                e = s + d
                init_fn(w[s:e, :], gain=self.init_gain)                 # per-head block init
                w.data[s:s+diag_len, ar] += self.diagonal_weight        # skinny diagonal bias

        init_linear(self.to_q)
        init_linear(self.to_k)
        init_linear(self.to_v)

    # ---------- helpers ----------
    @staticmethod
    def _norm_over_tokens(x, norm_fn, norm_type='instance'):
        """
        Apply InstanceNorm1d over the token axis.
        x: [B, H, L, d] -> (B*H, d, L) -> IN1d(d) -> [B, H, L, d]
        """
        if norm_type == 'layer':
            return norm_fn(x)
        b = x.shape[0]
        x_bn = rearrange(x, 'b h l d -> (b h) d l')
        x_bn = norm_fn(x_bn)
        return rearrange(x_bn, '(b h) d l -> b h l d', b=b)

    # ---------- forward ----------
    def forward(self,
                q_lat: torch.Tensor,                # [B, T, dim_q]
                kv_mem: torch.Tensor,               # [B, N, dim_kv]
                pos_q: torch.Tensor | None = None,  # [B, T, pos_dim] or None
                pos_kv: torch.Tensor | None = None  # [B, N, pos_dim] or None
                ) -> torch.Tensor:                  # returns [B, T, dim_q]
        B, T, _ = q_lat.shape
        N = kv_mem.shape[1]

        # 1) Projections and split into heads
        q = rearrange(self.to_q(q_lat),  'b t (h d) -> b h t d', h=self.heads)  # [B,H,T,d]
        k = rearrange(self.to_k(kv_mem), 'b n (h d) -> b h n d', h=self.heads)  # [B,H,N,d]
        v = rearrange(self.to_v(kv_mem), 'b n (h d) -> b h n d', h=self.heads)  # [B,H,N,d]

        # 2) InstanceNorm/LayerNorm for stability
        if self.attn_type == 'galerkin':
            k = self._norm_over_tokens(k, self.k_norm, self.norm_type)
            v = self._norm_over_tokens(v, self.v_norm, self.norm_type)
        else:  # 'fourier'
            q = self._norm_over_tokens(q, self.q_norm, self.norm_type)
            k = self._norm_over_tokens(k, self.k_norm, self.norm_type)

        # 3) Positional handling
        if self.relative_emb:
            assert (pos_q is not None) and (pos_kv is not None), \
                "pos_q and pos_kv are required when relative_emb=True"

            if self.relative_emb_dim == 2:
                # Q side (T tokens)
                fq_x = self.emb_module.forward(pos_q[..., 0], q_lat.device)  # [B,T,d/2]
                fq_y = self.emb_module.forward(pos_q[..., 1], q_lat.device)  # [B,T,d/2]
                fq_x = repeat(fq_x, 'b t d -> b h t d', h=self.heads)
                fq_y = repeat(fq_y, 'b t d -> b h t d', h=self.heads)
                q = apply_2d_rotary_pos_emb(q, fq_x, fq_y)

                # K side (N tokens)
                fk_x = self.emb_module.forward(pos_kv[..., 0], kv_mem.device)  # [B,N,d/2]
                fk_y = self.emb_module.forward(pos_kv[..., 1], kv_mem.device)  # [B,N,d/2]
                fk_x = repeat(fk_x, 'b n d -> b h n d', h=self.heads)
                fk_y = repeat(fk_y, 'b n d -> b h n d', h=self.heads)
                k = apply_2d_rotary_pos_emb(k, fk_x, fk_y)

            elif self.relative_emb_dim == 1:
                fq = self.emb_module.forward(pos_q[..., 0], q_lat.device)      # [B,T,d]
                fq = repeat(fq, 'b t d -> b h t d', h=self.heads)
                q  = apply_rotary_pos_emb(q, fq)

                fk = self.emb_module.forward(pos_kv[..., 0], kv_mem.device)    # [B,N,d]
                fk = repeat(fk, 'b n d -> b h n d', h=self.heads)
                k  = apply_rotary_pos_emb(k, fk)

            else:
                raise ValueError('relative_emb_dim > 2 not supported')

        elif self.cat_pos:
            # Concatenate absolute coords to Q, K, V (d -> d + pos_dim)
            assert (pos_q is not None) and (pos_kv is not None), \
                "pos_q and pos_kv are required when cat_pos=True"
            pos_qh  = pos_q.unsqueeze(1).repeat(1, self.heads, 1, 1)    # [B,H,T,pos_dim]
            pos_kvh = pos_kv.unsqueeze(1).repeat(1, self.heads, 1, 1)   # [B,H,N,pos_dim]
            q = torch.cat([pos_qh,  q], dim=-1)  # [B,H,T,d+pos_dim]
            k = torch.cat([pos_kvh, k], dim=-1)  # [B,H,N,d+pos_dim]
            v = torch.cat([pos_kvh, v], dim=-1)  # [B,H,N,d+pos_dim]

        # 4) Linearized cross-attention via associativity
        #    dots = K^T V  -> [B,H,d,d] (or d+pos_dim), then out = Q @ dots / N
        dots = torch.matmul(k.transpose(-1, -2), v)     # [B,H,d,d]
        out  = torch.matmul(q, dots) * (1. / N)         # [B,H,T,d]

        # 5) Merge heads and project out
        out = rearrange(out, 'b h t d -> b t (h d)')    # [B,T,H*d(+pos)]
        return self.to_out(out)                         # [B,T,dim_q]