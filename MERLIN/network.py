from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import math
from torch import nn
from torch.nn import init
from torch import Tensor
from torch.nn.parameter import Parameter

from einops import rearrange


#####################################################################
################################ MLP ################################
#####################################################################
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


nls = {'relu': partial(nn.ReLU),
       'sigmoid': partial(nn.Sigmoid),
       'leakyrelu': partial(nn.LeakyReLU),
       'tanh': partial(nn.Tanh),
       'selu': partial(nn.SELU),
       'softplus': partial(nn.Softplus),
       'gelu': partial(nn.GELU),
       'swish': partial(Swish),
       'elu': partial(nn.ELU)}


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None, num_layers=3, nl='swish',
                 *, last_bias=True, last_kaiming=False, last_kaiming_a=0.0,
                 last_zero_init: bool = False, last_gain: float | None = None,
                 use_layernorm: bool = False, ln_eps: float = 1e-5, ln_affine: bool = True,
                 norm_where: str = "pre"):  # "pre" | "post" | "none"
        super().__init__()
        out_dim = in_dim if out_dim is None else out_dim

        # Layer normalizations
        self.norm_where = norm_where
        self.ln_in  = nn.LayerNorm(in_dim,  eps=ln_eps, elementwise_affine=ln_affine) if use_layernorm and norm_where in ("pre", "both") else nn.Identity()
        self.ln_out = nn.LayerNorm(out_dim, eps=ln_eps, elementwise_affine=ln_affine) if use_layernorm and norm_where in ("post","both") else nn.Identity()

        layers = []
        # input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nls[nl]())

        # hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nls[nl]())

        # output layer
        layers.append(nn.Linear(hidden_dim, out_dim, bias=last_bias))

        self.net = nn.Sequential(*layers)

        last_layer: nn.Linear = self.net[-1]
        if last_zero_init:
            with torch.no_grad():
                last_layer.weight.zero_()
                if last_layer.bias is not None:
                    last_layer.bias.zero_()
        elif last_gain is not None:
            nn.init.kaiming_uniform_(last_layer.weight, a=last_kaiming_a if last_kaiming else 0.0)
            with torch.no_grad():
                last_layer.weight.mul_(last_gain)
                if last_layer.bias is not None:
                    last_layer.bias.zero_()
        elif last_kaiming:
            nn.init.kaiming_uniform_(last_layer.weight, a=last_kaiming_a)    # last_kaiming_a = math.sqrt(5)

    def forward(self, x):
        if not isinstance(self.ln_in, nn.Identity):
            x = self.ln_in(x)          # pre-norm
        x = self.net(x)
        if not isinstance(self.ln_out, nn.Identity):
            x = self.ln_out(x)         # post-norm
        return x
    

class GeGELU(nn.Module):
    """https://paperswithcode.com/method/geglu"""

    def __init__(self):
        super().__init__()
        self.fn = nn.GELU()

    def forward(self, x):
        c = x.shape[-1]  # channel last arrangement
        return self.fn(x[..., :int(c//2)]) * x[..., int(c//2):]


class FeedForward(nn.Module):
    """
    [..., dim] -> [..., dim]
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim*2),
            GeGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ReLUFeedForward(nn.Module):
    """
    [..., dim] -> [..., dim]
    """
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


#####################################################################
############################ FourierNet #############################
#####################################################################
class Bilinear(nn.Module):
    __constants__ = ['in1_feat_dim', 'in2_feat_dim', 'out_feat_dim']
    
    in1_feat_dim: int
    in2_feat_dim: int
    out_feat_dim: int

    def __init__(self, in1_feat_dim: int, in2_feat_dim: int, out_feat_dim: int, device=None, dtype=None) -> None:
        """
        Constructs a bilinear-like module that combines two types of inputs:
        - input1: spatial coordinates or features, shape [b, t, h, w, s, in1_feat_dim]
        - input2: control/code(latent) vector, shape [b, t, s, in2_feat_dim]
        - Output: [b, t, h, w, s, out_feat_dim]
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Bilinear, self).__init__()

        self.in1_feat_dim = in1_feat_dim  # Input feature size from input1 (e.g., spatial)
        self.in2_feat_dim = in2_feat_dim  # Input feature size from input2 (e.g., code)
        self.out_feat_dim = out_feat_dim  # Desired output feature dimension

        # Parameter A: maps input1 to output features
        # Shape: [out_feat_dim, in1_feat_dim]
        self.A = Parameter(torch.empty(out_feat_dim, in1_feat_dim, **factory_kwargs))

        # Parameter B: maps input2 to output features
        # Shape: [out_feat_dim, in2_feat_dim]
        self.B = Parameter(torch.empty(out_feat_dim, in2_feat_dim, **factory_kwargs))

        # Bias term: one bias per output feature
        self.bias = Parameter(torch.empty(out_feat_dim, **factory_kwargs))

        # Initialize A, B, and bias
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initializes the parameters:
        - A and B with Kaiming uniform initialization
        - bias with uniform distribution within a calculated bound
        """
        bound = 1 / math.sqrt(self.in1_feat_dim)
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.B, a=math.sqrt(5))
        init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        # input1: [b,t,h,w,s,in1_feat_dim]
        # input2: [b,t,s,in2_feat_dim]
        b, t, h, w, s, _ = input1.shape

        # 1. Input1 projection: reshape + matmul
        input1_flat = input1.reshape(-1, self.in1_feat_dim)  # [b*t*h*w*s, in1]
        input1_proj = input1_flat @ self.A.t()               # [b*t*h*w*s, out]
        input1_proj = input1_proj.view(b, t, h, w, s, self.out_feat_dim)

        # 2. Input2 projection: reshape + matmul
        input2_flat = input2.reshape(-1, self.in2_feat_dim)
        input2_proj = input2_flat @ self.B.t()  # [b*t*s, out_feat_dim]
        input2_proj = input2_proj.view(b, t, s, self.out_feat_dim)
        # unsqueeze for spatial broadcast: [b,t,1,1,s,out_feat_dim]
        input2_proj = input2_proj.unsqueeze(2).unsqueeze(2)

        # 3. Combine with in-place ops to reduce temporaries
        result = input1_proj
        result = result.add(input2_proj)
        result = result.add(self.bias.view(1,1,1,1,1,-1))
        return result

    def extra_repr(self) -> str:
        """
        Adds extra information when printing the module
        """
        return 'in1_feat_dim={}, in2_feat_dim={}, out_feat_dim={}, bias={}'.format(
            self.in1_feat_dim, self.in2_feat_dim, self.out_feat_dim, self.bias is not None)

    
class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.
    Adapted from https://github.com/boschresearch/multiplicative-filter-networks
    Expects the child class to define the 'filters' attribute, which should be 
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_feat_dim.
    """
    def __init__(self, grid_dim, hidden_feat_dim, code_dim, out_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_feat_dim = hidden_feat_dim

        self.bilinear = nn.ModuleList(
            [Bilinear(grid_dim, code_dim, hidden_feat_dim)] +
            [Bilinear(hidden_feat_dim, code_dim, hidden_feat_dim) for _ in range(int(n_layers))]
        )
        self.output_bilinear = nn.Linear(hidden_feat_dim, out_dim)

        # spatial filters
        self.filters = nn.ModuleList()

    def forward(self, grid: Tensor, latent_field: Tensor) -> Tensor:
        """
        Inputs:
        - grid: [h, w, grid_dim]
        - latent_field: Tensor of shape [b, t, s, code_dim]
        Returns:
        - out: Tensor of shape [b, t, h, w, s, out_dim]
             Output after applying filter and bilinear modulation.
        """
        b, t, s, _ = latent_field.shape
        h, w, grid_dim = grid.shape

        pos_emb0 = self.filters[0](grid).unsqueeze(0).unsqueeze(0).unsqueeze(4)    # expand b, t, s axis
        pos_emb0 = pos_emb0.expand(b, t, h, w, s, self.hidden_feat_dim) #??????????
        
        hidden_feat0 = self.bilinear[0](
            input1=torch.zeros((b, t, h, w, s, grid_dim), device=latent_field.device),
            input2=latent_field
        )
        out = pos_emb0 * hidden_feat0
        
        # subsequent layers
        for i in range(1, self.n_layers + 1):
            pos_embi = self.filters[i](grid).unsqueeze(0).unsqueeze(0).unsqueeze(4)
            pos_embi = pos_embi.expand(b, t, h, w, s, self.hidden_feat_dim)
            hidden_feati = self.bilinear[i](input1=out, input2=latent_field)
            out = pos_embi * hidden_feati

        out = self.output_bilinear(out)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out
    

class FourierLayer(nn.Module):
    """
    Adapted from https://github.com/boschresearch/multiplicative-filter-networks
    """
    def __init__(self, in_feat_dim, out_feat_dim, weight_scale):
        super().__init__()
        self.out_feat_dim = out_feat_dim
        self.weight = Parameter(torch.empty((out_feat_dim//2, in_feat_dim)))
        self.weight_scale = weight_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor):
        # x: [h, w, grid_dim] (only last axis viewed as coords)
        # also valid for input like [B, grid_dim]
        grid_dim = x.shape[-1]
        x_shape = list(x.shape)
        out_shape = x_shape
        out_shape[-1] = self.out_feat_dim
        x_flat = x.view(-1, grid_dim)
        proj = F.linear(x_flat, self.weight * self.weight_scale)
        out = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return out.view(out_shape)
    

class FourierNet(MFNBase):
    """
    Taken from https://github.com/boschresearch/multiplicative-filter-networks
    """
    def __init__(self, grid_dim, hidden_feat_dim, code_dim, out_dim, n_layers=3, input_scale=256.0, **kwargs):
        super().__init__(grid_dim, hidden_feat_dim, code_dim, out_dim, n_layers)

        self.filters = nn.ModuleList(
                [FourierLayer(grid_dim, hidden_feat_dim, input_scale / np.sqrt(n_layers + 1)) for _ in range(n_layers + 1)])

    def get_filters_weight(self):
        weights = list()
        for ftr in self.filters:
            weights.append(ftr.weight)
        return torch.cat(weights)
    

#####################################################################
####################### Self-Attention Module #######################
#####################################################################
from MERLIN.attention import LinearAttention

class TransformerCatNoCls(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,
                 scale=16,     # can be list, or an int
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 attention_init='orthogonal',
                 init_gain=None,
                 use_relu=False,
                 cat_pos=False,
                 pos_dim=2,
                 attention_norm='instance',    # 'layer' | 'instance'
                 pre_norm_residual=True,
                 ):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth
        assert len(scale) == depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln
        self.pos_dim = pos_dim
        self.pre_norm_residual = pre_norm_residual

        if attn_type == 'standard':
            raise NotImplementedError
        else:
            for d in range(depth):
                if scale[d] != -1 or cat_pos:
                    attn_module = LinearAttention(dim, attn_type,
                                                   heads=heads, dim_head=dim_head, dropout=dropout,
                                                   relative_emb=True, scale=scale[d],
                                                   relative_emb_dim=self.pos_dim,
                                                   min_freq=min_freq,
                                                   init_method=attention_init,
                                                   init_gain=init_gain,
                                                   norm_type=attention_norm
                                                   )
                else:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  cat_pos=True,
                                                  pos_dim=relative_emb_dim,
                                                  relative_emb=False,
                                                  init_method=attention_init,
                                                  init_gain=init_gain,
                                                  norm_type=attention_norm
                                                  )
                if not use_ln:
                    self.layers.append(
                        nn.ModuleList([
                                        attn_module,
                                        FeedForward(dim, mlp_dim, dropout=dropout)
                                        if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout)
                        ]),
                        )
                else:
                    self.layers.append(
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module,
                            nn.LayerNorm(dim),
                            FeedForward(dim, mlp_dim, dropout=dropout)
                            if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout),
                        ]),
                    )

    def forward(self, x: torch.Tensor, pos_embedding: torch.Tensor) -> torch.Tensor:
        # x in [b n c], pos_embedding in [b n 2]
        b, n, c = x.shape

        for layer_no, attn_layer in enumerate(self.layers):
            if not self.use_ln:
                [attn, ffn] = attn_layer

                x = attn(x, pos_embedding) + x
                x = ffn(x) + x
            else:
                [ln1, attn, ln2, ffn] = attn_layer
                if self.pre_norm_residual:
                    x = x + attn(ln1(x), pos_embedding)
                    x = x + ffn(ln2(x))
                else:
                    x = ln1(x)
                    x = attn(x, pos_embedding) + x
                    x = ln2(x)
                    x = ffn(x) + x
        return x


# code copied from: https://github.com/ndahlquist/pytorch-fourier-feature-networks
# author: Nic Dahlquist
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):
        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


#####################################################################
########################## SET TRANSFORMER ##########################
#####################################################################
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


def _act(name: str):
    name = name.lower()
    return {"gelu": nn.GELU(), "relu": nn.ReLU(inplace=True), "silu": nn.SiLU()}[name]


class TrainablePosEncoder(nn.Module):
    """
    input:  (B, N, 2)  —— B=batch, N=num_point
    output:  (B, N, pos_dim)

    parameters:
        pos_dim: output dimension
        num_frequencies: frequency bands K (generate 2K sin/cos channel)
        gaussian_init_scale: initial scale of the frequency matrix, the larger the higher
        append_xy: if original (x, y) be concatenated to sin/cos before projection
        dropout
        learnable_proj: if the frequency matrix trainable (default trainable)
    """
    def __init__(
        self,
        pos_dim: int,
        num_frequencies: int = 64,
        gaussian_init_scale: float = 10.0,
        append_xy: bool = False,
        dropout: float = 0.0,
        learnable_proj: bool = True,
    ):
        super().__init__()
        self.pos_dim = pos_dim
        self.num_frequencies = num_frequencies
        self.append_xy = append_xy

        # frequency matrix B ∈ R^{K×2}
        B = gaussian_init_scale * torch.randn(num_frequencies, 2)
        if learnable_proj:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

        self.phase = nn.Parameter(torch.zeros(num_frequencies))
        self.gain  = nn.Parameter(torch.ones(num_frequencies))

        raw_dim = 2 * num_frequencies + (2 if append_xy else 0)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(raw_dim, pos_dim)

        self.reset_parameters()

    def reset_parameters(self):
        # linear layer
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        xy: (B, N, 2)
        return: (B, N, pos_dim)
        """
        assert xy.dim() == 3 and xy.size(-1) == 2, "Input must be (B, N, 2)."

        # (B, N, 2) @ (2, K) -> (B, N, K)
        z = torch.matmul(xy, self.B.transpose(0, 1))
        # Add phase & band gain to automatically broadcast to (B, N, K)
        z = z + self.phase
        g = self.gain

        s = torch.sin(z) * g      # (B, N, K)
        c = torch.cos(z) * g      # (B, N, K)

        feats = torch.cat([s, c], dim=-1)  # (B, N, 2K)
        if self.append_xy:
            feats = torch.cat([feats, xy], dim=-1)  # (B, N, 2K+2)

        feats = self.dropout(feats)
        out = self.proj(feats)  # (B, N, pos_dim)
        return out


class FourierPosEncoder(nn.Module):
    def __init__(self, num_bands, max_freq=16.0, logspace=True, include_input=False):
        super().__init__()
        self.num_bands = num_bands
        self.include_input = include_input

        if logspace:
            bands = torch.logspace(0.0, math.log2(max_freq), steps=num_bands, base=2.0)
        else:
            bands = torch.linspace(1.0, max_freq, steps=num_bands)
        bands = 2 * math.pi * bands
        self.register_buffer("bands", bands)

    @property
    def out_dim(self):
        base = 4 * self.num_bands                           # x: sin/cos 2*num_bands, y: sin/cos 2*num_bands
        return base + (2 if self.include_input else 0)

    def forward(self, xy):                                  # xy: [B, K, 2] in [0,1]
        assert xy.size(-1) == 2
        bands = self.bands.to(xy.dtype)                     # [num_bands]
        x = xy[..., 0:1] * bands                            # [B, K, num_bands]
        y = xy[..., 1:2] * bands

        def emb(v):
            return torch.cat([torch.sin(v), torch.cos(v)], dim=-1)
        
        feats = torch.cat([emb(x), emb(y)], dim=-1)         # [B, K, 4*num_bands]
        if self.include_input:
            feats = torch.cat([xy, feats], dim=-1)          # [B, K, 4*num_bands+2]
        return feats


class PreEncoder(nn.Module):
    """
    Two-path precoding: pos branch + val branch → fusion (concat further dimensionality reduction)
    It is used to transform [B,K,pos in] and [B,K,val in=1] to [B,K,out dim], and then send to SetTransformer(dim input=out dim)

    Default configuration: pos in=64 (Fourier), val in=1 (standardized u)
             pos_hidden=64, val_hidden=64, out_dim=128, fusion='concat'
    """
    def __init__(
        self,
        pos_in_dim: int,
        val_in_dim: int = 1,
        pos_hidden: int = 64,
        val_hidden: int = 64,
        out_dim: int = 128,
        fusion: str = "concat",
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layernorm: bool = True,
        post_ln: bool = True,       # Whether to perform LayerNorm on the fused vectors again
    ):
        super().__init__()
        act = _act(activation)

        self.pos = nn.Sequential(
            nn.Linear(pos_in_dim, pos_hidden, bias=True),
            act,
            nn.Dropout(dropout),
            nn.Linear(pos_hidden, pos_hidden, bias=True),
            act,
        )
        self.val = nn.Sequential(
            nn.Linear(val_in_dim, val_hidden, bias=True),
            act,
            nn.Dropout(dropout),
            nn.Linear(val_hidden, val_hidden, bias=True),
            act,
        )
        self.pos_ln = nn.LayerNorm(pos_hidden) if use_layernorm else nn.Identity()
        self.val_ln = nn.LayerNorm(val_hidden) if use_layernorm else nn.Identity()

        assert fusion in ("concat", "sum")
        self.fusion = fusion
        if fusion == "concat":
            self.fuse = nn.Sequential(
                nn.Linear(pos_hidden + val_hidden, out_dim, bias=True),
                act,
                nn.Dropout(dropout),
            )
        else:  # 'sum' iff pos_hidden == val_hidden
            assert pos_hidden == val_hidden, "sum fusion assert pos_hidden == val_hidden"
            self.fuse = nn.Identity()
            out_dim = pos_hidden

        self.post_ln = nn.LayerNorm(out_dim) if (post_ln and use_layernorm) else nn.Identity()
        self.out_dim = out_dim

    def forward(self, pos_feat, u):  # pos_feat: [B, K, P], u: [B, K, 1]
        zp = self.pos_ln(self.pos(pos_feat))
        zv = self.val_ln(self.val(u))
        if self.fusion == "concat":
            z = torch.cat([zp, zv], dim=-1)
            z = self.fuse(z)
        else:  # sum
            z = self.fuse(zp + zv)
        z = self.post_ln(z)
        return z  # [B, K, out_dim]


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))
