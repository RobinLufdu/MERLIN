import torch
import torch.nn as nn
from einops import repeat

from MERLIN.network import TransformerCatNoCls, FeedForward
from MERLIN.attention import CrossLinearAttention
from layers.Embedding import unified_pos_embedding
from layers.Basic import MLP


class LatentGlobalEncoder(nn.Module):
    """
    Global latent function encoder based on Galerkin Transformer.

    Pipeline:
      0) coordinate handling:
            pos_emb=False (default): concatenate the runtime input_pos, so sparse / irregular
            point sets are supported directly.
            pos_emb=True: concatenate fixed unified_pos_embedding(shapelist, ref), which assumes
            a full regular domain with N_pt == prod(shapelist).
      1) preprocess: [B, N_pt, input_channels + pos_channels] -> [B, N_pt, in_emb_dim]
      2) spatial Transformer (Galerkin/Fourier linear attention, LayerNorm variant)
         over observed points for `spatial_depth` layers.
      3) create K learned latent tokens (+ optional learned latent positions)
      4) stack `latent_depth` global cross-attention blocks:
            latents = CrossLinearAttention(latents <- points, with RoPE) + residual
            latents = FFN(latents) + residual
      5) project_to_latent: [B, K, in_emb_dim] -> [B, K, token_dim]

    x         : [B, N_pt, input_channels], (delay-stacked) field values only
    input_pos : [B, N_pt, spatial_dim], sparse, irregular, or regular coordinates
    return    : [B, latent_tokens, token_dim]
    """
    def __init__(self,
                 input_channels,
                 in_emb_dim,
                 token_dim,
                 heads,
                 spatial_depth,                  # how many spatial Transformer layers
                 dim_head=None,                  # per-head dim (default: in_emb_dim // heads)
                 mlp_dim=None,                   # FFN hidden dim (default: 2*in_emb_dim)
                 attn_type='galerkin',
                 dropout=0.,
                 # spatial RoPE settings
                 relative_emb_dim=2,
                 min_freq=1/64,
                 scale_spatial=None,             # list len=spatial_depth, e.g. [32,16,8,1]
                 use_ln=True,
                 # latent settings
                 latent_tokens=4,                # K
                 latent_depth=2,                 # how many cross-attn blocks
                 use_latent_ln=True,
                 use_latent_pos=True,            # give latent tokens their own learned positions
                 scale_latent=8.0,               # RoPE scale in cross-attn for latents
                 # optional fixed position embedding for full regular domains
                 pos_emb=False,                  
                 shapelist=None,                 # required only when pos_emb=True
                 ref=8,                          # controls unified_pos_embedding channel count: ref ** spatial_dim
                 activation='gelu',
                 device=None):
        super().__init__()
        assert attn_type in ['galerkin', 'fourier']
        self.spatial_dim = relative_emb_dim
        self.pos_emb = pos_emb
        self.pos_dim = relative_emb_dim  # dim of latent positions
        self.use_latent_pos = use_latent_pos

        if dim_head is None:
            assert in_emb_dim % heads == 0
            dim_head = in_emb_dim // heads
        if mlp_dim is None:
            mlp_dim = in_emb_dim * 2

        # 0) coordinate / position features to concatenate before the point-wise MLP.
        if pos_emb:
            if shapelist is None:
                raise ValueError("shapelist is required when pos_emb=True.")
            pos = unified_pos_embedding(shapelist, ref, device=device)  # [1, prod(shapelist), ref^d]
            self.register_buffer('pos', pos, persistent=False)
            pos_channels = ref ** len(shapelist)
        else:
            pos_channels = self.spatial_dim

        # 1) point-wise preprocessing after concatenating field values and position features.
        self.preprocess = MLP(input_channels + pos_channels, in_emb_dim * 2,
                              in_emb_dim, n_layers=0, res=False, act=activation)

        # 2) spatial transformer over observed points.
        if scale_spatial is None:
            if spatial_depth <= 2:
                scale_spatial = [32, 1][:spatial_depth]
            elif spatial_depth == 3:
                scale_spatial = [32, 16, 1]
            else:
                scale_spatial = [32, 16] + [8] * (spatial_depth - 3) + [1]
        assert len(scale_spatial) == spatial_depth

        self.s_transformer = TransformerCatNoCls(
            dim=in_emb_dim,
            depth=spatial_depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=in_emb_dim * 2,
            attn_type=attn_type,
            use_ln=use_ln,
            scale=scale_spatial,
            dropout=dropout,
            relative_emb_dim=relative_emb_dim,
            min_freq=min_freq,
            attention_init='orthogonal',
            init_gain=None,
            use_relu=False,
            cat_pos=False,
            pos_dim=self.pos_dim,
            attention_norm='layer',
            pre_norm_residual=True,
        )

        # 3) learned global latent tokens (+ optional learned latent positions).
        self.latent_tokens = nn.Parameter(
            torch.randn(latent_tokens, in_emb_dim) * (1 / (in_emb_dim ** 0.5))
        )
        if use_latent_pos:
            self.latent_pos = nn.Parameter(torch.rand(latent_tokens, self.pos_dim))

        # 4) latent cross-attention blocks: K latent tokens attend to all observed points.
        self.lat_blocks = nn.ModuleList([])
        for _ in range(latent_depth):
            cross = CrossLinearAttention(
                dim_q=in_emb_dim, dim_kv=in_emb_dim,
                heads=heads, dim_head=dim_head,
                attn_type=attn_type, dropout=dropout,
                relative_emb=True,                # use RoPE in cross-attn
                relative_emb_dim=relative_emb_dim,
                min_freq=min_freq,
                scale=scale_latent,               # usually a bit smaller than spatial's first layer
                cat_pos=False,
                pos_dim=self.pos_dim,
                norm_type='layer',
            )
            ffn = FeedForward(in_emb_dim, mlp_dim, dropout=dropout)
            if use_latent_ln:
                block = nn.ModuleList([nn.LayerNorm(in_emb_dim), cross,
                                       nn.LayerNorm(in_emb_dim), ffn])
            else:
                block = nn.ModuleList([cross, ffn])
            self.lat_blocks.append(block)

        # 5) projection to decoder latent token dimension.
        self.project_to_latent = nn.Linear(in_emb_dim, token_dim, bias=False)
        self.use_latent_ln = use_latent_ln

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        """
        x         : [B, N_pt, input_channels]  field/delay values only; coordinates are not pre-concatenated
        input_pos : [B, N_pt, pos_dim]         coordinates for RoPE and, when pos_emb=False, MLP input
        return    : [B, K, token_dim]          K = latent_tokens
        """
        B, N, _ = x.shape
        if input_pos.shape[-1] != self.pos_dim:
            raise ValueError(f"Expected input_pos dim {self.pos_dim}, got {input_pos.shape[-1]}.")

        # 0) concatenate coordinate features.
        if self.pos_emb:
            if self.pos.shape[1] != N:
                raise ValueError(
                    f"pos_emb=True requires N={self.pos.shape[1]} points from shapelist, got N={N}."
                )
            concat_pos = self.pos.expand(B, -1, -1)
        else:
            concat_pos = input_pos
        x = torch.cat((x, concat_pos), dim=-1)
        x = self.preprocess(x)                                      # [B, N_pt, in_emb_dim]

        # 1) spatial self-attention over observed points.
        x = self.s_transformer(x, input_pos)                        # [B, N_pt, in_emb_dim]

        # 2) prepare learned latent tokens and their learned positions.
        lat = repeat(self.latent_tokens, 't c -> b t c', b=B)       # [B, K, in_emb_dim]
        if self.use_latent_pos:
            latent_pos = repeat(self.latent_pos, 't c -> b t c', b=B)  # [B, K, pos_dim]
        else:
            latent_pos = torch.zeros(B, self.latent_tokens.shape[0], self.pos_dim,
                                     device=x.device, dtype=x.dtype)

        # 3) latent cross-attention stack with pre-LN residuals.
        for blk in self.lat_blocks:
            if self.use_latent_ln:
                ln1, cross, ln2, ffn = blk
                lat = lat + cross(ln1(lat), x, pos_q=latent_pos, pos_kv=input_pos)
                lat = lat + ffn(ln2(lat))
            else:
                cross, ffn = blk
                lat = lat + cross(lat, x, pos_q=latent_pos, pos_kv=input_pos)
                lat = lat + ffn(lat)

        # 4) project to decoder latent dimension.
        lat = self.project_to_latent(lat)                           # [B, K, token_dim]
        return lat


################################ Set Transformer based Encoder ################################
from MERLIN.network import TrainablePosEncoder, FourierPosEncoder, PreEncoder, SetTransformer

class SetEncoder2D(nn.Module):
    def __init__(self,
                 input_channels: int,
                 pos_emb_dim: int,
                 pos_emb_type: str = "trainable",
                 pos_hidden: int = 256,
                 val_hidden: int = 128,
                 set_dim: int = 128, 
                 set_hidden: int = 128,
                 num_heads: int = 4,
                 num_inds: int = 64, 
                 token_dim: int = 64,
                 latent_tokens: int = 4,                # K
                 use_ln: bool = True,
                 fourier_max_freq: float = 16.0,
                 dropout: float = 0.1,
                 ):
        super().__init__()
        assert pos_emb_type in ["trainable", "fourier"]
        # (B, N_PT, 2) -> (B, N_PT, POS_EMB_DIM)
        if pos_emb_type == "trainable":
            self.pos_encoder = TrainablePosEncoder(pos_dim=pos_emb_dim, num_frequencies=64)
        elif pos_emb_type == "fourier":
            self.pos_encoder = FourierPosEncoder(num_bands=pos_emb_dim//4, max_freq=fourier_max_freq)
        self.pre_encoder = PreEncoder(pos_in_dim=pos_emb_dim, val_in_dim=input_channels, 
                                      pos_hidden=pos_hidden, val_hidden=val_hidden, out_dim=set_dim, dropout=dropout)
        # (B, N_PT, SET_DIM) -> (B, K, token_dim)
        self.set_encoder = SetTransformer(dim_input=set_dim, num_outputs=latent_tokens, dim_output=token_dim,
                                          num_inds=num_inds, dim_hidden=set_hidden, num_heads=num_heads, ln=use_ln)
    
    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        """
        x         : [B, N_pt, input_channels]  (e.g., T_in*C_in + 2 if you've concatenated coords upstream)
        input_pos : [B, N_pt, pos_dim]         (usually 2D coords)
        return    : [B, K, token_dim] (K: number of tokens; token_dim: dimension of the cls token)
        """
        pos_emb = self.pos_encoder(input_pos)
        set_tokens = self.pre_encoder(pos_emb, x)
        out = self.set_encoder(set_tokens)
        return out


