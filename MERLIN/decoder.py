import numpy as np
import matplotlib
matplotlib.use('agg')
import torch
import torch.nn as nn

import math
from torch import nn
from torch.nn import init


#####################################################################
# Fouriernet-based function decoder
#####################################################################
from MERLIN.network import MLP, FourierLayer

class LatentModulation(nn.Module):
    def __init__(self, pos_feat_dim: int, latent_dim: int, out_feat_dim: int, device=None, dtype=None,
                 *, mlp_layers: int = 2, mlp_act: str = "gelu"):
        """
        pos_in: [B, N_pts, pos_feat_dim]
        latent_feat: [B, latent_dim]
        out: [B, N_pts, out_feat_dim]
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LatentModulation, self).__init__()

        self.pos_feat_dim = pos_feat_dim
        self.latent_dim = latent_dim
        self.out_feat_dim = out_feat_dim

        self.A = nn.Parameter(torch.empty(out_feat_dim, pos_feat_dim, **factory_kwargs))
        self.B = nn.Parameter(torch.empty(out_feat_dim, latent_dim, **factory_kwargs))
        self.mlp_modulation = MLP(
            in_dim=latent_dim, hidden_dim=latent_dim*2, out_dim=out_feat_dim, 
            num_layers=mlp_layers, nl=mlp_act, 
            last_bias=False, last_kaiming=False, last_kaiming_a=math.sqrt(5), last_zero_init=True,
            use_layernorm=True, norm_where="pre"
        )
        self.bias = nn.Parameter(torch.empty(out_feat_dim, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.pos_feat_dim)
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.B, a=math.sqrt(5))
        init.uniform_(self.bias, -bound, bound)

    def forward(self, pos_feat: torch.Tensor, latent_feat: torch.Tensor) -> torch.Tensor:
        """
        pos_feat: [B, N_pts, pos_feat_dim]
        latent_feat: [B, latent_dim]
        out: [B, N_pts, out_feat_dim]
        """
        pos_flat = pos_feat.reshape(-1, self.pos_feat_dim)    # [B*N_pts, pos_feat_dim]
        pos_proj = pos_flat @ self.A.t()    # [B*N_pts, out_feat_dim]
        pos_proj = pos_proj.view(pos_feat.shape[0], pos_feat.shape[1], self.out_feat_dim)

        latent_proj = latent_feat @ self.B.t()
        latent_proj = latent_proj.unsqueeze(1).expand(-1, pos_feat.shape[1], -1)
        latent_proj_res = self.mlp_modulation(latent_feat)    # [B, out_feat_dim]
        latent_proj_res = latent_proj_res.unsqueeze(1).expand(-1, pos_feat.shape[1], -1)

        return pos_proj + latent_proj +  latent_proj_res + self.bias.view(1, 1, -1)

                 
class FourierDecoder(nn.Module):
    def __init__(self, grid_dim: int, fourier_hidden_dim: int, latent_dim: int, out_dim: int, n_fourier_layers: int = 3,
                 input_scale: float = 256.0,
                 *, modmlp_layers: int = 2, modmlp_act: str = "gelu"):
        
        super(FourierDecoder, self).__init__()
        self.n_layers = n_fourier_layers
        self.hidden_feat_dim = fourier_hidden_dim
        
        self.mods = nn.ModuleList(
            [LatentModulation(grid_dim, latent_dim, self.hidden_feat_dim, mlp_layers=modmlp_layers, mlp_act=modmlp_act)] +
            [LatentModulation(self.hidden_feat_dim, latent_dim, self.hidden_feat_dim, mlp_layers=modmlp_layers, mlp_act=modmlp_act)
             for _ in range(int(n_fourier_layers))]
        )

        self.out_proj = nn.Linear(self.hidden_feat_dim, out_dim)

        self.filters = nn.ModuleList(
            [FourierLayer(grid_dim, fourier_hidden_dim, input_scale / np.sqrt(n_fourier_layers+1))
             for _ in range(n_fourier_layers+1)]
        )

    def forward(self, grid: torch.Tensor, latent_feat: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
        - grid: [N_pt, grid_dim]
        - latent_feat: [B, latent_dim]
        Outputs:
        - out: [B, N_pts, out_dim]
        """
        bs = latent_feat.shape[0]
        pos_emb0 = self.filters[0](grid)
        if pos_emb0.dim() == 2:
            pos_emb0 = pos_emb0.unsqueeze(0).expand(bs, -1, -1)  # [B, N_pt, hidden_feat_dim]
        pos_feat = torch.zeros(bs, *grid.shape, device=latent_feat.device)
        # print(pos_feat.shape, latent_feat.shape)
        hidden_feat0 = self.mods[0](
            pos_feat=pos_feat, latent_feat=latent_feat
        )
        out = pos_emb0 * hidden_feat0    # [B, N_pt, hidden_feat_dim]

        for i in range(1, self.n_layers + 1):
            pos_embi = self.filters[i](grid)
            if pos_embi.dim() == 2:
                pos_embi = pos_embi.unsqueeze(0).expand(bs, -1, -1)  # [B, N_pt, hidden_feat_dim]
            hidden_feati = self.mods[i](pos_feat=out, latent_feat=latent_feat)
            out = pos_embi * hidden_feati
        
        out = self.out_proj(out)    # [B, N_pt, out_dim]
        return out
        
