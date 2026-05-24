import argparse

from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any, Optional, Tuple
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


######################################################################################################################
######################################################################################################################
"""
DATASET config
"""

@dataclass(frozen=True)
class DatasetCfg:
    STATE_DIM: int
    SPATIAL_DIM: int
    GRID_DIM: int
    INPUT_SCALE: int
    SHAPELIST: tuple[int, int]
    DATA_PATH: str


_CFG_TABLE = {
    "ns_1e-3": DatasetCfg(
        STATE_DIM=1, SPATIAL_DIM=2, GRID_DIM=2, INPUT_SCALE=64, SHAPELIST=(64, 64),
        DATA_PATH="./data/ns_V1e-3_N5000_T50.mat"
    ),
    "wave": DatasetCfg(
        STATE_DIM=2, SPATIAL_DIM=2, GRID_DIM=2, INPUT_SCALE=128, SHAPELIST=(64, 64),
        DATA_PATH="./data/wave.h5"
    ),
    "sst": DatasetCfg(
        STATE_DIM=1, SPATIAL_DIM=2, GRID_DIM=2, INPUT_SCALE=128, SHAPELIST=(64, 64),
        DATA_PATH="./data/sst_T20_N1000.pt"
    ),
    "era5": DatasetCfg(
        STATE_DIM=1, SPATIAL_DIM=2, GRID_DIM=2, INPUT_SCALE=180, SHAPELIST=(180, 360),
        DATA_PATH="./data/ERA5_N550_T20.npz"
    )
}


def get_dataset_cfg(name: str) -> DatasetCfg:
    """Return immutable config for a given dataset name."""
    try:
        return _CFG_TABLE[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Available: {list(_CFG_TABLE.keys())}")
    

######################################################################################################################
######################################################################################################################
"""
MODEL config
"""
from MERLIN.encoder import LatentGlobalEncoder, SetEncoder2D
from MERLIN.decoder import FourierDecoder
from MERLIN.latent_discrete import LatentProcessDiscrete

# —— submodule dataclass ——
@dataclass(slots=True)
class EncoderParams:
    input_channels: int
    shapelist: Tuple[int, ...] | None = None
    pos_emb: bool = False
    ref: int = 8
    activation: str = "gelu"
    device: torch.device = torch.device("cpu")
    in_emb_dim: int = 128
    token_dim: int = 64
    heads: int = 4
    spatial_depth: int = 3
    dim_head: int | None = 32
    mlp_dim: int | None = None
    attn_type: Literal["galerkin", "fourier"] = "galerkin"
    dropout: float = 0.0
    relative_emb_dim: int = 2
    min_freq: float = 1/64
    scale_spatial: int | None = None
    use_ln: bool = True
    latent_tokens: int = 4
    latent_depth: int = 2
    use_latent_ln: bool = True
    use_latent_pos: bool = True
    scale_latent: int = 8
    def to_params(self) -> Dict:
        return asdict(self)
    

@dataclass(slots=True)
class SetEncoderParams:
    input_channels: int
    pos_emb_dim: int
    pos_emb_type: str = "trainable"
    pos_hidden: int = 256
    val_hidden: int = 128
    set_dim: int = 128
    set_hidden: int = 128
    num_heads: int = 4
    num_inds: int = 64 
    token_dim: int = 64
    latent_tokens: int = 4                # K
    use_ln: bool = True
    fourier_max_freq: float = 16.0
    dropout: float = 0.1
    def to_params(self) -> Dict:
        return asdict(self)


@dataclass(slots=True)
class FourierDecoderParams:
    grid_dim: int = 2
    fourier_hidden_dim: int = 256
    latent_dim: int = 128
    out_dim : int = 1
    n_fourier_layers: int = 3
    input_scale: float = 64
    modmlp_layers: int = 2
    modmlp_act: str = "gelu"
    def to_params(self) -> Dict:
        return asdict(self)


@dataclass(slots=True)
class LatentProcessDiscreteParams:
    state_dim: int
    code_dim: int
    memory_dim: Optional[int] = None
    memory_type: str = "leaky"          # {'leaky','gru','lstm','residual'}
    # dec/enc MLPs for leaky backend
    enc_hidden_dim: int = 128
    enc_layers: int = 2
    dec_hidden_dim: int = 128
    dec_layers: int = 2
    nl: str = "swish"
    # RNN config (for gru/lstm)
    rnn_layers: int = 2 
    rnn_dropout: float = 0.0 
    # gate
    gate_per_dim: bool = True
    # init options
    init_tau_steps: Optional[float] = None    # for leaky: initialize gamma from tau,
    use_layer_norm: bool = True
    # ---- residual LSTM window configs ----
    context_window: int = 3
    window_pad: str = "repeat"                # "repeat" / "zero"
    augment: bool = False                    
    augment_variant: str = "history"          # "history" / "current"
    rnn_hidden: int = 256
    def to_params(self) -> Dict:
        return asdict(self)



# —— Model Factory for MERLIN ——
@dataclass(slots=True)
class MERLINParamBundle:
    encoder: EncoderParams
    set_encoder: SetEncoderParams
    fourier_decoder: FourierDecoderParams
    latent_process_discrete: LatentProcessDiscreteParams
    
    _state_dim: int
    _latent_dim: int
    _code_dim: int
    _n_frames_cond: int 
    _input_channels: int

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def code_dim(self) -> int:
        return self._code_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def n_frames_cond(self) -> int:          
        return self._n_frames_cond
    
    @property
    def input_channels(self) -> int:
        return self._input_channels
    
    def as_model_kwargs(self, include_meta: bool = True) -> Dict[str, Any]:
        """Return per-module kwargs; optionally include top-level meta fields."""
        out = {
            "encoder": self.encoder.to_params(),
            "set_encoder": self.set_encoder.to_params(),
            "fourier_decoder": self.fourier_decoder.to_params(),
            "latent_process_discrete": self.latent_process_discrete.to_params(),
        }
        if include_meta:
            out.update({
                "state_dim": self.state_dim,
                "latent_dim": self.latent_dim,
                "code_dim": self.code_dim,
                "n_frames_cond": self.n_frames_cond,
                "input_channels": self.input_channels,
            })
        return out

    @staticmethod
    def from_args(
        *,
        dataset_cfg: DatasetCfg,
        n_frames_cond: int,    # number of conditional frames
        # ------------------------------------------------
        fourier_hidden_dim: int = 64,
        n_fourier_layers: int = 3,
        input_scale: Optional[int] = None,  # None -> use dataset_cfg.INPUT_SCALE
        # ------------------------------------------------
        shapelist: Tuple[int, ...] = (64, 64),
        pos_emb: bool = False,
        ref: int = 8,
        activation: str = "gelu",
        device: torch.device = "cpu",
        # ------------------------------------------------
        in_emb_dim: int = 128,
        token_dim: int = 64,
        enc_heads: int = 4,
        spatial_depth: int = 3,
        dim_head: int | None = 32,
        min_freq: float = 1/64,
        latent_tokens: int = 4,
        latent_depth: int = 2,
        # ------------------------------------------------
        pos_emb_dim: int = 64,
        pos_emb_type: str = "trainable",
        pos_hidden: int = 128,
        val_hidden: int = 128,
        set_dim: int = 128,
        set_hidden: int = 128,
        num_inds: int = 64,
        use_ln: bool = True,
        fourier_max_freq: float = 16.0,
        dropout: float = 0.1,
        # ------------------------------------------------
        modmlp_layers: int = 2,
        modmlp_act: str = "gelu",
        # ------------------------------------------------
        memory_dim: int | None = None,
        memory_enc_hidden_dim: int | None = None,
        memory_dec_hidden_dim: int | None = None,
        memory_enc_layers: int = 2,
        memory_dec_layers: int = 2,
        memory_nl: str = "swish",
        # ------------------------------------------------
        memory_type: str = "leaky",           # {'leaky','gru','lstm','residual'}
        rnn_layers: int = 2,
        rnn_dropout: float = 0.0,
        # gate
        gate_per_dim: bool = True,
        # init options
        init_tau_steps: Optional[float] = None,  # for leaky: initialize gamma from tau,
        latent_ln: bool = True,
        context_window: int = 4,                    # d
        window_pad: str = "repeat",                 # {'repeat','zero'} padding for the initial window
        augment: bool = False,                      # channel-wise augmentation
        augment_variant: str = "history",           # {'history','current'}
        rnn_hidden: int = 256,
    ) -> "MERLINParamBundle":
        
        STATE_DIM = dataset_cfg.STATE_DIM
        SPATIAL_DIM = dataset_cfg.SPATIAL_DIM
        GRID_DIM = dataset_cfg.GRID_DIM
        INPUT_SCALE = dataset_cfg.INPUT_SCALE if input_scale is None else int(input_scale)

        input_channels = int(n_frames_cond * STATE_DIM)
        latent_dim = int(token_dim * latent_tokens)
        assert latent_dim % STATE_DIM == 0
        code_dim = latent_dim // STATE_DIM

        encoder = EncoderParams(
            input_channels=input_channels,
            shapelist=shapelist,
            pos_emb=pos_emb,
            ref=ref,
            activation=activation,
            device=device,
            in_emb_dim=in_emb_dim,
            token_dim=token_dim,
            heads=enc_heads,
            spatial_depth=spatial_depth,
            dim_head=dim_head,
            min_freq=min_freq,
            latent_tokens=latent_tokens,
            latent_depth=latent_depth,
            relative_emb_dim=SPATIAL_DIM,
        )

        set_encoder = SetEncoderParams(
            input_channels=input_channels,
            pos_emb_dim=pos_emb_dim,
            pos_emb_type=pos_emb_type,
            pos_hidden=pos_hidden,
            val_hidden=val_hidden,
            set_dim=set_dim,
            set_hidden=set_hidden,
            num_heads=enc_heads,
            num_inds=num_inds,
            token_dim=token_dim,
            latent_tokens=latent_tokens,
            use_ln=use_ln, 
            fourier_max_freq=fourier_max_freq,
            dropout=dropout
        )

        fourier_decoder = FourierDecoderParams(
            grid_dim=GRID_DIM,
            fourier_hidden_dim=fourier_hidden_dim,
            latent_dim=latent_dim,
            out_dim=STATE_DIM,
            n_fourier_layers=n_fourier_layers,
            input_scale=INPUT_SCALE,
            modmlp_layers=modmlp_layers,
            modmlp_act=modmlp_act
        )

        latent_process_discrete = LatentProcessDiscreteParams(
            state_dim=STATE_DIM, 
            code_dim=int(code_dim),
            memory_dim=memory_dim,
            memory_type=memory_type, 
            enc_hidden_dim=memory_enc_hidden_dim,
            enc_layers=memory_enc_layers,
            dec_hidden_dim=memory_dec_hidden_dim,
            dec_layers=memory_dec_layers,
            nl=memory_nl,
            rnn_layers=rnn_layers,
            rnn_dropout=rnn_dropout,
            gate_per_dim=gate_per_dim,
            init_tau_steps=init_tau_steps,
            use_layer_norm=latent_ln,
            context_window=context_window,
            window_pad=window_pad,
            augment=augment, augment_variant=augment_variant, 
            rnn_hidden=rnn_hidden
        )
        
        return MERLINParamBundle(
            encoder=encoder,
            set_encoder=set_encoder,
            fourier_decoder=fourier_decoder,
            latent_process_discrete=latent_process_discrete,
            _state_dim = int(STATE_DIM),
            _latent_dim=int(latent_dim),
            _code_dim=int(code_dim),
            _n_frames_cond=int(n_frames_cond),
            _input_channels=int(input_channels),
        )


def make_config(**kw) -> MERLINParamBundle:
    name = kw.get("model_name")    # "MERLIN"
    if name == "MERLIN":
        dataset = kw.pop("dataset")                 
        n_frames_cond = kw.pop("n_frames_cond")  

        dataset_cfg = get_dataset_cfg(name=dataset)
        allowed = {
            "fourier_hidden_dim", "n_fourier_layers", "input_scale",
            "in_emb_dim", "token_dim", "enc_heads", "spatial_depth", "dim_head", "min_freq", "latent_tokens", "latent_depth",

            "shapelist", "pos_emb", "ref", "device", "activation",

            "pos_emb_dim", "pos_emb_type", "pos_hidden", "val_hidden", "set_dim", "set_hidden", "num_inds",
            "use_ln", "dropout", "fourier_max_freq",

            "modmlp_layers", "modmlp_act",

            "memory_dim",
            "memory_enc_hidden_dim", "memory_dec_hidden_dim", "memory_enc_layers", "memory_dec_layers", "memory_nl",

            "memory_type", "rnn_layers", "rnn_dropout",
            "gate_per_dim", "init_tau_steps", "latent_ln",
            "context_window", "window_pad", "augment", "augment_variant", "rnn_hidden"
        }
        overrides = {k: kw[k] for k in list(kw.keys()) if k in allowed}
        return MERLINParamBundle.from_args(
            dataset_cfg=dataset_cfg,
            n_frames_cond=n_frames_cond,
            **overrides
        )
    else:
        raise ValueError(f"Unknown model_name: {name}")


def _as_model_kwargs(model_cfg):
    if hasattr(model_cfg, "to_params"):
        return model_cfg.to_params()
    return dict(model_cfg)


def build_encoder(model_cfg: EncoderParams | SetEncoderParams | dict, enc_mode: str | None = None):
    cfg = _as_model_kwargs(model_cfg)
    mode = enc_mode.lower() if enc_mode is not None else None

    if mode in {"set_transformer", "set_encoder", "set"}:
        return SetEncoder2D(**cfg)
    if mode in {"galerkin_transformer", "global_encoder", "global", "transformer"}:
        return LatentGlobalEncoder(**cfg)

    if "pos_emb_dim" in cfg or "num_inds" in cfg:
        return SetEncoder2D(**cfg)
    return LatentGlobalEncoder(**cfg)


def build_decoder(model_cfg: FourierDecoderParams | dict):
    return FourierDecoder(**_as_model_kwargs(model_cfg))


def build_latent(model_cfg: LatentProcessDiscreteParams | dict):
    return LatentProcessDiscrete(**_as_model_kwargs(model_cfg))
