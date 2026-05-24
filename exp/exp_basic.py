import os, json
import torch
from datetime import datetime
import dataclasses
from typing import Any, Literal

from dataclasses import dataclass
from data.data_process import Dataloader_Configs, PDEDataProcessor
from utilities.utils import create_logger, log_config


@dataclass
class ExpConfigs:
    model_name: str
    epochs: int
    device: torch.device
    out_dir: str 
    seed: int = 0
    
    optimizer: str = "Adam"          # 'AdamW' / 'Adam'
    
    lr: float = 0.005                # 1e-2 for DINO
    lr_adapt: float = 0.01           # learning rate for DINO auto-decoding
    weight_decay: float = 1e-4
    max_grad_norm: float | None = None

    scheduler: str = "OneCycleLR"    # 'OneCycleLR' / 'CosineAnnealingLR' / 'StepLR'
    step_size: int = 100             # step size for StepLR scheduler
    gamma: float = 0.5               # decay parameter for StepLR scheduler
    pct_start: float = 0.3           # 'oncycle lr schedule' 

    teacher_forcing: bool = True
    tf_epsilon: float = 0.99         # weight decaying for teacher-forcing 
    epsilon: float = 0.95            
    tf_epsilon_min: float = 0.10
    update_every: int = 100          # teacher forcing parameter update

    # If provided, we will load (traj_id, t0) lists from this file instead of sampling.
    split_metadata_path: str | None = None

    # --------------------------- Specialize for MERLIN ---------------------------
    loss_with_mask: bool = True
    enc_mode: Literal["galerkin_transformer", "set_transformer"] = "galerkin_transformer"
    rollout_steps: int = 5
    gamma_decay: float = 0.8

    use_bias: bool = True            # bias term for phase I latent dynamics
    dt_eval: float = 0.25

    lambda_dyn: float = 1.0
    lambda_pred: float = 1.0
    lambda_corr: float | None = None 
    lambda_spectral: float | None = None
    lambda_lt_pred: float | None = None
    lambda_residual: float | None = None

    ms_consistency_enable: bool = False         # latent multi-step consistency with batch A*
    freq_ms_enable: bool = False                # turn on spectral+multiscale penalty
    lambda_freq: float | None = None
    freq_hf_power: float = 0.0
    ms_pool_scales: tuple[int, ...] = (2, 4)

    diag_every: int | None = 50

    lr_dyn_mem: float = 5e-3
    lr_dyn_lin: float = 1e-4
    lr_encdec: float = 0.0
    lr_dec: float = 0.0

    use_diag_whiten: bool = False
    global_A_mode: str = "ema"

    # Phase-I linear backbone fitting strategy
    phase1_linear_mode: Literal["ridge", "joint_gd"] = "ridge"
    lr_phase1_linear: float | None = None
    wd_phase1_linear: float = 0.0


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


class Exp_Basic(object):
    def __init__(self, args, exp_cfg: ExpConfigs, model_cfg, data_processor: PDEDataProcessor):
        self.args = args
        self.model_cfg = model_cfg

        self.cfg = exp_cfg
        self.lr = exp_cfg.lr
        self.device = exp_cfg.device

        self.data_processor = data_processor
        self.pos_feat = data_processor.x_grid    # [(*spatial_dims), d]
        self.spatial_dim = self.pos_feat.shape[-1]
        self.shapelist = data_processor.spatial_dims    # tuple of ints
        self.dataset = data_processor.dataset
        # dataloaders, None initialization
        self.n_frames_train = data_processor.n_frames_train
        self.n_frames_cond = data_processor.n_frames_cond
        self.mask_ratio = data_processor.mask_ratio    # mask_ratio=0 <-> without mask
        self.train_loader = self.train_eval_loader = self.test_loader = None

        # If a split metadata file is provided, load frozen sample indices from it.
        # Otherwise, fall back to deterministic sampling via data_processor.compute_samples(...).
        if self.cfg.split_metadata_path:
            p = self.cfg.split_metadata_path
            self.train_sample_idx      = self.load_samples_from_json(p, "train")
            self.train_eval_sample_idx = self.load_samples_from_json(p, "train_eval")
            self.test_sample_idx       = self.load_samples_from_json(p, "test")
        else:
            self.train_sample_idx      = data_processor.compute_samples("train")
            self.train_eval_sample_idx = data_processor.compute_samples("train_eval")
            self.test_sample_idx       = data_processor.compute_samples("test")
        self.n_seqs_tr       = len(self.train_sample_idx)
        self.n_seqs_tr_eval  = len(self.train_eval_sample_idx)
        self.n_seqs_ts       = len(self.test_sample_idx)

        
    def build_dataloader(self):
        pass
    

    def init_optim(self):
        pass


    def _save_split_and_samples(self):
        """
        Gather per-split (traj_id, t0) sample indices and save them together
        with trajectory IDs into a single JSON via the base-class helper.
        """
        sample_indices = {
            "train": self.train_sample_idx,
            "train_eval": self.train_eval_sample_idx,
            "test": self.test_sample_idx,
        }
        # Delegates to Exp_Basic.save_split_metadata
        path = self.save_split_metadata(sample_indices=sample_indices, filename="split_metadata.json")
        if hasattr(self, "logger") and self.logger is not None:
            self.logger.info(f"Saved split metadata and sample indices to: {path}")


    def load_samples_from_json(self, saved_json_path: str, group: str) -> list[tuple[int, int]]:
        """
        Load the frozen (traj_id, t0) list for a given split from a split_metadata.json file.

        Args:
            saved_json_path: Absolute (or relative) path to the split_metadata.json.
            group: One of {"train", "train_eval", "test"}.

        Returns:
            A list of (traj_id, t0) pairs (as Python ints), preserving the saved order.

        Raises:
            ValueError: If the JSON file does not contain the expected "samples[group]" section.
        """
        with open(saved_json_path, "r") as f:
            payload = json.load(f)

        if "samples" not in payload or group not in payload["samples"]:
            raise ValueError(f"'samples[\"{group}\"]' not found in {saved_json_path}")

        items = payload["samples"][group]
        samples = [(int(tid), int(t0)) for tid, t0 in items]

        # Optional sanity check: dataset name must match (warn only)
        ds_in_file = payload.get("dataset", None)
        if ds_in_file is not None and ds_in_file != self.dataset:
            if hasattr(self, "logger") and self.logger:
                self.logger.warning(
                    f"[warn] dataset mismatch: file='{ds_in_file}' vs current='{self.dataset}'"
                )
        # You could also validate window_cfg here if you want hard guarantees.
        return samples


    def setup_logger(self):
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        out_dir = os.path.join(self.cfg.out_dir, f"{run_id}")
        os.makedirs(out_dir, exist_ok=True)
        # self.logger = create_logger(out_dir, os.path.join(out_dir, "log"))
        # Pass a UNIQUE logger name (run_id makes it unique)
        self.logger = create_logger(
            out_dir, os.path.join(out_dir, "log"),
            name=f"exp.{self.dataset}.{run_id}",
            mode="w",          # start fresh each run
            rotating=False,    # set True if you want rotation
            fmt="%(message)s"
        )

        self.logger.info(f"Run ID: {run_id}")
        self.logger.info("========== DATA CONFIGS ==========")
        log_config(cfg=self.data_processor.cfg, logger=self.logger)
        # save_traj_ids(cfg=self.data_processor.cfg, out_dir=out_dir, logger=self.logger)
        self.logger.info("==================================")
        self.logger.info("========== MODEL CONFIGS =========")
        log_config(cfg=self.model_cfg, logger=self.logger)
        self.logger.info("==================================")
        self.logger.info("=========== EXP CONFIGS ==========")
        log_config(cfg=self.cfg, logger=self.logger)
        self.logger.info("==================================")


    def train(self):
        pass


    def evaluate(self):
        pass


    def save_model(self):
        pass


    def _collect_split_info(self) -> dict:
        """
        Build a dictionary that contains dataset split information which is
        always available from the data processor, independent of any loader.
        """
        dp = self.data_processor
        cfg = dp.cfg
        info = {
            "run_id": getattr(self, "run_id", None),
            "dataset": self.dataset,
            "spatial_dims": list(getattr(dp, "spatial_dims", ())),
            "out_channels": int(getattr(dp, "channels", -1)),
            "total_trajs": int(getattr(dp, "total_trajs", -1)),
            "total_len": int(getattr(dp, "total_len", -1)),
            "train_traj_ids": list(getattr(dp, "train_traj_ids", [])),
            "test_traj_ids": list(getattr(dp, "test_traj_ids", [])),
            # Helpful training-time window sizes (for reproducibility/introspection)
            "window_cfg": {
                "n_frames_train": int(cfg.n_frames_train),
                "n_frames_out": int(cfg.n_frames_out),
                "n_frames_cond": int(cfg.n_frames_cond),
                "sample_strategy": str(cfg.sample_strategy),
                "mode": str(cfg.mode),
                "dt_eval": float(cfg.dt_eval),
            },
            "seed": int(getattr(cfg, "seed", -1)),
        }
        return info


    def save_split_metadata(
        self,
        sample_indices: dict | None = None,
        filename: str = "split_metadata.json"
    ) -> str:
        """
        Save a *single* JSON that unifies trajectory IDs and (optional) per-split sample indices.

        Args:
            sample_indices: optional mapping from split name to a list of (traj_id, t0) pairs.
                            Example:
                              {
                                "train": [(0, 5), (3, 10), ...],
                                "train_eval": [...],
                                "test": [...]
                              }
                            If None, only trajectory IDs (train/test) and split info are saved.
            filename: output JSON file name.

        Returns:
            Absolute path to the saved JSON file.
        """
        out_dir = os.path.join(self.cfg.out_dir, f"{self.run_id}")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)

        payload = self._collect_split_info()
        # Normalize tuples -> lists for JSON
        def _normalize_samples(v):
            if v is None:
                return None
            out = []
            for item in v:
                # item could be (traj_id, t0) or list-like
                out.append([int(item[0]), int(item[1])])
            return out

        if sample_indices is not None:
            payload["samples"] = {}
            for k, v in sample_indices.items():
                payload["samples"][k] = _normalize_samples(v)

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path
    

    @staticmethod
    def _jsonify_config(obj: Any) -> dict:
        """
        Convert a config-like object to a JSON-serializable dict.
        Handles:
          - dataclasses / dict / objects with __dict__
          - torch.device / torch.dtype / non-serializable values -> string
        """
        def _to_jsonable(x):
            if dataclasses.is_dataclass(x):
                x = dataclasses.asdict(x)
            elif hasattr(x, "__dict__") and not isinstance(x, dict):
                # Convert object's attributes but drop callables
                x = {k: v for k, v in vars(x).items() if not callable(v)}

            if isinstance(x, dict):
                out = {}
                for k, v in x.items():
                    out[k] = _to_jsonable(v)
                return out
            elif isinstance(x, (list, tuple)):
                return [_to_jsonable(v) for v in x]
            elif isinstance(x, (int, float, str, bool)) or x is None:
                return x
            else:
                # Fallback for torch/device/dtype or any exotic types
                if isinstance(x, torch.device) or isinstance(x, torch.dtype):
                    return str(x)
                return str(x)

        return _to_jsonable(obj)


    def save_repro_artifacts(self) -> None:
        """
        Save artifacts needed for full experiment reproducibility:
          1) Normalizer state (if available) to configs/normalizer.pt
          2) JSON snapshots of model_cfg and exp_cfg to configs/*.json

        Precondition: setup_logger() has been called so self.run_id/out_dir exist.
        """
        out_dir = os.path.join(self.cfg.out_dir, f"{self.run_id}")
        cfg_dir = os.path.join(out_dir, "configs")
        os.makedirs(cfg_dir, exist_ok=True)

        # 1) Save normalizer state (prefer state_dict, fallback to common attributes)
        norm = getattr(self.data_processor, "normalizer", None)
        if norm is not None:
            norm_state = None
            if hasattr(norm, "state_dict") and callable(getattr(norm, "state_dict")):
                try:
                    norm_state = norm.state_dict()
                except Exception:
                    norm_state = None

            if norm_state is None:
                # Fallback: gather common fields like mean/std/eps/min/max if present
                norm_state = {"__class__": type(norm).__name__}
                for name in ("mean", "std", "eps", "min", "max"):
                    if hasattr(norm, name):
                        val = getattr(norm, name)
                        try:
                            if torch.is_tensor(val):
                                norm_state[name] = val.detach().cpu()
                            else:
                                norm_state[name] = val
                        except Exception:
                            norm_state[name] = str(val)

            torch.save(norm_state, os.path.join(cfg_dir, "normalizer.pt"))

        # 2) Save config snapshots
        try:
            with open(os.path.join(cfg_dir, "model_cfg.json"), "w") as f:
                json.dump(self._jsonify_config(self.model_cfg), f, indent=2)
        except Exception as e:
            print(f"[warn] failed to save model_cfg.json: {e}")

        try:
            with open(os.path.join(cfg_dir, "exp_cfg.json"), "w") as f:
                json.dump(self._jsonify_config(self.cfg), f, indent=2)
        except Exception as e:
            print(f"[warn] failed to save exp_cfg.json: {e}")

        try:
            with open(os.path.join(cfg_dir, "dataloader_cfg.json"), "w") as f:
                json.dump(self._jsonify_config(self.data_processor.cfg), f, indent=2)
        except Exception as e:
            print(f"[warn] failed to save dataloader_cfg.json: {e}")

    
    @staticmethod
    def _load_cfg_json(run_dir: str, name: str) -> dict | None:
        """
        name ∈ {"exp_cfg", "model_cfg", "dataloader_cfg"}
        """
        path = os.path.join(run_dir, "configs", f"{name}.json")
        if not os.path.isfile(path):
            return None
        with open(path, "r") as f:
            return json.load(f)


    @staticmethod
    def load_all_configs(run_dir: str) -> dict:
        return {
            "exp_cfg": Exp_Basic._load_cfg_json(run_dir, "exp_cfg"),
            "model_cfg": Exp_Basic._load_cfg_json(run_dir, "model_cfg"),
            "dataloader_cfg": Exp_Basic._load_cfg_json(run_dir, "dataloader_cfg"),
        }


    @staticmethod
    def build_dataloader_cfg_from_json(d: dict) -> Dataloader_Configs:
        if d is None:
            raise ValueError("dataloader_cfg dict is None")
        dd = dict(d)
        if "block_size" in dd and isinstance(dd["block_size"], list):
            dd["block_size"] = tuple(dd["block_size"])
        return Dataloader_Configs(**dd)


    @staticmethod
    def rebuild_processor_from_artifacts(*, run_dir: str, data_tensor: torch.Tensor) -> PDEDataProcessor:
        cfgs = Exp_Basic.load_all_configs(run_dir)
        dcfg_dict = cfgs.get("dataloader_cfg", None)
        if dcfg_dict is None:
            raise FileNotFoundError(f"configs/dataloader_cfg.json not found under {run_dir}")
        dcfg = Exp_Basic.build_dataloader_cfg_from_json(dcfg_dict)
        return PDEDataProcessor(data_tensor=data_tensor, cfg=dcfg)
