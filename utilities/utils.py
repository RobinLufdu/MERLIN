import torch, random, numpy as np
import os, json, sys
import logging
from logging.handlers import RotatingFileHandler
from logging import FileHandler, StreamHandler

from dataclasses import asdict, fields, is_dataclass
from pprint import pformat

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch_gen = torch.Generator()
    torch_gen.manual_seed(seed)
    np_gen = np.random.default_rng(seed)
    return torch_gen, np_gen


def make_worker_init_fn(base_seed: int):
    def _init_fn(worker_id: int):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return _init_fn


def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf


def create_logger(folder: str,
                  outfile: str,
                  *,
                  name: str | None = None,
                  level: int = logging.INFO,
                  mode: str = "w",
                  rotating: bool = False,
                  max_bytes: int = 10 * 1024 * 1024,
                  backup_count: int = 5,
                  fmt: str = "%(message)s",            # ← no timestamp/level/name
                  datefmt: str | None = None) -> logging.Logger:
    """
    Create a fresh logger that writes to `outfile` and stdout.
    - Uses a unique name per run (pass `name=`) so handlers don't accumulate.
    - Disables propagation to root.
    - Clears & closes existing handlers (important in notebooks).
    - `fmt` controls the log format; default is message-only.
    """
    os.makedirs(folder, exist_ok=True)
    if name is None:
        base = os.path.basename(os.path.normpath(folder)) or "exp"
        name = f"exp.{base}"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    for h in logger.handlers[:]:
        logger.removeHandler(h)
        try: h.close()
        except Exception: pass

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if rotating:
        fh = RotatingFileHandler(outfile, mode=mode, maxBytes=max_bytes,
                                 backupCount=backup_count, encoding="utf-8")
    else:
        fh = FileHandler(outfile, mode=mode, encoding="utf-8")
    fh.setLevel(level); fh.setFormatter(formatter)

    sh = StreamHandler(sys.stdout)
    sh.setLevel(level); sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def log_config(cfg, logger, title: str | None = None):
    """
    Robust config logger:
      - dataclass: respects metadata {'log': False} and recurses
      - dict: logs keys/values (recurses)
      - object: logs vars(obj) (recurses)
      - converts torch.device/dtype/tensors to printable forms
    """

    def _to_printable(x):
        # Dataclass -> dict (respect metadata log=False)
        if is_dataclass(x):
            out = {}
            skip = {f.name for f in fields(x) if f.metadata.get("log") is False}
            for f in fields(x):
                if f.name in skip:
                    continue
                out[f.name] = _to_printable(getattr(x, f.name))
            return out

        # Dict -> recurse
        if isinstance(x, dict):
            return {k: _to_printable(v) for k, v in x.items()}

        # Sequence -> recurse
        if isinstance(x, (list, tuple)):
            return [_to_printable(v) for v in x]

        # Torch-specific
        if isinstance(x, torch.device) or isinstance(x, torch.dtype):
            return str(x)
        if isinstance(x, torch.Tensor):
            # Avoid dumping full arrays; show shape/dtype/device
            return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device})"

        # Objects with attributes -> vars (drop callables)
        if hasattr(x, "__dict__") and not isinstance(x, type):
            return {k: _to_printable(v) for k, v in vars(x).items() if not callable(v)}

        # Primitives / fallback
        return x

    payload = _to_printable(cfg)
    name = title or (cfg.__class__.__name__ if not isinstance(cfg, dict) else "ConfigDict")
    logger.info("%s:\n%s", name, pformat(payload, width=100, compact=False))


def save_traj_ids(cfg, out_dir: str, logger):
    ids = {}
    if cfg.train_traj_ids is not None:
        ids["train_traj_ids"] = cfg.train_traj_ids
    if cfg.test_traj_ids is not None:
        ids["test_traj_ids"] = cfg.test_traj_ids
    if ids:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "traj_ids.json")
        with open(path, "w") as f:
            json.dump(ids, f, indent=2)
        logger.info("Saved traj IDs to %s", path)

    



