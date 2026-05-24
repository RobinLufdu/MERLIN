import torch, os, json, h5py
import numpy as np
from data.data_process import PDEDataProcessor, Dataloader_Configs
from utilities.read_file import MatlabFileReader
from config import get_dataset_cfg


def build_proc_from_run_dir(run_dir: str, dataset: str = "ns_1e-3", args=None) -> PDEDataProcessor:
    """
    Rebuild PDEDataProcessor from a previous run directory.
    Uses configs/dataloader_cfg.json verbatim (including `limit_trajs`).
    """
    cfg_path = os.path.join(run_dir, "configs", "dataloader_cfg.json")
    with open(cfg_path, "r") as f:
        dl_cfg_dict = json.load(f)
    if args is not None and args.phase == "phase2":
        dl_cfg_dict["n_samples_per_traj"] = args.n_samples_per_traj
        dl_cfg_dict["n_frames_cond"] = args.n_frames_cond 
        dl_cfg_dict["n_frames_train"] = args.n_frames_train
        dl_cfg_dict["n_frames_out"] = args.n_frames_out
        dl_cfg_dict["sample_strategy"] = args.sample_strategy
    data_cfg = Dataloader_Configs(**dl_cfg_dict)
    dscfg = get_dataset_cfg(name=data_cfg.dataset)
    if dataset == "ns_1e-3":
        data_np = MatlabFileReader(dscfg.DATA_PATH).read_file("u")
        data = torch.from_numpy(data_np).permute(3, 0, 1, 2).unsqueeze(2).contiguous()
    elif dataset == "wave":
        with h5py.File(dscfg.DATA_PATH, "r") as f:
            data_np = f["data"][:]
            # data = torch.from_numpy(data_np)[..., 0:1].permute(0, 1, 4, 2, 3)    # [N, T, H, W, C] -> [N, T, C, H, W]
            data = torch.from_numpy(data_np).permute(0, 1, 4, 2, 3)    # [N, T, H, W, C] -> [N, T, C, H, W]
    elif dataset == "sst":
        data = torch.load("./data/sst_T20_N1000.pt", map_location="cpu")["data"]    # [N, T, C, H, W]
    elif dataset == "era5":
        data_np = np.load("./data/ERA5_N550_T20.npz")["data"]
        data = torch.from_numpy(data_np)    # [550, 20, 2, 180, 360]
    if dataset == "era5":
        return PDEDataProcessor(data_tensor=data, cfg=data_cfg,
                                train_ids=list(range(500)), test_ids=list(range(500, 550)))
    else:
        return PDEDataProcessor(data_tensor=data, cfg=data_cfg)