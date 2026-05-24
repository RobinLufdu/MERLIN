from html import parser
import argparse, os, json, h5py
import torch
import numpy as np
from data.data_process import PDEDataProcessor, Dataloader_Configs
from utilities.read_file import MatlabFileReader
from utilities.builder import build_proc_from_run_dir

from config import make_config, get_dataset_cfg
from exp.exp_basic import ExpConfigs
from exp.exp_merlin import Exp_MERLIN

def parse_int_tuple(s: str):
    """Parse '2,2' or '2x2' -> (2, 2)"""
    s = s.strip().lower().replace('x', ',')
    parts = [p for p in s.split(',') if p != '']
    return tuple(int(p) for p in parts)
    

def load_enc_mode_from_run(run_dir: str):
    with open(os.path.join(run_dir, "configs/exp_cfg.json"), "r") as f:
        old_exp = json.load(f)
    return old_exp.get("enc_mode", "set_transformer")


def load_mask_ratio_from_run(run_dir: str):
    with open(os.path.join(run_dir, "configs/dataloader_cfg.json"), "r") as f:
        cfgs = json.load(f)
    return cfgs.get("mask_ratio", 0.0)


def make_phase2_model_cfg_from_run(run_dir: str, current_model_cfg: dict):
    # run_dir: phase1 run directory to load from
    with open(os.path.join(run_dir, "configs/model_cfg.json"), "r") as f:
        base = json.load(f)
    keep_keys = ["encoder","set_encoder","fourier_decoder",
                 "n_frames_cond","state_dim","latent_dim","code_dim","input_channels",]
    for k in keep_keys:
        if k not in base and k in current_model_cfg:
            base[k] = current_model_cfg[k]
    if "latent_process_discrete" in current_model_cfg:
        base["latent_process_discrete"] = current_model_cfg["latent_process_discrete"]
        base["latent_process_discrete"]["code_dim"] = base["code_dim"]
    return base


def main():
    parser = argparse.ArgumentParser(
        description="Command-line training for MERLIN."
    )
    parser.add_argument("--phase", type=str, choices=["phase1", "phase2"], default="phase1")
    parser.add_argument("--train_proj", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--phase1_path", type=str)
    # -------------------- System / Device --------------------
    parser.add_argument("--gpu", type=int, default=0, help="GPU id; ignored if no CUDA.")
    parser.add_argument("--seed", type=int, default=777)
    # -------------------- Data --------------------
    parser.add_argument("--dataset", type=str, default="ns_1e-3")
    parser.add_argument("--limit_trajs", type=int, default=1000, help="Use first N trajectories after load.")
    parser.add_argument("--n_train_trajs", type=int, default=800)
    parser.add_argument("--n_test_trajs", type=int, default=200)
    parser.add_argument("--n_samples_per_traj", type=int, default=2)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--test_bs", type=int, default=32)
    parser.add_argument("--n_frames_train", type=int, default=10)
    parser.add_argument("--n_frames_out", type=int, default=10)
    parser.add_argument("--n_frames_cond", type=int, default=3)
    parser.add_argument("--sample_strategy", type=str, choices=["random", "disjoint"], default="random")

    parser.add_argument("--dt_eval", type=float, default=0.25)
    parser.add_argument("--mask_ratio", type=float, default=0.0)
    parser.add_argument("--block_size", type=str, default="2,2")
    # -------------------- Model (MERLIN defaults) --------------------
    parser.add_argument("--fourier_hidden_dim", type=int, default=128)
    parser.add_argument("--n_fourier_layers", type=int, default=4)
    parser.add_argument("--input_scale", type=float, default=64.0)
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--latent_tokens", type=int, default=4)
    # -----------------------
    parser.add_argument("--pos_emb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--galerkin_in_emb_dim", type=int, default=128)
    parser.add_argument("--enc_heads", type=int, default=4)
    parser.add_argument("--galerkin_spatial_depth", type=int, default=2)
    parser.add_argument("--galerkin_dim_head", type=int, default=32)
    parser.add_argument("--min_freq", type=float, default=1/64)
    parser.add_argument("--galerkin_latent_depth", type=int, default=2)
    # -----------------------
    parser.add_argument("--pos_emb_dim", type=int, default=64)
    parser.add_argument("--pos_hidden", type=int, default=256)
    parser.add_argument("--val_hidden", type=int, default=256)
    parser.add_argument("--set_dim", type=int, default=128)
    parser.add_argument("--set_hidden", type=int, default=256)
    parser.add_argument("--fourier_max_freq", type=float, default=16.0)
    # -----------------------
    parser.add_argument("--modmlp_layers", type=int, default=2)
    parser.add_argument("--modmlp_act", type=str, default="swish")
    # -----------------------
    parser.add_argument("--memory_dim", type=int, default=64)
    parser.add_argument("--memory_type", type=str, default="leaky")
    parser.add_argument("--memory_enc_hidden_dim", type=int, default=256)
    parser.add_argument("--memory_dec_hidden_dim", type=int, default=256)
    parser.add_argument("--memory_enc_layers", type=int, default=2)
    parser.add_argument("--memory_dec_layers", type=int, default=2)
    parser.add_argument("--memory_nl", type=str, default="swish")
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--context_window", type=int, default=3)
    parser.add_argument("--window_pad", type=str, default="repeat")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--augment_variant", type=str, default="current")
    parser.add_argument("--rnn_hidden", type=int, default=256)

    parser.add_argument("--set_num_inds", type=int, default=64)
    parser.add_argument("--set_dropout", type=float, default=0.1)
    parser.add_argument("--rnn_dropout", type=float, default=0.0)
    # -------------------- Optim / Sched --------------------
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr_phase1", type=float, default=5e-3)
    parser.add_argument("--lr_dyn_mem", type=float, default=5e-4)
    parser.add_argument("--lr_dyn_lin", type=float, default=0.0)
    parser.add_argument("--lr_dec", type=float, default=0.0)
    parser.add_argument("--lambda_dyn", type=float, default=1.0)
    parser.add_argument("--lambda_pred", type=float, default=0.0)
    parser.add_argument("--lambda_corr", type=float, default=0.01)
    parser.add_argument("--lambda_spectral", type=float, default=0.01)
    parser.add_argument("--lambda_residual", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.005)
    parser.add_argument("--ema_beta", type=float, default=0.97)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # parser.add_argument("--max_grad_norm", type=float, default=-1.0, help="-1 means disabled")
    parser.add_argument("--scheduler", type=str, choices=["StepLR", "OneCycleLR", "CosineAnnealingLR"], default="StepLR")
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--pct_start", type=float, default=0.3, help="Used by OneCycleLR")
    parser.add_argument("--tf_epsilon", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.99)
    parser.add_argument("--tf_epsilon_min", type=float, default=0.0)
    parser.add_argument("--update_every", type=int, default=200)
    # parser.add_argument("--dec_mode", type=str, choices=["fouriermlp"], default="fouriermlp", help=argparse.SUPPRESS)
    parser.add_argument("--enc_mode", type=str, choices=["galerkin_transformer", "set_transformer"], default="galerkin_transformer")
    # parser.add_argument("--latent_mode", type=str, choices=["discrete"], default="discrete", help=argparse.SUPPRESS)

    # ----------------------------------------
    parser.add_argument("--use_bias", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rollout_steps", type=int, default=2)
    parser.add_argument("--gamma_decay", type=float, default=0.8)
    parser.add_argument("--lambda_lt_pred", type=float, default=0.0)
    parser.add_argument("--lambda_freq", type=float, default=0.0)
    parser.add_argument("--ms_consistency_enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freq_ms_enable", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freq_hf_power", type=float, default=0.0)
    parser.add_argument("--pool_scales", type=str, default="2,4", help="multi-scale pooling scales")

    parser.add_argument("--global_A_mode", type=str, default="ema")
    parser.add_argument("--phase1_linear_mode", type=str, choices=["ridge", "joint_gd"], default="ridge")
    parser.add_argument("--lr_phase1_linear", type=float, default=None)
    parser.add_argument("--wd_phase1_linear", type=float, default=0.0)

    # -------------------- training low dimensional projector --------------------
    parser.add_argument("--proj_epochs", type=int, default=20)
    parser.add_argument("--d", type=int, default=64, help="dimension for the projector")
    parser.add_argument("--lr_proj", type=float, default=0.01)
    parser.add_argument("--lr_dec_proj", type=float, default=1e-4)
    parser.add_argument("--lam_dyn_proj", type=float, default=0.05)
    parser.add_argument("--lam_ortho", type=float, default=0.00)
    parser.add_argument("--stiefel", action=argparse.BooleanOptionalAction, default=False)

    # -------------------- Logging / Eval --------------------
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=5)
    
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    dataset_cfg = get_dataset_cfg(name=args.dataset)
    shapelist = dataset_cfg.SHAPELIST
    data_path = dataset_cfg.DATA_PATH

    if args.phase == "phase2":
        if not args.phase1_path:
            raise ValueError("--phase1_path is required for phase2")

    if args.phase == "phase1":
        mask_ratio = args.mask_ratio
    else:
        mask_ratio = load_mask_ratio_from_run(run_dir=args.phase1_path)
    print(f"Train on mask ratio = {mask_ratio}")
    if mask_ratio != 0.0:
        out_dir = f"./results/MERLIN/{args.dataset}/mask{mask_ratio}"
    elif args.train_proj:
        out_dir = f"./results/MERLIN/{args.dataset}/d={args.d}"
    else:
        out_dir = f"./results/MERLIN/{args.dataset}"

    ############################ Load Data ############################
    if args.phase == "phase1":
        if args.dataset == "ns_1e-3":
            data_np = MatlabFileReader(data_path).read_file("u")
            data = torch.from_numpy(data_np)
            data = data.permute(3, 0, 1, 2).unsqueeze(2)  # [N, T, 1, H, W]
        elif args.dataset == "wave":
            with h5py.File(data_path, "r") as f:
                data_np = f["data"][:]
                # data = torch.from_numpy(data_np)[..., 0:1].permute(0, 1, 4, 2, 3)    # [N, T, H, W, C] -> [N, T, C, H, W]
                data = torch.from_numpy(data_np).permute(0, 1, 4, 2, 3)    # [N, T, H, W, C] -> [N, T, C, H, W]
        elif args.dataset == "sst":
            data = torch.load("./data/sst_T20_N1000.pt", map_location="cpu")["data"]    # [N, T, C, H, W]
        elif args.dataset == "era5":
            data_np = np.load("./data/ERA5_N550_T20.npz")["data"]
            data = torch.from_numpy(data_np)    # [550, 20, 2, 180, 360]
    
        if args.limit_trajs is not None and args.limit_trajs > 0:
            data = data[: args.limit_trajs]

        data_cfg = Dataloader_Configs(
            dataset=args.dataset,
            n_train_trajs=args.n_train_trajs,
            n_test_trajs=args.n_test_trajs,
            n_samples_per_traj=args.n_samples_per_traj,         
            train_bs=args.train_bs,                  
            test_bs=args.test_bs,
            num_workers=0,                
            n_frames_train=args.n_frames_train,
            n_frames_out=args.n_frames_out,
            n_frames_cond=args.n_frames_cond,  
            limit_trajs=args.limit_trajs,            
            normalize=True,
            sample_strategy=args.sample_strategy,     
            mode="interpolation", 
            dt_eval=args.dt_eval,
            seed=args.seed,
            mask_ratio=args.mask_ratio,
            block_size=parse_int_tuple(args.block_size),
            same_over_time=True
        )
        if args.dataset == "era5":
            proc = PDEDataProcessor(data_tensor=data, cfg=data_cfg,
                                    train_ids=list(range(500)), test_ids=list(range(500, 550)))
        else:
            proc = PDEDataProcessor(data_tensor=data, cfg=data_cfg)
    elif args.phase == "phase2":
        proc = build_proc_from_run_dir(run_dir=args.phase1_path, dataset=args.dataset, args=args)

    ############################ Model Configs ############################
    model_cfg = make_config(
        model_name="MERLIN", dataset=args.dataset, n_frames_cond=args.n_frames_cond, 
        fourier_hidden_dim=args.fourier_hidden_dim,        ###################################
        n_fourier_layers=args.n_fourier_layers,            ###################################
        input_scale=args.input_scale,
        ################ Galerkin-Transformer Params ################
        pos_emb=args.pos_emb,                              ###################################
        in_emb_dim=args.galerkin_in_emb_dim,               ###################################
        token_dim=args.token_dim, latent_tokens=args.latent_tokens,
        enc_heads=args.enc_heads,                          ###################################
        spatial_depth=args.galerkin_spatial_depth,
        dim_head=args.galerkin_dim_head,
        min_freq=args.min_freq,
        latent_depth=args.galerkin_latent_depth,
        shapelist=shapelist,
        device=device,
        ################ Set-Transformer Params ################
        pos_emb_dim=args.pos_emb_dim,                      ###################################
        pos_emb_type="fourier",
        pos_hidden=args.pos_hidden,
        val_hidden=args.val_hidden,
        set_dim=args.set_dim,
        set_hidden=args.set_hidden,
        num_inds=args.set_num_inds,  
        use_ln=False,
        fourier_max_freq=args.fourier_max_freq,            ###################################
        dropout=args.set_dropout,
        ################ Fourier-Decoder Params ################
        modmlp_layers=args.modmlp_layers,
        modmlp_act=args.modmlp_act,
        ################ Latent-Process Params ################
        memory_dim=args.memory_dim,
        memory_enc_hidden_dim=args.memory_enc_hidden_dim,
        memory_dec_hidden_dim=args.memory_dec_hidden_dim,
        memory_enc_layers=args.memory_enc_layers,
        memory_dec_layers=args.memory_dec_layers,
        memory_nl=args.memory_nl,
        memory_type=args.memory_type,
        rnn_layers=args.rnn_layers,
        rnn_dropout=args.rnn_dropout,
        gate_per_dim=True,
        latent_ln=True,
        context_window=args.context_window,
        window_pad=args.window_pad,
        augment=args.augment, 
        augment_variant=args.augment_variant, 
        rnn_hidden=args.rnn_hidden
    )
    if args.phase == "phase2":
        model_cfg_ = make_phase2_model_cfg_from_run(run_dir=args.phase1_path, 
                                                    current_model_cfg=model_cfg.as_model_kwargs(include_meta=True))

    ############################ Experiment Configs ############################
    if args.phase == "phase1":
        enc_mode_ = args.enc_mode
    elif args.phase == "phase2":
        enc_mode_ = load_enc_mode_from_run(run_dir=args.phase1_path)
    exp_cfg = ExpConfigs(
        model_name="MERLIN",
        epochs=args.epochs, 
        device=device,
        out_dir=out_dir,
        optimizer="Adam",
        lr=args.lr_phase1,             # for phase I
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,             ############ or None
        scheduler=args.scheduler,
        step_size=args.step_size, gamma=args.gamma, pct_start=args.pct_start,
        teacher_forcing=True,
        tf_epsilon=args.tf_epsilon,
        epsilon=args.epsilon,
        tf_epsilon_min=args.tf_epsilon_min,
        update_every=args.update_every, 
        split_metadata_path=None,

        enc_mode=enc_mode_,
        ####################################################################
        lambda_dyn=args.lambda_dyn,    # for phase I
        lambda_pred=args.lambda_pred,
        lambda_corr=args.lambda_corr,
        lambda_spectral=args.lambda_spectral,
        lambda_lt_pred=args.lambda_lt_pred,
        lambda_residual=args.lambda_residual,

        lr_dyn_mem=args.lr_dyn_mem,
        lr_dyn_lin=args.lr_dyn_lin,
        lr_dec=args.lr_dec,

        use_diag_whiten=True,
        use_bias=args.use_bias,
        rollout_steps=args.rollout_steps,
        gamma_decay=args.gamma_decay,
        lambda_freq=args.lambda_freq,
        ms_consistency_enable=args.ms_consistency_enable,
        freq_ms_enable=args.freq_ms_enable,
        freq_hf_power=args.freq_hf_power,
        ms_pool_scales=parse_int_tuple(args.pool_scales),
        global_A_mode=args.global_A_mode,
        phase1_linear_mode=args.phase1_linear_mode,
        lr_phase1_linear=args.lr_phase1_linear,
        wd_phase1_linear=args.wd_phase1_linear,
    )

    if args.phase == "phase1":
        exp = Exp_MERLIN(args=None, exp_cfg=exp_cfg, model_cfg=model_cfg, data_processor=proc)
        exp.train_phase1_linear(epochs=args.epochs, ridge=args.ridge, ema_beta=args.ema_beta, 
                                use_bias=args.use_bias, use_pred_loss=True, 
                                lambda_pred=args.lambda_pred, lambda_dyn=args.lambda_dyn, 
                                log_every=args.log_every, eval_every=args.eval_every,
                                ms_consistency_enable=args.ms_consistency_enable, freq_ms_enable=args.freq_ms_enable)
    elif args.phase == "phase2":
        exp = Exp_MERLIN(args=None, exp_cfg=exp_cfg, model_cfg=model_cfg_, data_processor=proc)
        if args.train_proj:
            exp.train_projector_from_phase1_ckpt(phase1_path=os.path.join(args.phase1_path, "phase1_best_rec.pth"),
                                                 d=args.d, epochs=args.proj_epochs, lr=args.lr_proj, lr_dec=args.lr_dec_proj,
                                                 lambda_dyn=args.lam_dyn_proj, lambda_ortho=args.lam_ortho,
                                                 log_every=args.log_every, stiefel=args.stiefel)
        exp.train_phase2(phase1_path=os.path.join(args.phase1_path, "phase1_best_rec.pth"), log_every=args.log_every, eval_every=args.eval_every)

if __name__ == "__main__":
    main()
