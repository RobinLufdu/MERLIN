import argparse, os, json
import torch
from utilities.builder import build_proc_from_run_dir
from exp.exp_basic import Exp_Basic, ExpConfigs
from exp.exp_merlin import Exp_MERLIN
from exp.exp_vis import plot_Ad_spectrum
from exp.exp_vis import plot_dyn_energy_stats, plot_lowdim_time_series, save_rollout_comparison
from exp.exp_vis import visualize_random_rollout, visualize_rollout_by_index, visualize_sample_evolution

def _json_default(o):
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.item() if o.numel() == 1 else o.detach().cpu().tolist()
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def main():
    parser = argparse.ArgumentParser(
        description="Command-line evaluation for MERLIN."
    )
    parser.add_argument("--eval_mode", type=str, default="all")    # "all" / "phase1"
    parser.add_argument("--phase1_path", type=str)
    parser.add_argument("--model_path", type=str)   # final model path after phase2 training
    parser.add_argument("--gpu", type=int, default=0, help="GPU id; ignored if no CUDA.")
    parser.add_argument("--dataset", type=str, default="ns_1e-3")
    parser.add_argument("--seq_id", type=int, default=0)
    parser.add_argument("--traj_id", type=int, default=909)    # (traj_id,t0) should match the group, see split_metadata.json
    parser.add_argument("--t0", type=int, default=12)
    parser.add_argument("--rollout_steps", type=int, default=15)
    parser.add_argument("--rom", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    proc = build_proc_from_run_dir(run_dir=args.model_path, dataset=args.dataset)

    cfgs = Exp_Basic.load_all_configs(args.model_path)
    model_cfg_dict  = cfgs["model_cfg"]
    exp_cfg_dict    = cfgs["exp_cfg"] 
    exp_cfg_dict = dict(exp_cfg_dict)
    if isinstance(exp_cfg_dict.get("device"), str):
        exp_cfg_dict["device"] = device
    exp_cfg = ExpConfigs(**exp_cfg_dict)
    exp = Exp_MERLIN(args=None, exp_cfg=exp_cfg, model_cfg=model_cfg_dict, data_processor=proc)

    if not args.rom:
        _ = exp.load_phase1_ckpt(path=os.path.join(args.phase1_path, "phase1_best_rec.pth"))
        out_dir_phase1 = os.path.join(args.model_path, "vis/phase1")
        os.makedirs(out_dir_phase1, exist_ok=True)
        plot_Ad_spectrum(exp, save_dir=out_dir_phase1)
        visualize_random_rollout(exp, 
            group="test", batch_size=1, rollout_steps=args.rollout_steps,
            out_dir=os.path.join(out_dir_phase1, "test"), mode="all", dyn_type="linear"
        )
    
    if args.eval_mode == "all":
        info = exp.load_from_ckpt(ckpt_path=os.path.join(args.model_path, "model_tr_best.pth"))
        assert exp.whiten_scale is not None    
        out_dir_phase2 = os.path.join(args.model_path, "vis/phase2")
        os.makedirs(out_dir_phase2, exist_ok=True)

        exp._ensure_loader("train_eval")
        exp._ensure_loader("test")
        train_errs = exp.evaluate(exp.train_eval_loader)
        test_errs = exp.evaluate(exp.test_loader)
        with open(os.path.join(args.model_path, "metrics_train.json"), "w") as f:
            json.dump(train_errs, f, indent=2, default=_json_default)
        with open(os.path.join(args.model_path, "metrics_test.json"), "w") as f:
            json.dump(test_errs, f, indent=2, default=_json_default)
        
        visualize_random_rollout(exp, 
            group="test", batch_size=1, rollout_steps=args.rollout_steps,
            out_dir=os.path.join(out_dir_phase2, "test"), mode="all", dyn_type="memory"
        )
        visualize_rollout_by_index(exp, 
            group="train_eval", seq_index=args.seq_id, rollout_steps=args.rollout_steps,  
            out_dir=os.path.join(out_dir_phase2, f"idx_{args.seq_id}"), mode="all", dyn_type="memory"
        )

        save_linear = not args.rom
        exp.save_rollout_tensors(out_dir=os.path.join(args.model_path, "saved_tensors/phase2"),
                                 traj_id=args.traj_id, t0=args.t0, rollout_steps=args.rollout_steps,
                                 phase="phase2")
        if save_linear:
            exp.save_rollout_tensors(out_dir=os.path.join(args.model_path, "saved_tensors/phase1"),
                                     traj_id=args.traj_id, t0=args.t0, rollout_steps=args.rollout_steps,
                                     phase="phase1")
            
        ############################## Latent Visualizations ##############################
        if args.dataset != "sst":
            if save_linear:
                plot_dyn_energy_stats(exp, exp.train_eval_loader, save_dir=os.path.join(args.model_path, "latent/linear_vs_memory"))
                visualize_sample_evolution(exp, 
                    group="test", traj_id=args.traj_id, t0=args.t0, save_dir=os.path.join(args.model_path, "latent/linear_vs_memory"),
                    bg_mode="landscape", bg_alpha=0.75
                )
                # exp.visualize_phase_plane_2d(os.path.join(args.model_path, "latent/phase"))
            else:
                plot_lowdim_time_series(exp, 
                    time_scale=4.0 if args.dataset=="ns_1e-3" else 1.0,
                    traj_id=args.traj_id, t0=0, steps=30,
                    use_center=True, use_whiten=True, use_projector=True,
                    save_dir=os.path.join(args.model_path, "rom/latent_modes"), fname_prefix="ytime",
                    topk_by_var=8
                )


if __name__ == "__main__":
    main()
