#!/bin/bash

set -u
# ======= global config =======
GPU_ID=1
SEED=777
DATASET="ns_1e-3"
LIMIT_TRAJS=1000
N_TRAIN_TRAJS=800
N_TEST_TRAJS=200
N_SAMPLE_PER_TRAJ=4
TRAIN_BS=16
TEST_BS=32
N_FRAMES_TRAIN=10
N_FRAMES_COND=3
SAMPLE_STRATEGY="random"
MASK_RATIO=0.0
EPOCHS=800

mkdir -p logs
fails=()
ONLY="${ONLY:-}"
should_run() {
  local name="$1"
  [[ -z "$ONLY" ]] && return 0
  IFS=, read -ra arr <<<"$ONLY"
  for x in "${arr[@]}"; do [[ "$name" == "$x" ]] && return 0; done
  return 1
}

# ==================== Launch Main Program ====================
try_run () {
  local name="$1"; shift
  echo "======== $name ========"
  python run_merlin.py "$@" 2>&1 | tee "logs/${name}.log"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "[WARN] $name failed with exit code $rc; continue..."
    fails+=("$name:$rc")
  fi
}

should_run run1 && try_run run1 \
    --phase "phase1" \
    --gpu "$GPU_ID" \
    --seed "$SEED" \
    --dataset "$DATASET" \
    --limit_trajs "$LIMIT_TRAJS" \
    --n_train_trajs "$N_TRAIN_TRAJS"\
    --n_test_trajs "$N_TEST_TRAJS"\
    --n_samples_per_traj "$N_SAMPLE_PER_TRAJ"\
    --train_bs "$TRAIN_BS"\
    --test_bs "$TEST_BS"\
    --n_frames_cond "$N_FRAMES_COND"\
    --sample_strategy "$SAMPLE_STRATEGY"\
    --mask_ratio "$MASK_RATIO"\
    --block_size "2,2"\
    --fourier_hidden_dim 128\
    --n_fourier_layers 4\
    --token_dim 16\
    --latent_tokens 8\
    --modmlp_layers 2\
    --modmlp_act "swish"\
    --galerkin_in_emb_dim 128\
    --enc_heads 4\
    --galerkin_spatial_depth 3\
    --galerkin_dim_head 32\
    --galerkin_latent_depth 2\
    --epochs "$EPOCHS"\
    --lr_phase1 1e-3\
    --lambda_dyn 0.05\
    --lambda_pred 0.1\
    --ridge 0.005\
    --ema_beta 0.97\
    --scheduler "StepLR"\
    --step_size 25\
    --gamma 0.9\
    --pct_start 0.3\
    --enc_mode "galerkin_transformer"\
    --log_every 5\
    --eval_every 5

should_run run2 && try_run run2 \
    --phase "phase1" \
    --gpu 2 \
    --seed "$SEED" \
    --dataset "$DATASET" \
    --limit_trajs "$LIMIT_TRAJS" \
    --n_train_trajs "$N_TRAIN_TRAJS"\
    --n_test_trajs "$N_TEST_TRAJS"\
    --n_samples_per_traj "$N_SAMPLE_PER_TRAJ"\
    --train_bs "$TRAIN_BS"\
    --test_bs "$TEST_BS"\
    --n_frames_cond "$N_FRAMES_COND"\
    --sample_strategy "$SAMPLE_STRATEGY"\
    --mask_ratio "$MASK_RATIO"\
    --block_size "2,2"\
    --fourier_hidden_dim 128\
    --n_fourier_layers 4\
    --token_dim 16\
    --latent_tokens 8\
    --modmlp_layers 2\
    --modmlp_act "swish"\
    --pos_emb \
    --galerkin_in_emb_dim 128\
    --enc_heads 4\
    --galerkin_spatial_depth 3\
    --galerkin_dim_head 32\
    --galerkin_latent_depth 2\
    --epochs "$EPOCHS"\
    --lr_phase1 1e-3\
    --lambda_dyn 0.05\
    --lambda_pred 0.1\
    --ridge 0.005\
    --ema_beta 0.97\
    --scheduler "StepLR"\
    --step_size 25\
    --gamma 0.9\
    --pct_start 0.3\
    --enc_mode "galerkin_transformer"\
    --log_every 5\
    --eval_every 5

if ((${#fails[@]})); then
  echo "==== Failed runs ===="
  printf '%s\n' "${fails[@]}"
  exit 1
else
  echo "All runs finished."
fi