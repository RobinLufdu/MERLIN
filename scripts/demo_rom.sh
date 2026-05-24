#!/bin/bash

set -u
# ======= global config =======
GPU_ID=0
SEED=777
DATASET="ns_1e-3"
LIMIT_TRAJS=1000
N_TRAIN_TRAJS=800
N_TEST_TRAJS=200
N_SAMPLE_PER_TRAJ=2
TRAIN_BS=16
TEST_BS=32
N_FRAMES_TRAIN=10
N_FRAMES_OUT=10
SAMPLE_STRATEGY="random"
MASK_RATIO=0.0
EPOCHS=500

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
    --phase "phase2" \
    --phase1_path "./results/MERLIN/ns_1e-3/phase1_ckpt" \
    --gpu 5 \
    --seed "$SEED" \
    --dataset "$DATASET" \
    --n_frames_cond 3\
    --n_frames_train "$N_FRAMES_TRAIN"\
    --n_frames_out "$N_FRAMES_OUT"\
    --sample_strategy "$SAMPLE_STRATEGY"\
    --n_samples_per_traj "$N_SAMPLE_PER_TRAJ"\
    --memory_dim 128 \
    --memory_type "leaky" \
    --memory_enc_hidden_dim 512 \
    --memory_dec_hidden_dim 512 \
    --memory_enc_layers 2 \
    --memory_dec_layers 2 \
    --memory_nl "swish" \
    --epochs 500 \
    --lr_dyn_mem 5e-4 \
    --lr_dyn_lin 0.0 \
    --lr_dec 1e-4 \
    --lambda_pred 0.01\
    --lambda_corr 0.01\
    --lambda_spectral 0.0\
    --lambda_lt_pred 0.0\
    --lambda_residual 1.0\
    --scheduler "StepLR"\
    --step_size 50\
    --gamma 0.8\
    --pct_start 0.3\
    --tf_epsilon 0.9\
    --epsilon 0.99\
    --tf_epsilon_min 0.0\
    --update_every 200\
    --log_every 5\
    --eval_every 200\
    --train_proj \
    --proj_epochs 100 \
    --d 16 \
    --lr_proj 0.01 \
    --lr_dec_proj 1e-4\
    --lam_dyn_proj 0.05\
    --lam_ortho 0.05 \


if ((${#fails[@]})); then
  echo "==== Failed runs ===="
  printf '%s\n' "${fails[@]}"
  exit 1
else
  echo "All runs finished."
fi