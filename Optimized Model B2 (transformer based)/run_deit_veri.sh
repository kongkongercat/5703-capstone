#!/bin/bash
set -e

# ---------------------------------------------------------
# TransReID - VeRi-776 training/testing script (DeiT only)
# ---------------------------------------------------------
# Usage:
#   bash run_deit_veri.sh [EPOCHS] [BATCH]
#   - EPOCHS (optional): total epochs to train (default: 10)
#   - BATCH  (optional): batch size (default: 64)
#
# This script:
#   1) Auto-detects device (cuda > mps > cpu) and sets num_workers.
#   2) Trains TransReID (DeiT backbone) on VeRi-776, saving a checkpoint each epoch.
#   3) Evaluates every saved checkpoint epoch-by-epoch.
# ---------------------------------------------------------

# --------- device auto-detect: cuda > mps (Apple Silicon) > cpu ----------
DEVICE=$(
python - <<'PY'
import torch
if torch.cuda.is_available():
    print("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)

# Suggested dataloader workers (tune if needed)
if [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "mps" ]; then
  WORKERS=8
else
  WORKERS=0
fi

echo "Using device: $DEVICE (num_workers=$WORKERS)"

# ----------------- arguments & defaults -------------------
EPOCHS=${1:-10}                       # total training epochs (default: 10)
BATCH=${2:-64}                        # batch size (default: 64)

# ----------------- paths & configs ------------------------
# IMPORTANT: ensure the following paths exist in your project
CFG="configs/VeRi/deit_transreid_stride.yml"       # DeiT config for VeRi
DATA="./data"                                      # dataset root (expects data/VeRi/{image_train,image_query,image_test})
PRE="./pretrained/deit_base_distilled_patch16_224-df68dfff.pth"  # DeiT ImageNet pretrained weight

OUTDIR="./logs/veri776_deit_run"     # training outputs (checkpoints, logs)
TESTDIR="./logs/veri776_deit_test"   # per-epoch evaluation outputs

mkdir -p "$OUTDIR" "$TESTDIR" "./pretrained"

# Sanity checks
if [ ! -f "$CFG" ]; then
  echo "ERROR: Config not found: $CFG"
  exit 1
fi

if [ ! -f "$PRE" ]; then
  echo "ERROR: Pretrained weight not found: $PRE"
  echo "Place the DeiT weight here or update PRE path."
  exit 1
fi

if [ ! -d "$DATA/VeRi/image_train" ] || [ ! -d "$DATA/VeRi/image_query" ] || [ ! -d "$DATA/VeRi/image_test" ]; then
  echo "ERROR: VeRi dataset folders not found under $DATA/VeRi/"
  echo "Expected: image_train, image_query, image_test"
  exit 1
fi

# Optional: also require viewpoint files if SIE-View is enabled in config
# (uncomment if your config turns on MODEL.SIE_VIEW: True)
# for f in "datasets/keypoint_train.txt" "datasets/keypoint_test.txt"; do
#   [ -f "$f" ] || { echo "ERROR: Missing $f (viewpoint labels)"; exit 1; }
# done

# Log files
TRAIN_LOG="$OUTDIR/train.log"
TEST_LOG="$TESTDIR/test_all.log"

# -------------------- TRAIN -------------------------------
# Notes:
# - Saves a checkpoint each epoch via SOLVER.CHECKPOINT_PERIOD 1
# - Uses your same CLI style (MODEL.DEVICE / ROOT_DIR / PRETRAIN_PATH etc.)
echo "===== Begin training (DeiT) ====="
python train.py \
  --config_file "$CFG" \
  MODEL.DEVICE "$DEVICE" \
  DATASETS.ROOT_DIR "$DATA" \
  MODEL.PRETRAIN_PATH "$PRE" \
  DATALOADER.NUM_WORKERS "$WORKERS" \
  SOLVER.IMS_PER_BATCH "$BATCH" \
  SOLVER.MAX_EPOCHS "$EPOCHS" \
  SOLVER.CHECKPOINT_PERIOD 1 \
  OUTPUT_DIR "$OUTDIR" | tee "$TRAIN_LOG"

# -------------------- TEST (per epoch) --------------------
echo "===== Begin per-epoch evaluation (DeiT) =====" | tee "$TEST_LOG"

for ((e=1; e<=EPOCHS; e++)); do
  CKPT="$OUTDIR/transformer_${e}.pth"
  if [ -f "$CKPT" ]; then
    SUBTEST="$TESTDIR/epoch_${e}"
    mkdir -p "$SUBTEST"
    echo ">>> Testing epoch $e with weight: $CKPT"
    python test.py \
      --config_file "$CFG" \
      MODEL.DEVICE "$DEVICE" \
      DATASETS.ROOT_DIR "$DATA" \
      TEST.WEIGHT "$CKPT" \
      OUTPUT_DIR "$SUBTEST" | tee -a "$TEST_LOG"
  else
    echo ">>> Skip epoch $e (checkpoint not found: $CKPT)" | tee -a "$TEST_LOG"
  fi
done

echo "===== All done (DeiT). Check logs in: $OUTDIR and $TESTDIR ====="
