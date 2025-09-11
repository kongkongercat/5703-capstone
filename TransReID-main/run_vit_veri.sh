#!/bin/bash

# Auto-detect: cuda > mps (Apple Silicon) > cpu
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

# num_workers suggestion
if [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "mps" ]; then
  WORKERS=8
else
  WORKERS=0
fi

echo "Using device: $DEVICE (num_workers=$WORKERS)"

# --------- config ----------
EPOCHS=${1:-10}                                # 可选：第1个入参控制总epoch数，默认10
BATCH=${2:-64}                                 # 可选：第2个入参控制batch size，默认64（论文设置）
OUTDIR="./logs/veri776_run"
TESTDIR="./logs/veri776_test"
CFG="configs/VeRi/vit_transreid_stride.yml"
DATA="./data"
PRE="./pretrained/jx_vit_base_p16_224-80ecf9dd.pth"

mkdir -p "$OUTDIR" "$TESTDIR"

# 可选：把训练日志也保存下来
TRAIN_LOG="$OUTDIR/train.log"
TEST_LOG="$TESTDIR/test_all.log"

# -------------------------
# Step 1. Train on VeRi-776
#   - 每 1 个 epoch 存一次 (CHECKPOINT_PERIOD=1)
# -------------------------
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

# -------------------------
# Step 2. Test checkpoints epoch-by-epoch
#   - 依次用 transformer_1.pth ... transformer_${EPOCHS}.pth 做测试
# -------------------------
echo "===== Begin per-epoch evaluation =====" | tee "$TEST_LOG"

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

echo "===== All done. Check logs in: $OUTDIR and $TESTDIR ====="
