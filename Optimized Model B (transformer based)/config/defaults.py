# defaults.py
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Change Log (for this file only)
# [2025-09-11 | Hang Zhang] Added basic SupCon configs (LOSS.SUPCON: ENABLE/W/T).
# [2025-09-14 | Hang Zhang] Added SupCon SEARCH grid (LOSS.SUPCON.SEARCH) for B1.
# [2025-09-15 | Hang Zhang] Narrowed SEARCH to W=[0.25,0.30,0.35], T=[0.07].
# [2025-09-15 | Hang Zhang] Added TripletX stub (LOSS.TRIPLETX) for B2 YAML merge.
# [2025-09-16 | Zeyu Yang] Added `TRAINING_MODE` to toggle between self-supervised and supervised training.
# [2025-09-16 | Hang Zhang] Default to supervised; decouple SupCon from TRAINING_MODE.
#                           Set MODEL.TRAINING_MODE="supervised" and LOSS.SUPCON.ENABLE=False by default.
# [2025-09-16 | Hang Zhang] **Update defaults to current model baseline (TransReID + DeiT stride):**
#                           ViT backbone, STRIDE_SIZE=[12,12], SIE/JPM on, VeRi dataset, P×K sampler,
#                           SGD schedule (120 epochs), test batch=256.
# [2025-09-17 | Hang Zhang] Added SOLVER.MARGIN=0.3 for backward compatibility with legacy TripletLoss.
# [2025-10-14 | Hang Zhang] **Add LOSS.PHASED guard switch (default False) to avoid impacting
#                           non-phased configs; phased scheduling is only active when enabled.**
# [2025-10-14 | Hang Zhang] Add DATALOADER.PHASED guard with K_WHEN_TRIPLETX/K_OTHER (default OFF).
# [2025-10-15 | Hang Zhang] Add editable phased-loss keys so YACS can merge from CLI safely:
#                           LOSS.PHASED.{BOUNDARIES, METRIC_SEQ, W_METRIC_SEQ, W_SUP_SPEC}.
# [2025-10-15 | Hang Zhang] **NEW:** Add FEAT_SRC routing keys for each loss:
#                           LOSS.CE.FEAT_SRC, LOSS.TRIPLET.FEAT_SRC, LOSS.TRIPLETX.FEAT_SRC, LOSS.SUPCON.FEAT_SRC.
#                           Defaults: CE/SupCon='bnneck', Triplet/TripletX='pre_bn'
# [2025-10-19 | Hang Zhang] **Remove dynamic CE source key**:
#                           - Deleted LOSS.CE.FEAT_SRC; CE always uses classifier logits (score).
#                           - Any YAML/CLI setting for CE.FEAT_SRC is no longer supported/needed.
# [2025-10-19 | Hang Zhang]  **Add phased keys used by make_loss**:
#                           - LOSS.PHASED.TRIPLETX_END (epoch boundary for TripletX→Triplet).
#                           - LOSS.SUPCON.W0 / DECAY_TYPE / DECAY_START / DECAY_END (epoch-aware SupCon).
#[2025-10-21 | Hang Zhang] Set LOSS.SUPCON.CAM_AWARE=True and POS_RULE='class' to match actual behavior;
#                          SupConLoss already runs in camera-aware mode (same ID, different camera),
#                          so enabling it by default prevents misleading configuration semantics.

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Config definition root
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# MODEL (baseline: TransReID + DeiT + official stride)
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Device
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = '0'

# Backbone
_C.MODEL.NAME = 'transformer'                           # ViT-TransReID as default
_C.MODEL.TRANSFORMER_TYPE = 'deit_base_patch16_224_TransReID'
_C.MODEL.LAST_STRIDE = 1
_C.MODEL.STRIDE_SIZE = [12, 12]                         # official stride
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0

# Pretrain
_C.MODEL.PRETRAIN_PATH = ''                             # set in YAML/--opts if needed
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'                   # imagenet | self | finetune

# Neck / losses (supervised baseline)
_C.MODEL.NECK = 'bnneck'
_C.MODEL.IF_WITH_CENTER = 'no'
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
_C.MODEL.NO_MARGIN = True                               # soft triplet
_C.MODEL.IF_LABELSMOOTH = 'off'
_C.MODEL.COS_LAYER = False

# DDP / misc
_C.MODEL.DIST_TRAIN = False

# JPM / SIE
_C.MODEL.JPM = True
_C.MODEL.SHIFT_NUM = 8
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = True
_C.MODEL.SIE_VIEW = True

# Training mode (decoupled from loss toggles)
_C.MODEL.TRAINING_MODE = "supervised"                   # "supervised" | "self_supervised"

# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------
_C.LOSS = CN()

# ====== Cross-Entropy (ID) ======
# CE uses classifier logits (score) directly; no FEAT_SRC needed.
_C.LOSS.CE = CN()

# ====== Triplet (baseline) ======
_C.LOSS.TRIPLET = CN()
# Which feature branch to use for Triplet: ['bnneck', 'pre_bn']
_C.LOSS.TRIPLET.FEAT_SRC = 'pre_bn'

# ====== TripletX (enhanced triplet) ======
_C.LOSS.TRIPLETX = CN()
_C.LOSS.TRIPLETX.ENABLE = False
_C.LOSS.TRIPLETX.W = 1.0
_C.LOSS.TRIPLETX.MARGIN = 0.30
_C.LOSS.TRIPLETX.SOFT_WARMUP = True
_C.LOSS.TRIPLETX.WARMUP_EPOCHS = 10
_C.LOSS.TRIPLETX.K = 5
_C.LOSS.TRIPLETX.ALPHA = 2.0
_C.LOSS.TRIPLETX.CROSS_CAM_POS = True
_C.LOSS.TRIPLETX.SAME_CAM_NEG_BOOST = 1.2
_C.LOSS.TRIPLETX.NORM_FEAT = True
# Which feature branch to use for TripletX: ['bnneck', 'pre_bn']
_C.LOSS.TRIPLETX.FEAT_SRC = 'pre_bn'

# ====== SupCon (supervised contrastive) ======
_C.LOSS.SUPCON = CN()
_C.LOSS.SUPCON.ENABLE = False
_C.LOSS.SUPCON.W = 0.30
_C.LOSS.SUPCON.T = 0.07
_C.LOSS.SUPCON.CAM_AWARE = True
_C.LOSS.SUPCON.POS_RULE = "class"
# Which feature branch to use for SupCon: ['bnneck', 'pre_bn']
_C.LOSS.SUPCON.FEAT_SRC = 'bnneck'
# Epoch-aware SupCon scheduling (used by make_loss when PHASED.ENABLE=True)
_C.LOSS.SUPCON.W0 = 0.30
_C.LOSS.SUPCON.DECAY_TYPE = "linear"  # ["linear","const","exp"]
_C.LOSS.SUPCON.DECAY_START = 30
_C.LOSS.SUPCON.DECAY_END = 60

# Grid for B1 auto search
_C.LOSS.SUPCON.SEARCH = CN()
_C.LOSS.SUPCON.SEARCH.W = [0.25, 0.30, 0.35]
_C.LOSS.SUPCON.SEARCH.T = [0.07]

# ---- Phased loss global guard (OFF by default) --------------------------------
# Only when LOSS.PHASED.ENABLE=True (e.g., in deit_transreid_stride_b2_phased_loss.yml),
# make_loss(...) will activate epoch-aware scheduling; otherwise, all configs keep static weights.
_C.LOSS.PHASED = CN()
_C.LOSS.PHASED.ENABLE = False

# Editable phased-loss keys (so CLI/YAML can override safely)
# Semantics: half-open segments using boundaries B = [b0, b1, ...]
#   [0, b0), [b0, b1), [b1, +∞)
_C.LOSS.PHASED.BOUNDARIES   = [30, 60]                      # default A/B/C boundaries
_C.LOSS.PHASED.METRIC_SEQ   = ['tripletx', 'triplet', 'triplet']
_C.LOSS.PHASED.W_METRIC_SEQ = [1.2, 1.0, 1.0]
# SupCon weight spec per phase: 'const:x' or 'linear:x->y' (linear interpolation within phase)
_C.LOSS.PHASED.W_SUP_SPEC   = ['const:0.30', 'linear:0.30->0.15', 'const:0.0']
# Simple boundary used by current make_loss implementation
_C.LOSS.PHASED.TRIPLETX_END = 30

# -----------------------------------------------------------------------------
# INPUT / AUGMENTATION
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [256, 256]
_C.INPUT.SIZE_TEST  = [256, 256]
_C.INPUT.PROB = 0.5
_C.INPUT.RE_PROB = 0.8
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD  = [0.229, 0.224, 0.225]
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# DATASETS
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.NAMES = 'veri'
_C.DATASETS.ROOT_DIR = '../datasets'

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.SAMPLER = 'softmax_triplet'               # P×K sampler
_C.DATALOADER.NUM_INSTANCE = 4

# ---- Phased PK-sampler K (global guard; OFF by default) -----------------------
_C.DATALOADER.PHASED = CN()
_C.DATALOADER.PHASED.ENABLE = False
_C.DATALOADER.PHASED.K_WHEN_TRIPLETX = 8
_C.DATALOADER.PHASED.K_OTHER = 4

# -----------------------------------------------------------------------------
# SOLVER / OPTIMIZATION
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.MAX_EPOCHS = 120
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.WEIGHT_DECAY_BIAS = 1e-4
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (40, 70)

# Triplet loss margin (legacy baseline)
_C.SOLVER.MARGIN = 0.3

# Warmup
_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.WARMUP_METHOD = "linear"

# Logging / eval / ckpt
_C.SOLVER.CHECKPOINT_PERIOD = 120
_C.SOLVER.LOG_PERIOD = 50
_C.SOLVER.EVAL_PERIOD = 120

# Global batch size
_C.SOLVER.IMS_PER_BATCH = 64

# Cosine head defaults
_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30
_C.SOLVER.CENTER_LR = 0.5
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
_C.SOLVER.SEED = 1234

# ====== Brightness gating (dynamic dark-image handling) ======
_C.LOSS.BRIGHTNESS = CN()
_C.LOSS.BRIGHTNESS.ENABLE = False   # turn on in YAML when needed
_C.LOSS.BRIGHTNESS.THRESH = 0.35    # mean RGB threshold in [0,1]
_C.LOSS.BRIGHTNESS.K = 0.08         # extra weight or gating factor (your usage)


# ---- CLIP (HF TinyCLIP local) ----
_C.MODEL.USE_CLIP = True
_C.MODEL.CLIP_HF_ID =  "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
_C.MODEL.CLIP_LOCAL_PATH = ""
_C.MODEL.CLIP_INPUT_SIZE = (256, 256)
_C.MODEL.CLIP_FINETUNE=True



# -----------------------------------------------------------------------------
# TEST / EVALUATION
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 256

_C.TEST.RE_RANKING = False
_C.TEST.WEIGHT = ""
_C.TEST.NECK_FEAT = 'before'
_C.TEST.FEAT_NORM = 'yes'
_C.TEST.DIST_MAT = "dist_mat.npy"
_C.TEST.EVAL = True

# -----------------------------------------------------------------------------
# Misc options
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = ""
