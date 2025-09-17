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
_C.MODEL.NAME = 'transformer'                           # ← ViT-TransReID as default
_C.MODEL.TRANSFORMER_TYPE = 'vit_base_patch16_224_TransReID'
_C.MODEL.LAST_STRIDE = 1
_C.MODEL.STRIDE_SIZE = [12, 12]                         # official stride
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0

# Pretrain
_C.MODEL.PRETRAIN_PATH = ''                             # leave empty; set in YAML/--opts
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

# SupCon (supervised contrastive) — default OFF; enable in B1/B2 YAML or via --opts
_C.LOSS.SUPCON = CN()
_C.LOSS.SUPCON.ENABLE = False
_C.LOSS.SUPCON.W = 0.30
_C.LOSS.SUPCON.T = 0.07
_C.LOSS.SUPCON.CAM_AWARE = False
_C.LOSS.SUPCON.POS_RULE = "class"

# Grid for B1 auto search
_C.LOSS.SUPCON.SEARCH = CN()
_C.LOSS.SUPCON.SEARCH.W = [0.25, 0.30, 0.35]
_C.LOSS.SUPCON.SEARCH.T = [0.07]

# TripletX (for B2) — stub defaults; enable explicitly when needed
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

# -----------------------------------------------------------------------------
# INPUT / AUGMENTATION
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [256, 256]                        # project baseline
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
_C.DATASETS.NAMES = 'veri'                              # ← string
_C.DATASETS.ROOT_DIR = '../datasets'                    # ← string

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.SAMPLER = 'softmax_triplet'               # P×K sampler
_C.DATALOADER.NUM_INSTANCE = 4

# -----------------------------------------------------------------------------
# SOLVER / OPTIMIZATION
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.MAX_EPOCHS = 120                              # full baseline schedule
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.LARGE_FC_LR = False
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.WEIGHT_DECAY_BIAS = 1e-4
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (40, 70)

# Triplet loss margin (legacy baseline, still referenced in make_loss.py)
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

# Cosine head defaults (kept for compatibility)
_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30
_C.SOLVER.CENTER_LR = 0.5
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
_C.SOLVER.SEED = 1234

# -----------------------------------------------------------------------------
# TEST / EVALUATION
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 256                             # larger test batch
_C.TEST.RE_RANKING = False
_C.TEST.WEIGHT = ""
_C.TEST.NECK_FEAT = 'before'
_C.TEST.FEAT_NORM = 'yes'
_C.TEST.DIST_MAT = "dist_mat.npy"
_C.TEST.EVAL = True

# -----------------------------------------------------------------------------
# Misc options
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = ""                                      # set via YAML/--opts
