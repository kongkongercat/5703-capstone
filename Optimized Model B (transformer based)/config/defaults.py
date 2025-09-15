from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Change Log (for this file only)
# [2025-09-11 | Hang Zhang] Added basic SupCon configs (LOSS.SUPCON: ENABLE/W/T).
# [2025-09-14 | Hang Zhang] Added SupCon SEARCH grid (LOSS.SUPCON.SEARCH) for B1.
# [2025-09-15 | Hang Zhang] Narrowed SEARCH to W=[0.25,0.30,0.35], T=[0.07].
# [2025-09-15 | Hang Zhang] Added TripletX stub (LOSS.TRIPLETX) for B2 YAML merge.
# [2025-09-16 | Zeyu Yang] Added `TRAINING_MODE` to toggle between self-supervised and supervised training.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
# Use ImageNet pretrained model to initialize backbone or use self trained model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'

# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train loss include center loss, options: 'yes' or 'no'
_C.MODEL.IF_WITH_CENTER = 'no'

# Loss function settings for supervised training
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'

# If train with multi-gpu ddp mode
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss
_C.MODEL.NO_MARGIN = False
# If train with label smooth
_C.MODEL.IF_LABELSMOOTH = 'on'
# If train with arcface loss
_C.MODEL.COS_LAYER = False

# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# JPM Parameter
_C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 4
_C.MODEL.RE_ARRANGE = True

# SIE Parameter 
_C.MODEL.SIE_COE = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# Training mode configuration (new)
_C.MODEL.TRAINING_MODE = "self_supervised"  # Options: "supervised", "self_supervised"

# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------
_C.LOSS = CN()

# ----- Supervised Contrastive (SupCon) -----
_C.LOSS.SUPCON = CN()
_C.LOSS.SUPCON.ENABLE = True if _C.MODEL.TRAINING_MODE == "self_supervised" else False  # Toggle based on TRAINING_MODE
_C.LOSS.SUPCON.W = 0.30                # default weight
_C.LOSS.SUPCON.T = 0.07                # default temperature
_C.LOSS.SUPCON.CAM_AWARE = False       # optional: camera-aware positives
_C.LOSS.SUPCON.POS_RULE = "class"      # "class" or "class+camera"

# Grid for B1 auto search
_C.LOSS.SUPCON.SEARCH = CN()
_C.LOSS.SUPCON.SEARCH.W = [0.25, 0.30, 0.35]
_C.LOSS.SUPCON.SEARCH.T = [0.07]

# ----- TripletX (B2) : stub to avoid KeyError on YAML merge -----
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
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.ROOT_DIR = ('../data')

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epochs
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Seed
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
# Weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
# LR decay
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (40, 70)
# warm up
_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.WARMUP_METHOD = "linear"
# cosine head defaults
_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30
# checkpoint / log / eval
_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.EVAL_PERIOD = 10
# global batch size
_C.SOLVER.IMS_PER_BATCH = 64

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test
_C.TEST.NECK_FEAT = 'after'
# Whether feature is normalized before test
_C.TEST.FEAT_NORM = 'yes'
# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score
_C.TEST.EVAL = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
