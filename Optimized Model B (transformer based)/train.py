# encoding: utf-8
"""
ReID Training Entrypoint

Original Authors:
- liaoxingyu (sherlockliao01@gmail.com)

Modified by Zhang Hang on 2025-09-11:
- Added support for Supervised Contrastive Loss (SupConLoss)
- Integrate cfg.LOSS.SUPCON.ENABLE / W / T into total loss calculation

Patched by Zhang Hang on 2025-09-14:
- Align SupCon cfg keys with make_loss/yml (T / W), remove ghost default 0.2 in logs
- Safer CfgNode checks; unified logging strictly from cfg
- More reproducible seed/CUDNN setup (benchmark=False with deterministic=True)
- Robust CUDA_VISIBLE_DEVICES handling for int/str cfg values
- Keep SupCon weight multiplication ONLY inside make_loss (avoid double-scaling)
"""

import os
import re
import random
import argparse
import numpy as np
import torch

from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
from config import cfg

# Optional import; training should gracefully continue if missing.
try:
    from loss.supcon_loss import SupConLoss  # requires your supcon_loss.py
except Exception:
    SupConLoss = None  # graceful fallback; do_train should handle None


# ==============================
# Device helpers
# ==============================
def _infer_device_str() -> str:
    """Pick device string by availability: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _optimizer_to(optim: torch.optim.Optimizer, device: torch.device):
    """Move all optimizer state tensors (e.g., momentum buffers) to target device.
    Fix 'found mps:0 and cpu' when resuming on MPS/CPU.
    """
    if optim is None:
        return
    for st in optim.state.values():
        for k, v in list(st.items()):
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device, non_blocking=False)


# ==============================
# Reproducibility
# ==============================
def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds. If deterministic=True, disable CUDNN autotune."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Determinism vs speed: choose one coherently.
    torch.backends.cudnn.deterministic = bool(deterministic)
    # If we want determinism, benchmark must be False; otherwise it breaks reproducibility.
    torch.backends.cudnn.benchmark = False if deterministic else True


# ==============================
# Main
# ==============================
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="", type=str, help="path to config file")
    parser.add_argument("opts", nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    # Merge config
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Seed & determinism: prioritize reproducibility by default
    seed = int(getattr(cfg.SOLVER, "SEED", 1234))
    set_seed(seed, deterministic=True)

    # Distributed
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    # Output dir
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Logger
    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        try:
            with open(args.config_file, 'r') as cf:
                config_str = "\n" + cf.read()
                logger.info(config_str)
        except Exception as e:
            logger.info(f"[WARN] Failed to read config file for logging: {e}")

    logger.info("Running with config:\n{}".format(cfg))

    # NCCL init for distributed
    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # CUDA_VISIBLE_DEVICES: coerce int/str; avoid tuple-like strings ('0')
    dev_id = getattr(cfg.MODEL, "DEVICE_ID", None)
    if dev_id is not None:
        if isinstance(dev_id, (list, tuple)):
            dev_str = ",".join(str(x) for x in dev_id)
        else:
            dev_str = str(dev_id).strip()
            # Normalize legacy forms like "('0')" -> "0"
            dev_str = dev_str.replace("(", "").replace(")", "").replace("'", "").replace('"', "").replace(" ", "")
        if dev_str:
            os.environ['CUDA_VISIBLE_DEVICES'] = dev_str
            logger.info(f"[Env] CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Dataloaders
    (train_loader,
     train_loader_normal,
     val_loader,
     num_query,
     num_classes,
     camera_num,
     view_num) = make_dataloader(cfg)

    # Model
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # Loss ctor (ID/Triplet & center)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # Optimizer & Scheduler
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    # ==============================
    # Seamless resume (weights + training state)
    # ==============================
    start_epoch = 1
    if getattr(cfg.MODEL, "PRETRAIN_CHOICE", "imagenet") == "resume":
        weight_path = getattr(cfg.MODEL, "PRETRAIN_PATH", "")
        if weight_path and os.path.isfile(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception:
                model.load_state_dict(state_dict, strict=False)

            # Infer epoch index k from filename transformer_k.pth
            m = re.search(r"transformer_(\d+)\.pth$", os.path.basename(weight_path))
            if m:
                start_epoch = int(m.group(1)) + 1

            # Try to load optimizer/scheduler/scaler/epoch
            state_path = weight_path.replace("transformer_", "state_")
            if os.path.isfile(state_path):
                state_blob = torch.load(state_path, map_location="cpu")
                if isinstance(state_blob, dict):
                    # epoch in state takes precedence
                    if "epoch" in state_blob:
                        start_epoch = int(state_blob["epoch"]) + 1
                    if "optimizer" in state_blob and state_blob["optimizer"] is not None:
                        try:
                            optimizer.load_state_dict(state_blob["optimizer"])
                        except Exception as e:
                            logger.info(f"[resume] optimizer.load_state_dict failed: {e}")
                    if "scheduler" in state_blob and state_blob["scheduler"] is not None:
                        try:
                            scheduler.load_state_dict(state_blob["scheduler"])
                        except Exception as e:
                            logger.info(f"[resume] scheduler.load_state_dict failed: {e}")
                    logger.info(f"[resume] Loaded {state_path}; start_epoch={start_epoch}")
            else:
                logger.info(f"[resume] State file not found for {weight_path}; fallback start_epoch={start_epoch}")

            # Ensure optimizer states live on the same device as the model
            dev_str = _infer_device_str()
            device = torch.device(dev_str)
            _optimizer_to(optimizer, device)
            if optimizer_center is not None:
                _optimizer_to(optimizer_center, device)
        else:
            logger.info(f"[resume] PRETRAIN_PATH not found: {weight_path}; start_epoch remains {start_epoch}")

    # ==============================
    # SupCon integration (keys aligned with make_loss/yml)
    #   Expected yml:
    #     LOSS:
    #       SUPCON:
    #         ENABLE: True/False
    #         T: 0.07
    #         W: 0.25
    #   NOTE:
    #     * SupCon weight is applied INSIDE make_loss; do not multiply again in do_train.
    # ==============================
    use_supcon = False
    supcon_criterion = None

    has_supcon = hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "SUPCON")
    supcon_enable = bool(getattr(cfg.LOSS.SUPCON, "ENABLE", False)) if has_supcon else False
    supcon_temp   = float(getattr(cfg.LOSS.SUPCON, "T", 0.07))       if has_supcon else 0.07
    supcon_weight = float(getattr(cfg.LOSS.SUPCON, "W", 0.2))        if has_supcon else 0.2  # for logging only

    if supcon_enable and SupConLoss is not None:
        use_supcon = True
        # Construct criterion; if your SupConLoss supports extra args (pos_rule/cam_aware), wire here.
        try:
            pos_rule = getattr(cfg.LOSS.SUPCON, "POS_RULE", "class")
            cam_aware = bool(getattr(cfg.LOSS.SUPCON, "CAM_AWARE", False))
            supcon_criterion = SupConLoss(temperature=supcon_temp, pos_rule=pos_rule, cam_aware=cam_aware)
            logger.info(f"[SupCon] ENABLED: temperature={supcon_temp}, weight={supcon_weight}, "
                        f"pos_rule={pos_rule}, cam_aware={cam_aware}")
        except TypeError:
            # Backward-compatible constructor
            supcon_criterion = SupConLoss(temperature=supcon_temp)
            logger.info(f"[SupCon] ENABLED: temperature={supcon_temp}, weight={supcon_weight}")
    elif supcon_enable and SupConLoss is None:
        logger.info("[SupCon] ENABLE requested but SupConLoss is not importable. Falling back to DISABLED.")
        use_supcon = False
        supcon_criterion = None
    else:
        logger.info("[SupCon] DISABLED")

    # ==============================
    # Launch Training
    # (Do NOT multiply supcon_weight again in do_train. Weighting is handled inside make_loss.)
    # ==============================
    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query,
        args.local_rank,
        start_epoch=start_epoch,

        # SupCon controls/objects for use inside training loop (logging/forward plumbing)
        use_supcon=use_supcon,
        supcon_criterion=supcon_criterion,
        supcon_weight=supcon_weight  # visibility/logging only; weighting is applied in make_loss
    )
