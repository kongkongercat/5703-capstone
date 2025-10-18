# encoding: utf-8
"""
ReID Training Entrypoint
"""

# ==========================================
# Change Log
# ==========================================
# [Original | liaoxingyu] Initial ReID baseline entry (sherlockliao01@gmail.com)
# [2025-09-11 | Zhang Hang] Added support for Supervised Contrastive Loss (SupConLoss)
#                            and integrated cfg.LOSS.SUPCON.ENABLE / W / T into total loss.
# [2025-09-14 | Zhang Hang] Aligned SupCon cfg keys with make_loss/yml (T/W),
#                            improved seed/CUDNN handling and CUDA_VISIBLE_DEVICES parsing.
# [2025-09-14 | Zhang Hang] Ensured SupCon weight scaling occurs only in make_loss.
# [2025-10-17 | Zhang Hang] Added unified feature-source summary logging
#                            (CE / Triplet / TripletX / SupCon) after make_model–make_loss.
#                            Added '--config' CLI alias for consistency.
# [2025-10-17 | Zhang Hang]  Removed redundant SupConLoss instantiation from entry script.
#                            SupConLoss is now built exclusively inside make_loss.py.
#                            Entry only logs SupCon ENABLED/DISABLED for clarity.
# ==========================================

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


# ==============================
# Device helpers
# ==============================
def _infer_device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _optimizer_to(optim: torch.optim.Optimizer, device: torch.device):
    """Move optimizer state tensors to target device."""
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


# ==============================
# Main
# ==============================
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--config_file", default="", type=str, help="path to config file")
    parser.add_argument("--config", dest="config_file", default="", type=str,
                        help="path to config file (alias)")
    parser.add_argument("opts", nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    # Merge config
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Seed & determinism
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
    logger.info(f"Saving model in the path: {cfg.OUTPUT_DIR}")
    logger.info(args)

    if args.config_file:
        logger.info(f"Loaded configuration file {args.config_file}")
        try:
            with open(args.config_file, 'r') as cf:
                logger.info("\n" + cf.read())
        except Exception as e:
            logger.info(f"[WARN] Failed to read config file for logging: {e}")

    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    dev_id = getattr(cfg.MODEL, "DEVICE_ID", None)
    if dev_id is not None:
        if isinstance(dev_id, (list, tuple)):
            dev_str = ",".join(str(x) for x in dev_id)
        else:
            dev_str = str(dev_id).replace("(", "").replace(")", "").replace("'", "").replace('"', "").replace(" ", "")
        if dev_str:
            os.environ["CUDA_VISIBLE_DEVICES"] = dev_str
            logger.info(f"[Env] CUDA_VISIBLE_DEVICES set to: {dev_str}")

    # =========================
    # Build components
    # =========================
    (train_loader, train_loader_normal, val_loader,
     num_query, num_classes, camera_num, view_num) = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # ==========================================
    # Unified feature-source summary logging
    # ==========================================
    def _safe_get(node, key, default):
        try:
            return getattr(node, key)
        except Exception:
            return default

    try:
        ce_src   = _safe_get(cfg.LOSS.CE, "FEAT_SRC", "bnneck (fixed)")
        tri_src  = _safe_get(cfg.LOSS.TRIPLET, "FEAT_SRC", "pre_bn")
        tx_src   = _safe_get(cfg.LOSS.TRIPLETX, "FEAT_SRC", "pre_bn")
        sup_src  = _safe_get(cfg.LOSS.SUPCON, "FEAT_SRC", "bnneck (model decides)")
        logger.info(f"[init][feat_src] CE={ce_src} | Triplet={tri_src} | "
                    f"TripletX={tx_src} | SupCon={sup_src}")
    except Exception as e:
        logger.info(f"[init][feat_src] Unable to summarize feature sources: {e}")

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    # =========================
    # Resume from checkpoint
    # =========================
    start_epoch = 1
    if getattr(cfg.MODEL, "PRETRAIN_CHOICE", "imagenet") == "resume":
        weight_path = getattr(cfg.MODEL, "PRETRAIN_PATH", "")
        if weight_path and os.path.isfile(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception:
                model.load_state_dict(state_dict, strict=False)
            m = re.search(r"transformer_(\d+)\.pth$", os.path.basename(weight_path))
            if m:
                start_epoch = int(m.group(1)) + 1
            state_path = weight_path.replace("transformer_", "state_")
            if os.path.isfile(state_path):
                state_blob = torch.load(state_path, map_location="cpu")
                if isinstance(state_blob, dict):
                    if "epoch" in state_blob:
                        start_epoch = int(state_blob["epoch"]) + 1
                    if "optimizer" in state_blob:
                        try:
                            optimizer.load_state_dict(state_blob["optimizer"])
                        except Exception as e:
                            logger.info(f"[resume] optimizer.load_state_dict failed: {e}")
                    if "scheduler" in state_blob:
                        try:
                            scheduler.load_state_dict(state_blob["scheduler"])
                        except Exception as e:
                            logger.info(f"[resume] scheduler.load_state_dict failed: {e}")
                    logger.info(f"[resume] Loaded {state_path}; start_epoch={start_epoch}")
            device = torch.device(_infer_device_str())
            _optimizer_to(optimizer, device)
            if optimizer_center:
                _optimizer_to(optimizer_center, device)
        else:
            logger.info(f"[resume] PRETRAIN_PATH not found: {weight_path}")

    # =========================
    # SupCon info log (delegated)
    # =========================
    has_supcon = hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "SUPCON")
    supcon_enable = bool(getattr(cfg.LOSS.SUPCON, "ENABLE", False)) if has_supcon else False

    if supcon_enable:
        logger.info("[SupCon] ENABLED (constructed in make_loss.py; entry will not rebuild).")
    else:
        logger.info("[SupCon] DISABLED")

    # =========================
    # Launch training
    # =========================
    do_train(cfg, model, center_criterion,
             train_loader, val_loader,
             optimizer, optimizer_center, scheduler,
             loss_func, num_query, args.local_rank,
             start_epoch=start_epoch)
