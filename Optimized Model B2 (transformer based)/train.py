from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg

# =========================================================================================
# [Modified by] Hang Zhang (张航)
# [Date] 2025-09-12
# [Content] SupCon integration (part 1/2) for training pipeline:
#   - Import SupConLoss
#   - Build supcon_criterion (temperature from cfg, default 0.07)
#   - Read supcon_weight from cfg (default 0.2)
#   - Pass {use_supcon, supcon_criterion, supcon_weight} to do_train(...)
#   - No breaking changes to existing CE/Triplet setup
# =========================================================================================
try:
    from loss.supcon_loss import SupConLoss  # requires your supcon_loss.py
except Exception:
    SupConLoss = None  # graceful fallback; do_train should handle None

# ==============================
# [2025-08-30 | Hang Zhang] Utils added for seamless resume on CUDA/MPS/CPU
# ==============================
def _infer_device_str() -> str:
    """Pick device string by availability: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _optimizer_to(optim: torch.optim.Optimizer, device: torch.device):
    """Move all optimizer state tensors (e.g. momentum buffers) to target device.
    [2025-08-30 | Hang Zhang] Fix 'found mps:0 and cpu' when resuming on MPS.
    """
    if optim is None:
        return
    for st in optim.state.values():
        for k, v in list(st.items()):
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device, non_blocking=False)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    # ============================================================
    # [2025-08-30 | Hang Zhang] Seamless resume: load weights + state
    # - Supports CUDA/MPS/CPU
    # - Restores optimizer/scheduler states
    # - Fixes optimizer state device to match model device
    # - Computes start_epoch (k+1) from state or filename
    # ============================================================
    start_epoch = 1
    if getattr(cfg.MODEL, "PRETRAIN_CHOICE", "imagenet") == "resume":
        weight_path = getattr(cfg.MODEL, "PRETRAIN_PATH", "")
        if weight_path and os.path.isfile(weight_path):
            # Load model weights (strict=False to be robust to head shapes)
            state_dict = torch.load(weight_path, map_location="cpu")
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception:
                model.load_state_dict(state_dict, strict=False)

            # Infer epoch index k from filename transformer_k.pth
            import re
            m = re.search(r"transformer_(\d+)\.pth$", os.path.basename(weight_path))
            if m:
                start_epoch = int(m.group(1)) + 1

            # Try to load training state (optimizer/scheduler/scaler/epoch)
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

    # =====================================================================================
    # [Modified by] Hang Zhang (张航)
    # [Date] 2025-09-12
    # [Content] SupCon integration (part 2/2):
    #   - Build supcon_criterion and read weight from cfg
    #   - Log enabling message
    #   - Pass into do_train(...) without breaking existing signature (add kwargs)
    #   Expected cfg fields (with defaults if missing):
    #       cfg.LOSS.SUPCON.ENABLE: bool
    #       cfg.LOSS.SUPCON.TEMPERATURE: float = 0.07
    #       cfg.LOSS.SUPCON.WEIGHT: float = 0.2
    #   Notes:
    #       * do_train(...) should be updated to accept these kwargs:
    #           use_supcon: bool
    #           supcon_criterion: callable or None
    #           supcon_weight: float
    #       * In forward pass, model returns z_supcon when training & enabled.
    # =====================================================================================
    use_supcon = False
    supcon_criterion = None
    supcon_weight = 0.2  # default

    # Robustly read cfg flags
    has_loss = hasattr(cfg, "LOSS")
    has_supcon = has_loss and hasattr(cfg.LOSS, "SUPCON")
    supcon_enable = bool(getattr(cfg.LOSS.SUPCON, "ENABLE", False)) if has_supcon else False
    supcon_temp = float(getattr(cfg.LOSS.SUPCON, "TEMPERATURE", 0.07)) if has_supcon else 0.07
    supcon_weight = float(getattr(cfg.LOSS.SUPCON, "WEIGHT", 0.2)) if has_supcon else 0.2

    if supcon_enable and SupConLoss is not None:
        use_supcon = True
        supcon_criterion = SupConLoss(temperature=supcon_temp)
        logger.info(f"[SupCon] ENABLED: temperature={supcon_temp}, weight={supcon_weight}")
    elif supcon_enable and SupConLoss is None:
        logger.info("[SupCon] ENABLE requested but SupConLoss is not importable. Falling back to disabled.")
        use_supcon = False
        supcon_criterion = None
    else:
        logger.info("[SupCon] DISABLED")

    # === run training ===
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
        num_query, args.local_rank,
        start_epoch=start_epoch,  # [2025-08-30 | Hang Zhang] pass start_epoch for seamless resume

        # -------------------------------------------------------------------------
        # [Modified by] Hang Zhang (张航) | [Date] 2025-09-12
        # [Content] Pass SupCon controls/objects for use inside training loop
        #           (Update processor/do_train to accept these kwargs)
        # -------------------------------------------------------------------------
        use_supcon=use_supcon,
        supcon_criterion=supcon_criterion,
        supcon_weight=supcon_weight
    )
