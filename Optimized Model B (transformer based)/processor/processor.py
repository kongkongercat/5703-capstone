#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Change Log (processor/processor.py)
# -----------------------------------------------------------------------------
# [2025-08-30 | Hang Zhang] Added start_epoch/resume & AMP scaler plumbing.
# [2025-09-12 | Hang Zhang] SupCon integration in trainer (external add).
# [2025-10-14 | Hang Zhang] Phased-loss integration (B2_phased_loss)  # [NEW]
#   - Pass current `epoch` into loss_func(...) to enable phased weights.  # [NEW]
#   - Route `z_supcon` into loss_func(...) and REMOVE external SupCon add  # [NEW]
#     (SupCon weighting now handled inside make_loss by dynamic schedule).  # [NEW]
#   - Simplify logs (no separate "SupCon+" piece) to avoid double counting. # [NEW]
# [2025-10-14 | Hang Zhang] Epoch-aware PK-sampler K: rebuild train_loader at phase boundaries.
#                           from datasets import make_dataloader
# =============================================================================

import os
import time
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Tuple

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from datasets import make_dataloader

def _pick_device_str(cfg) -> str:
    """
    Pick device string with a fallback order: cfg.MODEL.DEVICE -> cuda -> mps -> cpu.
    """
    # If user specifies MODEL.DEVICE, honor it (and check availability).
    dev = getattr(cfg.MODEL, "DEVICE", None)
    if isinstance(dev, str):
        dev = dev.lower()
        if dev == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "mps":
            return "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
        return "cpu"

    # Auto detection when not explicitly set
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def _validate_once(
    cfg,
    model: nn.Module,
    val_loader,
    evaluator: R1_mAP_eval,
    device: torch.device,
    use_cuda: bool,
    epoch: int,
) -> Tuple[float, float, float]:
    """
    Run a single validation pass and log metrics.
    """
    logger = logging.getLogger("transreid.train")
    model.eval()
    evaluator.reset()

    for _, (img, pid, camid, camids, target_view, _) in enumerate(val_loader):
        img = img.to(device, non_blocking=use_cuda)
        camids = camids.to(device, non_blocking=use_cuda)
        target_view = target_view.to(device, non_blocking=use_cuda)
        feat = model(img, cam_label=camids, view_label=target_view)
        evaluator.update((feat, pid, camid))

    cmc, mAP, *_ = evaluator.compute()
    logger.info(f"Validation Results - Epoch: {epoch}")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    # return Rank-1, Rank-5, mAP
    return float(cmc[0]), float(cmc[4]), float(mAP)

# ===== Helpers for phased PK-sampler K ======================================  # [NEW]
def _phased_pk_enabled(cfg) -> bool:
    """
    [NEW] Guard for epoch-aware PK-sampler K.
    Works only when BOTH switches are ON:
      - cfg.LOSS.PHASED.ENABLE == True
      - cfg.DATALOADER.PHASED.ENABLE == True
    """
    return (
        hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "PHASED")
        and bool(getattr(cfg.LOSS.PHASED, "ENABLE", False))
        and hasattr(cfg, "DATALOADER") and hasattr(cfg.DATALOADER, "PHASED")
        and bool(getattr(cfg.DATALOADER.PHASED, "ENABLE", False))
    )


def _tripletx_stage(epoch: int) -> bool:
    """[NEW] TripletX active only in Stage A (epoch <= 30)."""
    return int(epoch) <= 30


def _desired_pk_k(epoch: int, cfg) -> int:
    """
    [NEW] Return desired K for PK-sampler given epoch & config.
    A-stage (TripletX): K_WHEN_TRIPLETX, else K_OTHER.
    Falls back to cfg.DATALOADER.NUM_INSTANCE when phased is disabled.
    """
    if not _phased_pk_enabled(cfg):
        return int(getattr(cfg.DATALOADER, "NUM_INSTANCE", 4))
    return int(
        getattr(
            cfg.DATALOADER.PHASED,
            "K_WHEN_TRIPLETX" if _tripletx_stage(epoch) else "K_OTHER",
            8 if _tripletx_stage(epoch) else 4,
        )
    )


def do_train(
    cfg,
    model: nn.Module,
    center_criterion: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    optimizer_center: torch.optim.Optimizer,
    scheduler,  # may be any scheduler object with step()/state_dict()
    loss_func,
    num_query: int,
    local_rank: int,
    start_epoch: int = 1,  # [2025-08-30 | Hang Zhang] Added start_epoch for seamless resume

    # =====================================================================================
    # (Kept for backward-compatibility) SupCon flags from entrypoint.                    # [CHG]
    # NOTE: SupCon weighting is now handled INSIDE make_loss via phased schedule.        # [NEW]
    # =====================================================================================
    use_supcon: bool = False,
    supcon_criterion=None,
    supcon_weight: float = 0.2,
):
    """
    Main training loop.

    Notes (seamless resume):
    - The outer script/train.py is responsible for restoring optimizer/scheduler/scaler and
      passing correct `start_epoch` (k+1) when resuming from transformer_k.pth and state_k.pth.
    - Here we start from `start_epoch` and save both model weights and training state at each epoch.
    - SupCon handling:
        * z_supcon is forwarded to loss_func(..., z_supcon=...) and weighted INSIDE make_loss
          according to the current epoch (phased schedule).                                           # [NEW]
    - PK-sampler K scheduling:
        * If enabled (LOSS.PHASED & DATALOADER.PHASED), rebuild train_loader at epoch boundaries:
          - A-stage (epoch <= 30): K=8
          - B/C-stage (epoch > 30): K=4
    """

    logger = logging.getLogger("transreid.train")

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    epochs_this_run = cfg.SOLVER.MAX_EPOCHS

    # Device selection
    device_str = _pick_device_str(cfg)
    device = torch.device(device_str)
    use_cuda = (device_str == "cuda")

    logger.info(f"start training (device={device_str})")
    if use_supcon:
        # visibility only; real weighting is inside make_loss
        logger.info(f"[SupCon] training flag=True (note: weighted inside make_loss)")

    # Move model to device
    model.to(device)

    # DDP (if enabled)
    if use_cuda and torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    # meters & evaluator
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    # AMP scaler (CUDA only). On MPS/CPU this is disabled automatically.
    # [2025-08-30 | Hang Zhang] AMP scaler kept for state saving during seamless resume.
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    # Track current K (for phased PK switching)                                         # [NEW]
    current_K = int(getattr(cfg.DATALOADER, "NUM_INSTANCE", 4))                         # [NEW]

    # =========================
    # [2025-08-30 | Hang Zhang] Training loop now starts from `start_epoch`
    # =========================
    for epoch in range(start_epoch, start_epoch + epochs_this_run):

        # ----- Epoch-aware PK-sampler K switch (rebuild train_loader) -----           # [NEW]
        desired_K = _desired_pk_k(epoch, cfg)
        if desired_K != current_K:
            logging.getLogger("transreid.train").info(
                f"[PK-sampler] Rebuilding train_loader: NUM_INSTANCE {current_K} → {desired_K} (epoch {epoch})"
            )
            cfg.defrost()
            cfg.DATALOADER.NUM_INSTANCE = desired_K
            # Rebuild only train_loader; keep val_loader unchanged
            (train_loader,
             _train_loader_normal,
             _val_loader,
             _num_query,
             _num_classes,
             _camera_num,
             _view_num) = make_dataloader(cfg)
            cfg.freeze()
            current_K = desired_K

        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        # Step the epoch-based scheduler (if applicable)
        try:
            scheduler.step(epoch)
        except TypeError:
            try:
                scheduler.step()
            except Exception:
                pass

        model.train()

        for n_iter, batch in enumerate(train_loader):
            # Support tuples coming from your dataset:
            # (img, vid, target_cam, camids, target_view, _) or (img, vid, target_cam, target_view)
            if len(batch) == 6:
                img, vid, target_cam, camids, target_view, _ = batch
            elif len(batch) == 5:
                img, vid, target_cam, target_view, _ = batch
                camids = target_cam
            else:
                # Minimal fallback: (img, vid, target_cam, target_view)
                img, vid, target_cam, target_view = batch
                camids = target_cam

            optimizer.zero_grad(set_to_none=True)
            if optimizer_center is not None:
                optimizer_center.zero_grad(set_to_none=True)

            img = img.to(device, non_blocking=use_cuda)
            target = vid.to(device, non_blocking=use_cuda)
            target_cam = target_cam.to(device, non_blocking=use_cuda)
            target_view = target_view.to(device, non_blocking=use_cuda)
            camids = camids.to(device, non_blocking=use_cuda)

            # ---------------------------------------------------------------------------------
            # Forward & total loss (SupCon + Triplet(X) handled INSIDE make_loss).            # [CHG][NEW]
            # make_loss will:
            #   * choose TripletX (A-stage) or Triplet (B/C) based on `epoch`
            #   * apply dynamic w_metric and w_sup (SupCon) per epoch
            #   * safely fallback when z_supcon is None (use feat)
            # ---------------------------------------------------------------------------------
            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = model(img, target, cam_label=target_cam, view_label=target_view)

                # Init containers
                score = None
                feat = None
                z_supcon = None

                # Unpack by type/length safely
                if isinstance(out, (tuple, list)):
                    if len(out) == 3:
                        maybe_scores, maybe_feats, z_supcon = out
                        score, feat = maybe_scores, maybe_feats
                    elif len(out) == 2:
                        score, feat = out
                    else:
                        score = out[0]
                        feat = out[1] if len(out) > 1 else None
                        z_supcon = out[2] if len(out) > 2 else None
                else:
                    score = out
                    feat = out  # best-effort fallback

                # === The ONLY loss call (includes SupCon & phased weights) ===               # [NEW]
                loss = loss_func(
                    score,
                    feat,
                    target,
                    camids=camids,       # TripletX/SupCon need camera ids
                    z_supcon=z_supcon,   # may be None; make_loss will fallback to feat
                    epoch=epoch,         # critical: enable phased scheduling
                )

            # Backward & step with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Center loss optimizer (if enabled)
            if ("center" in cfg.MODEL.METRIC_LOSS_TYPE) and (center_criterion is not None) and (optimizer_center is not None):
                for p in center_criterion.parameters():
                    p.grad.data *= (1.0 / max(cfg.SOLVER.CENTER_LOSS_WEIGHT, 1e-12))
                scaler.step(optimizer_center)
                scaler.update()

            # Accuracy (use first branch if multi-head)
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(float(loss.item()), img.shape[0])
            acc_meter.update(float(acc), 1)

            if use_cuda:
                torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                # best-effort LR fetch
                try:
                    lr_val = optimizer.param_groups[0]["lr"]
                except Exception:
                    lr_val = 0.0

                # Unified log (SupCon already included inside total loss).                    # [CHG]
                logging.getLogger("transreid.train").info(
                    "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, LR: {:.2e}".format(
                        epoch, (n_iter + 1), len(train_loader),
                        loss_meter.avg, acc_meter.avg, lr_val
                    )
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / max((n_iter + 1), 1)
        if not cfg.MODEL.DIST_TRAIN:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / max(time_per_batch, 1e-12)
                )
            )

        # =========================
        # Save both model weights and training state for seamless resume.
        # =========================
        if (epoch % checkpoint_period == 0) or (epoch == start_epoch + epochs_this_run - 1):
            weight_path = os.path.join(cfg.OUTPUT_DIR, f"transformer_{epoch}.pth")
            state_path = os.path.join(cfg.OUTPUT_DIR, f"state_{epoch}.pth")

            def _save_all():
                # Save model weights
                if isinstance(model, nn.parallel.DistributedDataParallel):
                    torch.save(model.module.state_dict(), weight_path)
                else:
                    torch.save(model.state_dict(), weight_path)

                # Save training state (optimizer/scheduler/scaler/epoch)
                state_blob = {
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                }
                torch.save(state_blob, state_path)

            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    _save_all()
            else:
                _save_all()

        # =========================
        # Evaluation each `eval_period`
        # =========================
        if (epoch % eval_period == 0) or (epoch == start_epoch + epochs_this_run - 1):
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    _validate_once(cfg, model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model,
                                   val_loader, evaluator, device, use_cuda, epoch)
            else:
                _validate_once(cfg, model, val_loader, evaluator, device, use_cuda, epoch)


# =========================
# do_inference for test.py compatibility
# =========================
@torch.no_grad()
def do_inference(
    cfg,
    model: nn.Module,
    val_loader,
    num_query: int,
) -> Tuple[float, float]:
    """
    Run inference (evaluation) on the validation/test loader.

    Returns:
        (rank1, rank5)
    """
    # Auto-select device consistent with training
    dev_str = _pick_device_str(cfg)
    device = torch.device(dev_str)
    use_cuda = (dev_str == "cuda")

    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    # Multi-GPU DP only on CUDA
    if use_cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference")
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()

    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        img = img.to(device, non_blocking=use_cuda)
        camids = camids.to(device, non_blocking=use_cuda)
        target_view = target_view.to(device, non_blocking=use_cuda)

        feat = model(img, cam_label=camids, view_label=target_view)
        evaluator.update((feat, pid, camid))
        img_path_list.extend(imgpath)

    cmc, mAP, *_ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    # Return Rank-1 and Rank-5 for compatibility with some callers
    return float(cmc[0]), float(cmc[4])
