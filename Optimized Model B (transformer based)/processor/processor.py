#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Tuple

from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval


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
    # [Modified by] Hang Zhang (张航)
    # [Date] 2025-09-12
    # [Content] SupCon integration (accept kwargs from train.py)
    #   - use_supcon: bool
    #   - supcon_criterion: callable or None (features, labels, camids) -> loss
    #   - supcon_weight: float
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
    - SupCon integration:
        * model.forward(...) may return z_supcon in training mode when enabled in cfg.
        * This function will detect and add SupCon loss if (use_supcon and supcon_criterion is not None).
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
        logger.info(f"[SupCon] training enabled (weight={supcon_weight})")

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

    # =========================
    # [2025-08-30 | Hang Zhang] Training loop now starts from `start_epoch`
    # =========================
    for epoch in range(start_epoch, start_epoch + epochs_this_run):

        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        # Step the epoch-based scheduler (if applicable)
        try:
            # some schedulers expect step(epoch), others step() every iteration
            scheduler.step(epoch)
        except TypeError:
            # fallback: many schedulers work with step() per epoch
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
            # [Modified by] Hang Zhang (张航) | [Date] 2025-09-12
            # [Content] Forward & loss with SupCon (auto-unpack outputs):
            #   * Backbone/build_transformer:
            #       - (score, feat) or (score, feat, z_supcon)
            #   * build_transformer_local (JPM):
            #       - (scores, feats) or (scores, feats, z_supcon)
            #   Then:
            #       loss_total = loss_func(score, feat, target, target_cam)
            #       + supcon_weight * SupConLoss(z_supcon, target, camids)   [when enabled]
            # ---------------------------------------------------------------------------------
            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = model(img, target, cam_label=target_cam, view_label=target_view)

                # Init containers
                score = None
                feat = None
                z_supcon = None

                # Unpack by type/length safely
                if isinstance(out, tuple) or isinstance(out, list):
                    # JPM/Local branch often returns multiple heads
                    if len(out) == 3:
                        # could be (score, feat, z_supcon) or (scores, feats, z_supcon)
                        maybe_scores, maybe_feats, z_supcon = out
                        score = maybe_scores
                        feat = maybe_feats
                    elif len(out) == 2:
                        score, feat = out
                    else:
                        # unexpected length, best effort
                        score = out[0]
                        feat = out[1] if len(out) > 1 else None
                        z_supcon = out[2] if len(out) > 2 else None
                else:
                    # not expected, but keep compatibility
                    score = out

                # Base loss (CE + Triplet + Center inside make_loss)
                base_loss = loss_func(score, feat, target, target_cam)

                # Optional SupCon loss
                if use_supcon and (supcon_criterion is not None) and (z_supcon is not None):
                    supcon_loss = supcon_criterion(z_supcon, target, camids)
                    loss = base_loss + supcon_weight * supcon_loss
                else:
                    loss = base_loss

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
                # Log SupCon piece when enabled
                if use_supcon and (supcon_criterion is not None) and (z_supcon is not None):
                    try:
                        supcon_val = float((supcon_weight * supcon_loss).item())
                    except Exception:
                        supcon_val = 0.0
                    logging.getLogger("transreid.train").info(
                        "Epoch[{}] Iter[{}/{}] Loss: {:.3f} (SupCon+: {:.3f}), Acc: {:.3f}, LR: {:.2e}".format(
                            epoch, (n_iter + 1), len(train_loader),
                            loss_meter.avg, supcon_val, acc_meter.avg, lr_val
                        )
                    )
                else:
                    logging.getLogger("transreid.train").info(
                        "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
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
        # [2025-08-30 | Hang Zhang] Save both model weights and training state for seamless resume.
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
# [2025-08-30 | Hang Zhang] Added do_inference for test.py compatibility
# - Auto-detect device (cuda > mps > cpu)
# - Optional DataParallel on multi-GPU CUDA
# - Compute mAP and CMC (Rank-1/5/10)
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
