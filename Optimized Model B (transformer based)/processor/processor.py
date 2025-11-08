# processor/processor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# Change Log (processor/processor.py)
# -----------------------------------------------------------------------------
# [2025-08-30 | Hang Zhang] Added start_epoch/resume & AMP scaler plumbing.
# [2025-09-12 | Hang Zhang] SupCon integration in trainer (external add).
# [2025-10-14 | Hang Zhang] Phased-loss integration (B2_phased_loss)
#   - Pass current `epoch` into loss_func(...) to enable phased weights.
#   - Route `z_supcon` into loss_func(...) and REMOVE external SupCon add
#     (SupCon weighting now handled inside make_loss by dynamic schedule).
# [2025-10-14 | Hang Zhang] Epoch-aware PK-sampler K: rebuild train_loader at phase boundaries.
#                           from datasets import make_dataloader
# [2025-10-19 | Hang Zhang]  **Make phased boundary configurable & random-sampler guard**
#   - _tripletx_stage(epoch,cfg) reads cfg.LOSS.PHASED.TRIPLETX_END instead of hard-coded 30.
#   - When DATALOADER.SAMPLER=='random', skip PK-K rebuild (no P×K semantics).
# [2025-10-19 | Hang Zhang]  **Pass BNNeck feature into loss_func & safe acc**
#   - Forward path unpacks (score, feat, feat_bn, z_supcon) and passes feat_bn to loss_func.
#   - When score is None (e.g., SupCon-only), accuracy metric is safely set to 0.
# [2025-10-22 | Hang Zhang] **Brightness-aware training (minimal change)**
#   - Compute per-batch `dark_mask` and `dark_weight` in trainer (no Dataset/Collate change).
#   - Pass `dark_mask` / `dark_weight` to loss_func(...) as OPTIONAL kwargs.
#   - Keep loss composition inside make_loss; trainer remains thin.
# [2025-10-23 | Hang Zhang] **Fix brightness domain**
#   - Before BT.709 brightness, denormalize images using cfg.INPUT.PIXEL_MEAN/STD
#     to ~[0,1] domain; threshold 0.35 is now consistent with image domain.
# [2025-10-23 | Hang Zhang] **Brightness toggle via cfg**
#   - Guard brightness-aware training by cfg.LOSS.BRIGHTNESS.ENABLE (default False).
#   - Threshold/softness configurable via THRESH (default 0.35), K (default 0.08).
# [2025-11-09 | Hang Zhang] **Plan B: decouple PK-switch from LOSS.PHASED**
#   - _phased_pk_enabled() now only requires DATALOADER.PHASED.ENABLE=True
#     to support "loss-activated K" without epoch-phased loss.
# [2025-11-09 | Hang Zhang] **Finalize TX/SupCon-aware K switching**
#   - Replace _tripletx_stage(): decide by actual phase enablement (METRIC_SEQ) and cfg.ENABLE
#   - Add _supcon_eff_w(): effective SupCon weight (W_SUP_SPEC -> DECAY_* -> static W) gated by ENABLE
#   - Replace _desired_pk_k(): TX-active or SupCon(effW>0) -> expand K; else K_OTHER
#   - Add optional debug print for K decision per epoch.
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
    dev = getattr(cfg.MODEL, "DEVICE", None)
    if isinstance(dev, str):
        dev = dev.lower()
        if dev == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "mps":
            return "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
        return "cpu"

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

    return float(cmc[0]), float(cmc[4]), float(mAP)


# ========================= PK-sampler helpers ===============================

def _phased_pk_enabled(cfg) -> bool:
    """
    Enable epoch-begin PK (P×K) K-switching when dataloader-phased is ON.
    Plan B: only DATALOADER.PHASED.ENABLE matters; LOSS.PHASED.ENABLE is not required.
    """
    return bool(
        hasattr(cfg, "DATALOADER")
        and hasattr(cfg.DATALOADER, "PHASED")
        and getattr(cfg.DATALOADER.PHASED, "ENABLE", False)
    )


def _sampler_is_random(cfg) -> bool:
    """Whether SAMPLER is 'random' (SupCon-only / SSL)."""
    return str(getattr(cfg.DATALOADER, "SAMPLER", "softmax_triplet")).lower() == "random"


def _phased_pk_other(cfg) -> int:
    """Default fallback K when no special loss is active."""
    try:
        return int(getattr(getattr(cfg.DATALOADER, "PHASED", object()), "K_OTHER", 4))
    except Exception:
        return int(getattr(cfg.DATALOADER, "NUM_INSTANCE", 4))


# ---------------------- NEW: phase/index helpers ----------------------------

def _phase_index_local(e: int, boundaries) -> int:
    """Half-open segments: [0,b0), [b0,b1), ..., [b_{n-1}, +inf)."""
    for i, b in enumerate(list(boundaries or [])):
        if int(e) < int(b):
            return i
    return len(list(boundaries or []))


# ---------------------- NEW: TripletX phase detector ------------------------

def _tripletx_stage(epoch: int, cfg) -> bool:
    """
    TripletX considered active ONLY when it is actually enabled for the current phase:
      - If LOSS.PHASED.ENABLE=True: current phase in METRIC_SEQ must be 'tripletx'
        AND LOSS.TRIPLETX.ENABLE=True AND MODEL.METRIC_LOSS_TYPE contains 'triplet'.
      - If LOSS.PHASED.ENABLE=False: require LOSS.TRIPLETX.ENABLE=True and metric type is triplet;
        optionally gate by [0, TRIPLETX_END).
    """
    metric_is_triplet = "triplet" in str(getattr(cfg.MODEL, "METRIC_LOSS_TYPE", "")).lower()
    tx_cfg_enabled = bool(getattr(getattr(cfg.LOSS, "TRIPLETX", object()), "ENABLE", False))
    phased_on = bool(getattr(getattr(cfg.LOSS, "PHASED", object()), "ENABLE", False))

    if phased_on:
        bounds = list(getattr(getattr(cfg.LOSS, "PHASED", object()), "BOUNDARIES", []))
        mseq = getattr(getattr(cfg.LOSS, "PHASED", object()), "METRIC_SEQ", None)
        if isinstance(mseq, (list, tuple)) and len(mseq) > 0:
            idx = _phase_index_local(int(epoch), bounds)
            kind = str(mseq[min(idx, len(mseq) - 1)]).strip().lower()
            return (kind == "tripletx") and tx_cfg_enabled and metric_is_triplet
        return False

    if not (tx_cfg_enabled and metric_is_triplet):
        return False
    try:
        tx_end = int(getattr(getattr(cfg.LOSS, "PHASED", object()), "TRIPLETX_END", 40))
    except Exception:
        tx_end = 40
    return int(epoch) < int(tx_end)


# ---------------------- NEW: SupCon effective weight ------------------------

def _supcon_eff_w(cfg, epoch: int) -> float:
    """
    Effective SupCon weight at this epoch.
    Priority (only if LOSS.SUPCON.ENABLE=True):
      1) PHASED.W_SUP_SPEC (per-phase const/linear, using BOUNDARIES)
      2) SUPCON.DECAY_* (linear/const/exp)
      3) Static SUPCON.W
    If LOSS.SUPCON.ENABLE=False => return 0.0 always.
    """
    sup_enabled = bool(getattr(getattr(cfg.LOSS, "SUPCON", object()), "ENABLE", False))
    if not sup_enabled:
        return 0.0

    phased_on = bool(getattr(getattr(cfg.LOSS, "PHASED", object()), "ENABLE", False))

    if phased_on and (epoch is not None):
        spec = getattr(getattr(cfg.LOSS, "PHASED", object()), "W_SUP_SPEC", None)
        bounds = list(getattr(getattr(cfg.LOSS, "PHASED", object()), "BOUNDARIES", []))
        if isinstance(spec, (list, tuple)) and len(spec) > 0:
            i = _phase_index_local(int(epoch), bounds)
            i = max(0, min(i, len(spec) - 1))
            s = str(spec[i]).strip().lower()
            if s.startswith("const:"):
                try:
                    return max(0.0, float(s.split("const:")[1]))
                except Exception:
                    return 0.0
            if s.startswith("linear:"):
                try:
                    body = s.split("linear:")[1]
                    x, y = body.split("->")
                    start = 0 if i == 0 else int(bounds[i - 1])
                    end = int(bounds[i]) if i < len(bounds) else int(getattr(getattr(cfg.SOLVER, "MAX_EPOCHS", object()), "value", getattr(cfg.SOLVER, "MAX_EPOCHS", 120)))
                    end = max(end, start + 1)
                    t = (int(epoch) - start) / float(end - start)
                    t = min(max(t, 0.0), 1.0)
                    return max(0.0, (1.0 - t) * float(x) + t * float(y))
                except Exception:
                    return 0.0
        # fallback to DECAY_*

    if phased_on and (epoch is not None):
        sup_node = getattr(cfg.LOSS, "SUPCON", object())
        try:
            w0 = float(getattr(sup_node, "W0", getattr(sup_node, "W", 0.0)))
        except Exception:
            w0 = float(getattr(sup_node, "W", 0.0))
        decay_type = str(getattr(sup_node, "DECAY_TYPE", "linear")).lower()
        ds = int(getattr(sup_node, "DECAY_START", 0))
        de = int(getattr(sup_node, "DECAY_END", getattr(cfg.SOLVER, "MAX_EPOCHS", 120)))

        if decay_type == "linear":
            if epoch <= ds:
                return max(0.0, w0)
            if epoch >= de:
                return 0.0
            frac = (epoch - ds) / float(max(de - ds, 1))
            return max(0.0, w0 * (1.0 - frac))
        elif decay_type == "const":
            return max(0.0, w0)
        elif decay_type == "exp":
            import math
            if epoch < ds:
                return max(0.0, w0)
            return max(0.0, w0 * math.exp(-0.05 * (epoch - ds)))

    try:
        return max(0.0, float(getattr(getattr(cfg.LOSS, "SUPCON", object()), "W", 0.0)))
    except Exception:
        return 0.0


# ---------------------- NEW: desired K decision -----------------------------

def _desired_pk_k(epoch: int, cfg) -> int:
    """
    Dynamic P×K (K) selection driven by actual loss activation in the current phase:
      - If DATALOADER.PHASED.ENABLE=False → return static NUM_INSTANCE.
      - Else if TripletX is active in this phase → K_WHEN_TRIPLETX.
      - Else if SupCon is active in this phase (ENABLE=True) and effW>0 → K_WHEN_SUPCON.
      - Else → K_OTHER.
    """
    if not _phased_pk_enabled(cfg):
        return int(getattr(cfg.DATALOADER, "NUM_INSTANCE", 4))

    if _tripletx_stage(epoch, cfg):
        try:
            return int(getattr(cfg.DATALOADER.PHASED, "K_WHEN_TRIPLETX", 8))
        except Exception:
            return 8

    sup_w = _supcon_eff_w(cfg, int(epoch))
    if sup_w > 0.0:
        try:
            return int(getattr(cfg.DATALOADER.PHASED, "K_WHEN_SUPCON", 8))
        except Exception:
            return 8

    return _phased_pk_other(cfg)


# ---------------------- Brightness gating flag ------------------------------

def _brightness_enabled(cfg) -> bool:
    """Return True if brightness-aware training is enabled by config."""
    try:
        return bool(getattr(getattr(cfg.LOSS, "BRIGHTNESS", object()), "ENABLE", False))
    except Exception:
        return False


# =============================== Trainer ====================================

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
    start_epoch: int = 1,

    # Backward-compat flags (SupCon handled inside make_loss now)
    use_supcon: bool = False,
    supcon_criterion=None,
    supcon_weight: float = 0.2,
):
    """
    Main training loop.

    Notes:
    - SupCon weighting/schedule happens INSIDE make_loss via (epoch) and cfg.
    - PK-sampler K switching:
        * If DATALOADER.PHASED.ENABLE=True and SAMPLER!='random':
            - If TripletX phase is actually active -> K_WHEN_TRIPLETX
            - Else if SupCon effW>0 (and ENABLE=True) -> K_WHEN_SUPCON
            - Else -> K_OTHER
    """
    import sys
    train_logger = logging.getLogger("transreid.train")
    train_logger.propagate = False
    train_logger.setLevel(logging.INFO)
    for h in list(train_logger.handlers):
        train_logger.removeHandler(h)
    _stream = logging.StreamHandler(stream=sys.stdout)
    _formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _stream.setFormatter(_formatter)
    train_logger.addHandler(_stream)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    logger = train_logger

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    epochs_this_run = cfg.SOLVER.MAX_EPOCHS

    device_str = _pick_device_str(cfg)
    device = torch.device(device_str)
    use_cuda = (device_str == "cuda")

    logger.info(f"start training (device={device_str})")
    if use_supcon:
        logger.info(f"[SupCon] training flag=True (note: weighted inside make_loss)")

    model.to(device)

    if use_cuda and torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    current_K = int(getattr(cfg.DATALOADER, "NUM_INSTANCE", 4))

    for epoch in range(start_epoch, start_epoch + epochs_this_run):

        # ----- Dynamic P×K K switching at epoch boundaries -----
        if not _sampler_is_random(cfg):
            desired_K = _desired_pk_k(epoch, cfg)

            # Optional debug print of K decision factors
            if _phased_pk_enabled(cfg):
                _sup_w_dbg = _supcon_eff_w(cfg, int(epoch))
                logger.info(
                    f"[PK-sampler] epoch={epoch} | TX_phase={_tripletx_stage(epoch,cfg)} "
                    f"| SupConW={_sup_w_dbg:.3f} | K(target)={desired_K}"
                )

            if desired_K != current_K:
                logger.info(f"[PK-sampler] Rebuilding train_loader: NUM_INSTANCE {current_K} → {desired_K} (epoch {epoch})")
                cfg.defrost()
                cfg.DATALOADER.NUM_INSTANCE = desired_K
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

        # Step scheduler
        try:
            scheduler.step(epoch)
        except TypeError:
            try:
                scheduler.step()
            except Exception:
                pass

        model.train()

        for n_iter, batch in enumerate(train_loader):
            # Support tuples from dataset:
            # (img, vid, target_cam, camids, target_view, _) or (img, vid, target_cam, target_view)
            if len(batch) == 6:
                img, vid, target_cam, camids, target_view, _ = batch
            elif len(batch) == 5:
                img, vid, target_cam, target_view, _ = batch
                camids = target_cam
            else:
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

            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = model(img, target, cam_label=target_cam, view_label=target_view)

                score = None
                feat = None
                feat_bn = None
                z_supcon = None

                if isinstance(out, (tuple, list)):
                    if len(out) >= 4:
                        score, maybe_feats, feat_bn, z_supcon = out[0], out[1], out[2], out[3]
                        feat = maybe_feats
                    elif len(out) == 3:
                        score, maybe_feats, z_supcon = out
                        feat = maybe_feats
                    elif len(out) == 2:
                        score, feat = out
                    else:
                        score = out[0]
                        feat = out[1] if len(out) > 1 else out[0]
                        z_supcon = out[2] if len(out) > 2 else None
                else:
                    score = out
                    feat = out

                # ----- Brightness hints (optional) -----
                dark_mask = None
                dark_weight = None
                if _brightness_enabled(cfg):
                    with torch.no_grad():
                        mean = torch.as_tensor(cfg.INPUT.PIXEL_MEAN, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
                        std  = torch.as_tensor(cfg.INPUT.PIXEL_STD,  dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
                        img01 = (img * std + mean).clamp(0, 1)
                        r, g, b = img01[:, 0], img01[:, 1], img01[:, 2]
                        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
                        _brightness = y.mean(dim=(1, 2))

                    _THRESH = float(getattr(getattr(cfg.LOSS, "BRIGHTNESS", object()), "THRESH", 0.35))
                    _K = float(getattr(getattr(cfg.LOSS, "BRIGHTNESS", object()), "K", 0.08))
                    dark_mask = (_brightness < _THRESH)
                    dark_weight = torch.sigmoid((_THRESH - _brightness) / max(_K, 1e-6)).detach()

                loss = loss_func(
                    score,
                    feat,
                    target,
                    camids=camids,
                    z_supcon=z_supcon,
                    epoch=epoch,
                    feat_bn=feat_bn,
                    dark_mask=dark_mask,
                    dark_weight=dark_weight,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ("center" in cfg.MODEL.METRIC_LOSS_TYPE) and (center_criterion is not None) and (optimizer_center is not None):
                for p in center_criterion.parameters():
                    p.grad.data *= (1.0 / max(cfg.SOLVER.CENTER_LOSS_WEIGHT, 1e-12))
                scaler.step(optimizer_center)
                scaler.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            elif score is not None and hasattr(score, "max"):
                acc = (score.max(1)[1] == target).float().mean()
            else:
                acc = torch.tensor(0.0, device=device)

            loss_meter.update(float(loss.item()), img.shape[0])
            acc_meter.update(float(acc), 1)

            if use_cuda:
                torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0:
                try:
                    lr_val = optimizer.param_groups[0]["lr"]
                except Exception:
                    lr_val = 0.0

                logger.info(
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

        if (epoch % checkpoint_period == 0) or (epoch == start_epoch + epochs_this_run - 1):
            weight_path = os.path.join(cfg.OUTPUT_DIR, f"transformer_{epoch}.pth")
            state_path = os.path.join(cfg.OUTPUT_DIR, f"state_{epoch}.pth")

            def _save_all():
                if isinstance(model, nn.parallel.DistributedDataParallel):
                    torch.save(model.module.state_dict(), weight_path)
                else:
                    torch.save(model.state_dict(), weight_path)

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

        if (epoch % eval_period == 0) or (epoch == start_epoch + epochs_this_run - 1):
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    _validate_once(cfg, model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model,
                                   val_loader, evaluator, device, use_cuda, epoch)
            else:
                _validate_once(cfg, model, val_loader, evaluator, device, use_cuda, epoch)


# =============================== Inference ==================================

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
    dev_str = _pick_device_str(cfg)
    device = torch.device(dev_str)
    use_cuda = (dev_str == "cuda")

    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

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

    return float(cmc[0]), float(cmc[4])
