# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# ==========================================
# Change Log
# ==========================================
# [2025-09-11 | Hang Zhang] Added support for Supervised Contrastive Loss (SupConLoss)
#                           and integrated cfg.LOSS.SUPCON.ENABLE / W / T into total loss.
# [2025-09-22 | Zeyu Yang] Added new branch for sampler == "random" (SupCon-only mode).
# [2025-10-06 | Meng Fanyi] Fixed NameError when METRIC_LOSS_TYPE=none (triplet undefined).
#                           Added graceful fallback for SupCon-only (self-supervised) training.
# [2025-10-08 | Hang Zhang] Integrated TripletX path (B2/B4) with full argument binding from
#                           cfg.LOSS.TRIPLETX (BNNeck+L2, top-k mining, margin warmup, 
#                           camera-aware, hardness-aware). 
#                           Unified SupCon usage for B1/B2/B4 with safe z_supcon fallback.
#                           Added sampler='random' branch for B3 (SSL pretraining).
#                           Made triplet optional; added camids/epoch passthrough for TripletX.
# [2025-10-11 | Hang Zhang] Compatibility fix (Plan B):
#                           - Conditionally pass camids/epoch only when TRIPLETX.ENABLE=True.
#                           - Keep plain TripletLoss two-arg call in B0.
#                           - Robust handling for tuple/single return from metric loss.
# [2025-10-14 | Hang Zhang] Phased-loss scheduling (B2_phased_loss):
#                           - A(0–30): TripletX, w_metric=1.2, w_sup=0.30
#                           - B(30–60): Triplet,  w_metric=1.0, w_sup linearly 0.30→0.15
#                           - C(60–120): Triplet, w_metric=1.0, w_sup=0.0
#                           - If epoch is None → fallback to static cfg weights.
# [2025-10-14 | Hang Zhang] **Guard switch to avoid affecting other configs.**   # [NEW]
#                           - Add cfg.LOSS.PHASED.ENABLE (default False).
#                           - Only when PHASED.ENABLE=True AND epoch is not None, 
#                             run phased scheduling; otherwise use static weights.
# [2025-10-14 | Zeyu Yang] Triplet compatibility and gating:
#                          - Compute Triplet loss only if MODEL.TRIPLET_LOSS_WEIGHT > 0
#                          - Pass (camids, epoch) only when using TripletX; omit for plain Triplet
#                          - Fallback support for implementations without epoch argument
#                          - Added weight gating for both ID and SupCon losses as well
# ==========================================

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .supcon_loss import SupConLoss

# Optional: Enhanced TripletX (used in B2/B4)
try:
    from .tripletx_loss import TripletLossX as TripletXLoss
except Exception:
    TripletXLoss = None


def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _pick_z(z_supcon, feat):
    """Select z_supcon if provided; otherwise fall back to backbone features."""
    if z_supcon is not None:
        return z_supcon
    return feat[0] if isinstance(feat, list) else feat


def _reduce_triplet_output(out):
    """Return scalar loss if metric loss returns (loss, aux) or a single value."""
    return out[0] if isinstance(out, (list, tuple)) else out


# ------------------------------
# Phased scheduling helpers
# ------------------------------
def _supcon_weight_by_epoch(epoch: int) -> float:
    """A: 0–30 -> 0.30; B: 30–60 -> 0.30→0.15 linear; C: 60–120 -> 0.0"""
    if epoch <= 30:
        return 0.30
    elif epoch <= 60:
        return 0.30 - 0.15 * ((epoch - 30) / 30.0)
    else:
        return 0.0


def _phased_selector(epoch: int):
    """
    Returns: use_tripletx (bool), w_metric (float), w_sup (float)
    A(0–30): TripletX, w_metric=1.2, w_sup=0.30
    B(30–60): Triplet,  w_metric=1.0, w_sup=0.30→0.15
    C(60–120): Triplet, w_metric=1.0, w_sup=0.0
    """
    if epoch <= 30:
        return True, 1.2, 0.30
    elif epoch <= 60:
        return False, 1.0, _supcon_weight_by_epoch(epoch)
    else:
        return False, 1.0, 0.0


def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    use_gpu = torch.cuda.is_available()
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=use_gpu)

    # ---- weights with robust float casting ----
    id_w = _to_float(getattr(cfg.MODEL, "ID_LOSS_WEIGHT", 1.0), 1.0)
    tri_w = _to_float(getattr(cfg.MODEL, "TRIPLET_LOSS_WEIGHT", 1.0), 1.0)

    # --------------------------------
    # Cross-entropy (ID) loss
    # --------------------------------
    xent = None
    if getattr(cfg.MODEL, "IF_LABELSMOOTH", "off") == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print(f"[make_loss] Label smoothing ON, num_classes={num_classes}")

    # --------------------------------
    # Supervised Contrastive Loss (SupCon)
    # --------------------------------
    supcon_criterion = None
    supcon_enabled = False
    supcon_w = 0.0
    if hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "SUPCON") and bool(getattr(cfg.LOSS.SUPCON, "ENABLE", False)):
        supcon_criterion = SupConLoss(temperature=_to_float(cfg.LOSS.SUPCON.T, 0.07))
        supcon_w = _to_float(getattr(cfg.LOSS.SUPCON, "W", 0.0), 0.0)
        supcon_enabled = True
        print(f"[make_loss] SupCon enabled: W={supcon_w}, T={cfg.LOSS.SUPCON.T}")

    # --------------------------------
    # Metric loss: Triplet or TripletX
    # --------------------------------
    triplet = None
    USE_TRIPLETX_AVAILABLE = False  # availability flag
    metric_type = str(getattr(cfg.MODEL, "METRIC_LOSS_TYPE", "")).lower()

    # 仅当模型声明 triplet 且权重大于 0 时才实例化
    if ('triplet' in metric_type) and (tri_w > 0.0):
        USE_TRIPLETX_AVAILABLE = (
            TripletXLoss is not None
            and hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "TRIPLETX")
            and bool(getattr(cfg.LOSS.TRIPLETX, "ENABLE", False))
        )
        if USE_TRIPLETX_AVAILABLE:
            triplet = TripletXLoss(
                margin=_to_float(cfg.LOSS.TRIPLETX.MARGIN, 0.3),
                use_soft_warmup=bool(getattr(cfg.LOSS.TRIPLETX, "SOFT_WARMUP", False)),
                warmup_epochs=int(getattr(cfg.LOSS.TRIPLETX, "WARMUP_EPOCHS", 0)),
                k=int(getattr(cfg.LOSS.TRIPLETX, "K", 4)),
                normalize_feature=bool(getattr(cfg.LOSS.TRIPLETX, "NORM_FEAT", False)),
                cross_cam_pos=bool(getattr(cfg.LOSS.TRIPLETX, "CROSS_CAM_POS", False)),
                same_cam_neg_boost=_to_float(getattr(cfg.LOSS.TRIPLETX, "SAME_CAM_NEG_BOOST", 0.0), 0.0),
                alpha=_to_float(getattr(cfg.LOSS.TRIPLETX, "ALPHA", 1.0), 1.0),
            )
            print("[make_loss] Using TripletX loss "
                  f"(K={getattr(cfg.LOSS.TRIPLETX, 'K', 4)}, margin={getattr(cfg.LOSS.TRIPLETX, 'MARGIN', 0.3)}, "
                  f"warmup={getattr(cfg.LOSS.TRIPLETX, 'SOFT_WARMUP', False)}/{getattr(cfg.LOSS.TRIPLETX, 'WARMUP_EPOCHS', 0)}, "
                  f"cross_cam_pos={getattr(cfg.LOSS.TRIPLETX, 'CROSS_CAM_POS', False)})")
        else:
            if bool(getattr(cfg.MODEL, "NO_MARGIN", False)):
                triplet = TripletLoss()
                print("[make_loss] Using soft (no-margin) Triplet loss")
            else:
                triplet = TripletLoss(_to_float(getattr(cfg.SOLVER, "MARGIN", 0.3), 0.3))
                print(f"[make_loss] Using standard Triplet loss (margin={getattr(cfg.SOLVER, 'MARGIN', 0.3)})")
    else:
        if 'triplet' in metric_type:
            print(f"[make_loss] Triplet declared but weight={tri_w}; skip metric loss.")
        else:
            print(f"[make_loss] METRIC_LOSS_TYPE='{getattr(cfg.MODEL, 'METRIC_LOSS_TYPE', '')}' "
                  "does not include 'triplet'; TRI_LOSS will be skipped.")

    # -----------------------------
    # Define loss function per sampler type
    # -----------------------------
    if sampler == 'softmax':
        # B0/B1/B4：ID (+ optional SupCon)
        def loss_func(score, feat, target, camids=None, z_supcon=None, epoch=None):
            total = 0.0

            # ---- ID loss ----
            if id_w > 0:
                if xent is not None and getattr(cfg.MODEL, "IF_LABELSMOOTH", "off") == 'on':
                    ce = xent(score, target)
                else:
                    ce = F.cross_entropy(score, target)
                total += id_w * ce

            # ---- SupCon ----
            if supcon_enabled and supcon_w > 0:
                z = _pick_z(z_supcon, feat)
                total += supcon_w * supcon_criterion(z, target, camids)

            return total

    elif sampler == 'softmax_triplet':
        # B0（CE+Triplet）、B1（CE+Triplet+SupCon）、B2/B4（CE+TripletX+SupCon）
        def loss_func(score, feat, target, camids=None, z_supcon=None, epoch=None):
            total = 0.0

            # ---- ID loss ----
            if id_w > 0:
                if isinstance(score, list):
                    if xent is not None and getattr(cfg.MODEL, "IF_LABELSMOOTH", "off") == 'on':
                        id_list = [xent(s, target) for s in score[1:]]
                        ID_LOSS = (sum(id_list) / len(id_list) + xent(score[0], target)) * 0.5
                    else:
                        id_list = [F.cross_entropy(s, target) for s in score[1:]]
                        ID_LOSS = (sum(id_list) / len(id_list) + F.cross_entropy(score[0], target)) * 0.5
                else:
                    if xent is not None and getattr(cfg.MODEL, "IF_LABELSMOOTH", "off") == 'on':
                        ID_LOSS = xent(score, target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)
                total += id_w * ID_LOSS

            # ---- weights: phased vs static ----
            phased_on = (
                (epoch is not None)
                and hasattr(cfg, "LOSS")
                and hasattr(cfg.LOSS, "PHASED")
                and bool(getattr(cfg.LOSS.PHASED, "ENABLE", False))
            )
            if phased_on:
                use_tx_phase, w_metric, w_sup = _phased_selector(int(epoch))
            else:
                use_tx_phase = USE_TRIPLETX_AVAILABLE
                w_metric = tri_w
                w_sup = supcon_w if supcon_enabled else 0.0

            # ---- Triplet / TripletX ----
            if triplet is not None and w_metric > 0:
                def _call_triplet(f):
                    # TripletX：传 camids/epoch；普通 Triplet：不传
                    if use_tx_phase and USE_TRIPLETX_AVAILABLE:
                        try:
                            return _reduce_triplet_output(triplet(f, target, camids=camids, epoch=epoch))
                        except TypeError:
                            # 兼容旧签名（无 epoch）
                            return _reduce_triplet_output(triplet(f, target, camids=camids))
                    else:
                        try:
                            return _reduce_triplet_output(triplet(f, target))
                        except TypeError:
                            # 极端老版本只接受 (feat, target) 也能跑
                            return _reduce_triplet_output(triplet(f, target))

                if isinstance(feat, list):
                    tri_vals = [_call_triplet(f) for f in feat[1:]]
                    tri0 = _call_triplet(feat[0])
                    TRI_LOSS = (sum(tri_vals) / len(tri_vals) + tri0) * 0.5
                else:
                    TRI_LOSS = _call_triplet(feat)

                total += w_metric * TRI_LOSS

            # ---- SupCon ----
            if supcon_enabled and w_sup > 0:
                z = _pick_z(z_supcon, feat)
                total += w_sup * supcon_criterion(z, target, camids)

            return total

    elif sampler == 'random':
        # B3：自监督预训练（只用 SupCon）
        def loss_func(score, feat, target, camids=None, z_supcon=None, epoch=None):
            if not (supcon_enabled and supcon_w > 0):
                raise RuntimeError("sampler='random' requires LOSS.SUPCON.ENABLE=True and W>0.")
            z = _pick_z(z_supcon, feat)
            return supcon_w * supcon_criterion(z, target, camids)

    else:
        raise ValueError(f"[make_loss] sampler must be one of "
                         f"['softmax', 'softmax_triplet', 'random'], "
                         f"but got '{cfg.DATALOADER.SAMPLER}'")

    return loss_func, center_criterion
