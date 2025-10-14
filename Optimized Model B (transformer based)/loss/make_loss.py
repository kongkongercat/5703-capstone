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
def _supcon_weight_by_epoch(epoch: int) -> float:        # [NEW] phased SupCon weight schedule
    """A: 0–30 -> 0.30;  B: 30–60 -> 0.30→0.15 linear;  C: 60–120 -> 0.0"""
    if epoch <= 30:
        return 0.30
    elif epoch <= 60:
        # linear decay 0.30 -> 0.15 over 30 epochs
        return 0.30 - 0.15 * ((epoch - 30) / 30.0)
    else:
        return 0.0


def _phased_selector(epoch: int):                         # [NEW] decide TripletX/Triplet + metric/SupCon weights by epoch
    """
    Returns:
        use_tripletx (bool), w_metric (float), w_sup (float)
    A(0–30): TripletX, w_metric=1.2, w_sup=0.30
    B(30–60): Triplet,  w_metric=1.0, w_sup=0.30→0.15
    C(60–120): Triplet,  w_metric=1.0, w_sup=0.0
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
    if hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "SUPCON") and cfg.LOSS.SUPCON.ENABLE:
        supcon_criterion = SupConLoss(temperature=cfg.LOSS.SUPCON.T)
        print(f"[make_loss] SupCon enabled: W={cfg.LOSS.SUPCON.W}, T={cfg.LOSS.SUPCON.T}")

    # --------------------------------
    # Metric loss: Triplet or TripletX
    # --------------------------------
    triplet = None
    USE_TRIPLETX_AVAILABLE = False  # availability flag
    if 'triplet' in str(cfg.MODEL.METRIC_LOSS_TYPE):
        USE_TRIPLETX_AVAILABLE = (
            TripletXLoss is not None
            and hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "TRIPLETX")
            and getattr(cfg.LOSS.TRIPLETX, "ENABLE", False)
        )
        if USE_TRIPLETX_AVAILABLE:
            triplet = TripletXLoss(
                margin=cfg.LOSS.TRIPLETX.MARGIN,
                use_soft_warmup=cfg.LOSS.TRIPLETX.SOFT_WARMUP,
                warmup_epochs=cfg.LOSS.TRIPLETX.WARMUP_EPOCHS,
                k=cfg.LOSS.TRIPLETX.K,
                normalize_feature=cfg.LOSS.TRIPLETX.NORM_FEAT,
                cross_cam_pos=cfg.LOSS.TRIPLETX.CROSS_CAM_POS,
                same_cam_neg_boost=cfg.LOSS.TRIPLETX.SAME_CAM_NEG_BOOST,
                alpha=cfg.LOSS.TRIPLETX.ALPHA,
            )
            print("[make_loss] Using TripletX loss "
                  f"(K={cfg.LOSS.TRIPLETX.K}, margin={cfg.LOSS.TRIPLETX.MARGIN}, "
                  f"warmup={cfg.LOSS.TRIPLETX.SOFT_WARMUP}/{cfg.LOSS.TRIPLETX.WARMUP_EPOCHS}, "
                  f"cross_cam_pos={cfg.LOSS.TRIPLETX.CROSS_CAM_POS})")
        else:
            if getattr(cfg.MODEL, "NO_MARGIN", False):
                triplet = TripletLoss()
                print("[make_loss] Using soft (no-margin) Triplet loss")
            else:
                triplet = TripletLoss(cfg.SOLVER.MARGIN)
                print(f"[make_loss] Using standard Triplet loss (margin={cfg.SOLVER.MARGIN})")
    else:
        print(f"[make_loss][warn] METRIC_LOSS_TYPE='{cfg.MODEL.METRIC_LOSS_TYPE}' "
              "does not include 'triplet'; TRI_LOSS will be skipped.")

    # -----------------------------
    # Define loss function per sampler type
    # -----------------------------
    if sampler == 'softmax':
        # Used in B0/B1/B4 (ID + optional SupCon)
        def loss_func(score, feat, target, camids=None, z_supcon=None, epoch=None):
            ce = xent(score, target) if (xent is not None) else F.cross_entropy(score, target)
            total = ce
            if supcon_criterion is not None:
                z = _pick_z(z_supcon, feat)
                total += cfg.LOSS.SUPCON.W * supcon_criterion(z, target, camids)
            return total

    elif sampler == 'softmax_triplet':
        # Used in B0 (CE+Triplet), B1 (CE+Triplet+SupCon),
        # B2/B4 (CE+TripletX+SupCon); supports phased scheduling by switch.
        def loss_func(score, feat, target, camids=None, z_supcon=None, epoch=None):
            # ---- ID Loss ----
            if xent is not None and getattr(cfg.MODEL, "IF_LABELSMOOTH", "off") == 'on':
                if isinstance(score, list):
                    id_list = [xent(s, target) for s in score[1:]]
                    ID_LOSS = (sum(id_list) / len(id_list) + xent(score[0], target)) * 0.5
                else:
                    ID_LOSS = xent(score, target)
            else:
                if isinstance(score, list):
                    id_list = [F.cross_entropy(s, target) for s in score[1:]]
                    ID_LOSS = (sum(id_list) / len(id_list) + F.cross_entropy(score[0], target)) * 0.5
                else:
                    ID_LOSS = F.cross_entropy(score, target)

            # ---- Determine weights: phased vs static (guarded by cfg.LOSS.PHASED.ENABLE) ----  # [NEW]
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
                w_metric = cfg.MODEL.TRIPLET_LOSS_WEIGHT
                w_sup = getattr(cfg.LOSS.SUPCON, "W", 0.0) if supcon_criterion is not None else 0.0

            # ---- Triplet / TripletX ----
            TRI_LOSS = 0.0
            if triplet is not None:
                use_tripletx_now = bool(use_tx_phase and USE_TRIPLETX_AVAILABLE)
                if isinstance(feat, list):
                    tri_vals = []
                    for f in feat[1:]:
                        if use_tripletx_now:
                            tri_vals.append(_reduce_triplet_output(triplet(f, target, camids=camids, epoch=epoch)))
                        else:
                            tri_vals.append(_reduce_triplet_output(triplet(f, target)))
                    if use_tripletx_now:
                        tri0 = _reduce_triplet_output(triplet(feat[0], target, camids=camids, epoch=epoch))
                    else:
                        tri0 = _reduce_triplet_output(triplet(feat[0], target))
                    TRI_LOSS = (sum(tri_vals) / len(tri_vals) + tri0) * 0.5
                else:
                    if use_tripletx_now:
                        TRI_LOSS = _reduce_triplet_output(triplet(feat, target, camids=camids, epoch=epoch))
                    else:
                        TRI_LOSS = _reduce_triplet_output(triplet(feat, target))

            # ---- SupCon ----
            SUPCON_LOSS = 0.0
            if supcon_criterion is not None and w_sup > 0:
                z = _pick_z(z_supcon, feat)
                SUPCON_LOSS = supcon_criterion(z, target, camids)

            # ---- Total ----
            total = (
                cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS
                + w_metric * TRI_LOSS
                + w_sup * SUPCON_LOSS
            )
            return total

    elif sampler == 'random':
        # Used in B3 (self-supervised pretraining)
        def loss_func(score, feat, target, camids=None, z_supcon=None, epoch=None):
            if supcon_criterion is None:
                raise RuntimeError("sampler='random' requires LOSS.SUPCON.ENABLE=True.")
            z = _pick_z(z_supcon, feat)
            return cfg.LOSS.SUPCON.W * supcon_criterion(z, target, camids)

    else:
        raise ValueError(f"[make_loss] sampler must be one of "
                         f"['softmax', 'softmax_triplet', 'random'], "
                         f"but got '{cfg.DATALOADER.SAMPLER}'")

    return loss_func, center_criterion
