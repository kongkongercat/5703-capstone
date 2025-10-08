# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

# ==========================================
# Change Log
# ==========================================
# [2025-09-11 | Zhang Hang] Added support for Supervised Contrastive Loss (SupConLoss)
#                           and integrated cfg.LOSS.SUPCON.ENABLE / W / T into total loss.
# [2025-09-22 | Zeyu Yang] Added new branch for sampler == "random" (SupCon-only mode).
# [2025-10-06 | Meng Fanyi] Fixed NameError when METRIC_LOSS_TYPE=none (triplet undefined).
#                           Added graceful fallback for SupCon-only (self-supervised) training.
# [2025-10-08 | Zhang Hang] Integrated TripletX path (B2/B4) with full argument binding from
#                           cfg.LOSS.TRIPLETX (BNNeck+L2, top-k mining, margin warmup, 
#                           camera-aware, hardness-aware). 
#                           Unified SupCon usage for B1/B2/B4 with safe z_supcon fallback.
#                           Added sampler='random' branch for B3 (SSL pretraining).
#                           Made triplet optional; added camids/epoch passthrough for TripletX.
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
    if 'triplet' in str(cfg.MODEL.METRIC_LOSS_TYPE):
        use_tripletx = (
            TripletXLoss is not None
            and hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "TRIPLETX")
            and getattr(cfg.LOSS.TRIPLETX, "ENABLE", False)
        )
        if use_tripletx:
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
        # B2/B4 (CE+TripletX+SupCon)
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

            # ---- Triplet / TripletX ----
            if triplet is not None:
                if isinstance(feat, list):
                    tri_vals = []
                    for f in feat[1:]:
                        tri_vals.append(triplet(f, target, camids=camids, epoch=epoch)[0])
                    tri0 = triplet(feat[0], target, camids=camids, epoch=epoch)[0]
                    TRI_LOSS = (sum(tri_vals) / len(tri_vals) + tri0) * 0.5
                else:
                    TRI_LOSS = triplet(feat, target, camids=camids, epoch=epoch)[0]
            else:
                TRI_LOSS = 0.0

            total = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

            # ---- SupCon ----
            if supcon_criterion is not None:
                z = _pick_z(z_supcon, feat)
                total += cfg.LOSS.SUPCON.W * supcon_criterion(z, target, camids)

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
