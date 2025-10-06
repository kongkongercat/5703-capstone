# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

Modified by Zhang Hang on 2025-09-11:
- Added support for Supervised Contrastive Loss (SupConLoss)
- Integrate cfg.LOSS.SUPCON.ENABLE / W / T into total loss calculation
Modified by Zeyu Yang on 2025-09-22:
Added a new branch for sampler == "random": computes only the SupCon loss (no ID / Triplet)
Modified by Meng Fanyi on 2025-10-06:
- Fixed NameError when METRIC_LOSS_TYPE=none (triplet undefined)
- Added graceful fallback for SupCon-only (self-supervised) training
"""
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .supcon_loss import SupConLoss


def _pick_z(z_supcon, feat):
    """优先用 z_supcon；否则回退到主干特征 feat。"""
    if z_supcon is not None:
        return z_supcon
    return feat[0] if isinstance(feat, list) else feat


def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    use_gpu = torch.cuda.is_available()
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=use_gpu)

    # -------------------------------
    # Triplet Loss (may be disabled)
    # -------------------------------
    triplet = None
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            print(f"using triplet loss with margin: {cfg.SOLVER.MARGIN}")
    else:
        print(f"expected METRIC_LOSS_TYPE should be triplet but got {cfg.MODEL.METRIC_LOSS_TYPE}")

    # Cross-entropy loss (with optional label smoothing)
    xent = None
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print(f"label smooth on, numclasses: {num_classes}")

    # SupCon loss
    supcon_criterion = None
    if hasattr(cfg, "LOSS") and "SUPCON" in cfg.LOSS and cfg.LOSS.SUPCON.ENABLE:
        supcon_criterion = SupConLoss(temperature=cfg.LOSS.SUPCON.T)
        print(f"SupCon enabled: W={cfg.LOSS.SUPCON.W}, T={cfg.LOSS.SUPCON.T}")

    # -----------------------------
    # Loss function definitions
    # -----------------------------
    if sampler == 'softmax':
        def loss_func(score, feat, target, camids=None, z_supcon=None):
            ce_loss = F.cross_entropy(score, target)
            total_loss = ce_loss
            if supcon_criterion is not None:
                z = _pick_z(z_supcon, feat)
                total_loss += cfg.LOSS.SUPCON.W * supcon_criterion(z, target, camids)
            return total_loss

    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target, camids=None, z_supcon=None):
            # ----- ID Loss -----
            if cfg.MODEL.IF_LABELSMOOTH == 'on' and xent is not None:
                if isinstance(score, list):
                    id_losses = [xent(s, target) for s in score[1:]]
                    ID_LOSS = (sum(id_losses) / len(id_losses) + xent(score[0], target)) * 0.5
                else:
                    ID_LOSS = xent(score, target)
            else:
                if isinstance(score, list):
                    id_losses = [F.cross_entropy(s, target) for s in score[1:]]
                    ID_LOSS = (sum(id_losses) / len(id_losses) + F.cross_entropy(score[0], target)) * 0.5
                else:
                    ID_LOSS = F.cross_entropy(score, target)

            # ----- Triplet Loss -----
            if triplet is not None:
                if isinstance(feat, list):
                    tri_losses = [triplet(f, target)[0] for f in feat[1:]]
                    TRI_LOSS = (sum(tri_losses) / len(tri_losses) + triplet(feat[0], target)[0]) * 0.5
                else:
                    TRI_LOSS = triplet(feat, target)[0]
            else:
                TRI_LOSS = 0.0

            total_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                         cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

            # ----- SupCon Loss -----
            if supcon_criterion is not None:
                z = _pick_z(z_supcon, feat)
                total_loss += cfg.LOSS.SUPCON.W * supcon_criterion(z, target, camids)

            return total_loss

    elif sampler == 'random':
        # ----- SupCon-only -----
        def loss_func(score, feat, target, camids=None, z_supcon=None):
            if supcon_criterion is None:
                raise RuntimeError("sampler=random 需要启用 SupConLoss，但未在配置中开启。")
            z = _pick_z(z_supcon, feat)
            return cfg.LOSS.SUPCON.W * supcon_criterion(z, target, camids)

    else:
        raise ValueError(f"expected sampler should be softmax/softmax_triplet/random but got {cfg.DATALOADER.SAMPLER}")

    return loss_func, center_criterion
