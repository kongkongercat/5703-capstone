# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

Modified by Zhang Hang on 2025-09-11:
- Added support for Supervised Contrastive Loss (SupConLoss)
- Integrate cfg.LOSS.SUPCON.ENABLE / W / T into total loss calculation
"""

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .supcon_loss import SupConLoss  # <-- Added


def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    use_gpu = torch.cuda.is_available()
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=use_gpu)

    # Triplet loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    # Cross-entropy loss (with optional label smoothing)
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    # SupCon loss (if enabled)
    supcon_criterion = None
    if hasattr(cfg, "LOSS") and "SUPCON" in cfg.LOSS and cfg.LOSS.SUPCON.ENABLE:
        supcon_criterion = SupConLoss(temperature=cfg.LOSS.SUPCON.T)
        print("SupCon enabled: W={}, T={}".format(cfg.LOSS.SUPCON.W, cfg.LOSS.SUPCON.T))

    # -----------------------------
    # Loss function definitions
    # -----------------------------
    if sampler == 'softmax':
        def loss_func(score, feat, target, camids=None, z_supcon=None):
            ce_loss = F.cross_entropy(score, target)
            total_loss = ce_loss
            # Add SupCon if enabled and z_supcon provided
            if supcon_criterion is not None and z_supcon is not None:
                total_loss += cfg.LOSS.SUPCON.W * supcon_criterion(z_supcon, target, camids)
            return total_loss

    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target, camids=None, z_supcon=None):
            # ----- ID Loss -----
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
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
            if isinstance(feat, list):
                tri_losses = [triplet(f, target)[0] for f in feat[1:]]
                TRI_LOSS = (sum(tri_losses) / len(tri_losses) + triplet(feat[0], target)[0]) * 0.5
            else:
                TRI_LOSS = triplet(feat, target)[0]

            total_loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                         cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

            # ----- SupCon Loss -----
            if supcon_criterion is not None and z_supcon is not None:
                total_loss += cfg.LOSS.SUPCON.W * supcon_criterion(z_supcon, target, camids)

            return total_loss

    else:
        print('expected sampler should be softmax or softmax_triplet'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))

    return loss_func, center_criterion
