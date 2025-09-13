# -*- coding: utf-8 -*-
"""
@file: tripletx_loss.py
@description: Enhanced Triplet Loss (TripletLossX) with 5 improvements 
              (BNNeck+L2, top-k hard, margin schedule, camera-aware, hardness-aware)
              for vehicle Re-ID on VeRi-776.

@based_on: triplet_loss.py (original baseline TripletLoss implementation)

@created_by: Zhang Hang (hzha0521)
@created_date: 2025-09-14

@modifications:
    - Added BNNeck+L2 normalization before distance calculation
    - Replaced hardest-1 mining with top-k hard mining (stable average)
    - Added margin schedule (soft-margin warmup -> fixed margin)
    - Added camera-aware mining (cross-camera positive, boosted same-camera negative)
    - Added hardness-aware per-anchor weighting
"""

import torch
import torch.nn as nn


def _l2_normalize(x, dim=-1, eps=1e-12):
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


def _euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = (x**2).sum(1, keepdim=True).expand(m, n)
    yy = (y**2).sum(1, keepdim=True).expand(n, m).t()
    dist = (xx + yy - 2.0 * x @ y.t()).clamp_min(1e-12).sqrt()
    return dist


def _topk_mean(dist_mat, mask, k=5, largest=False):
    """
    dist_mat: [N,N], mask: [N,N] (True where valid)
    largest=False -> pick k smallest; largest=True -> pick k largest
    return: [N] per-row mean over available elems (>=1); fallback to min/max if row empty
    """
    N = dist_mat.size(0)
    fill_val = float('-inf') if largest else float('inf')
    work = dist_mat.masked_fill(~mask, fill_val)
    vals, _ = torch.topk(work, k=min(k, max(1, mask.sum(1).max().item())), dim=1, largest=largest)
    is_valid = torch.isfinite(vals)
    if largest:
        fallback = dist_mat.masked_fill(~mask, float('-inf')).amax(dim=1)
    else:
        fallback = dist_mat.masked_fill(~mask, float('inf')).amin(dim=1)
    safe_sum = (vals * is_valid).sum(1)
    counts = is_valid.sum(1).clamp_min(1)
    mean = safe_sum / counts
    mean = torch.where(is_valid.any(1), mean, fallback)
    return mean


class TripletLossX(nn.Module):
    """
    Five-point enhanced Triplet:
      (1) BNNeck + L2
      (2) top-k hard mining
      (3) margin schedule (soft-margin warmup -> fixed margin)
      (4) camera-aware mining (cross-camera positive, boosted same-camera negative)
      (5) hardness-aware weighting
    """
    def __init__(self,
                 margin=0.3,
                 use_soft_warmup=True,
                 warmup_epochs=10,
                 k=5,
                 normalize_feature=True,
                 cross_cam_pos=True,
                 same_cam_neg_boost=1.2,
                 alpha=2.0):
        super().__init__()
        self.margin = margin
        self.use_soft_warmup = use_soft_warmup
        self.warmup_epochs = warmup_epochs
        self.k = k
        self.normalize_feature = normalize_feature
        self.cross_cam_pos = cross_cam_pos
        self.same_cam_neg_boost = same_cam_neg_boost
        self.alpha = alpha
        self.rank_hinge = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.softplus = nn.Softplus(beta=1.0)

    @torch.no_grad()
    def _make_masks(self, labels, camids=None):
        N = labels.size(0)
        labels = labels.view(-1, 1)
        is_pos = labels.eq(labels.t())
        is_neg = ~is_pos
        eye = torch.eye(N, dtype=torch.bool, device=labels.device)
        is_pos = is_pos & (~eye)
        is_neg = is_neg & (~eye)
        same_cam = None
        if camids is not None:
            camids = camids.view(-1, 1)
            same_cam = camids.eq(camids.t())
            if self.cross_cam_pos:
                cross_cam = ~same_cam
                is_pos = is_pos & cross_cam
        return is_pos, is_neg, same_cam

    def forward(self, feats, labels, camids=None, epoch: int = None):
        # (1) L2 normalize
        x = _l2_normalize(feats) if self.normalize_feature else feats
        dist = _euclidean_dist(x, x)

        # masks
        is_pos, is_neg, same_cam = self._make_masks(labels, camids)

        # (4) camera-aware: boost same-cam negatives
        if same_cam is not None and self.same_cam_neg_boost > 1.0:
            neg_bias = torch.ones_like(dist)
            bias_mask = is_neg & same_cam
            neg_bias = neg_bias.masked_fill(bias_mask, self.same_cam_neg_boost)
            dist_neg_sel = dist / neg_bias.clamp_min(1e-6)
        else:
            dist_neg_sel = dist

        # (2) top-k hard
        d_ap = _topk_mean(dist, is_pos, k=self.k, largest=True)
        d_an = _topk_mean(dist_neg_sel, is_neg, k=self.k, largest=False)

        # (5) hardness-aware
        with torch.no_grad():
            w = torch.sigmoid(self.alpha * (d_ap - d_an))
            w = w / (w.mean().clamp_min(1e-12))

        # (3) margin schedule
        if self.use_soft_warmup and (epoch is not None) and (epoch < self.warmup_epochs):
            per_sample = self.softplus(-(d_an - d_ap))
        else:
            y = torch.ones_like(d_an)
            per_sample = self.rank_hinge(d_an, d_ap, y)

        loss = (w * per_sample).mean()
        return loss, d_ap.detach(), d_an.detach()
