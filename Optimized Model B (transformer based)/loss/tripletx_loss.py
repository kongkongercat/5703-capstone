# -*- coding: utf-8 -*-
# =============================================================================
# File: tripletx_loss.py
# Description: Enhanced Triplet Loss (TripletLossX) with 5 improvements
#              (L2, top-k hard, margin schedule, camera-aware, hardness-aware)
#              for vehicle Re-ID on VeRi-776.
#
# Based on: triplet_loss.py (original baseline TripletLoss implementation)
#
# Change Log
# [2025-09-14 | Hang Zhang] Added BNNeck+L2 normalization before distance calculation
# [2025-09-14 | Hang Zhang] Replaced hardest-1 mining with top-k hard mining (stable average)
# [2025-09-14 | Hang Zhang] Added margin schedule (soft-margin warmup -> fixed margin)
# [2025-09-14 | Hang Zhang] Added camera-aware mining (cross-camera positive, boosted same-camera negative)
# [2025-09-14 | Hang Zhang] Added hardness-aware per-anchor weighting
# [2025-10-09 | Hang Zhang] FIX: Added cross-camera positive fallback to avoid -inf/+inf
# [2025-10-09 | Hang Zhang] FIX: Ensured finite top-k selection via row-wise fallback and cleanup
# [2025-10-09 | Hang Zhang] FIX: Safe reduction using nan_to_num() and weight normalization
# [2025-10-18 | Hang Zhang] CHANGE: Remove all "row-wise fallback" and "pos_all fallback";
#                           skip anchors with no positive/negative via valid_mask during reduction.
# [2025-10-18 | Hang Zhang] CHANGE: _topk_mean() adds `fallback_when_empty=False` (default);
#                           invalid rows remain invalid (masked out later).
# [2025-10-21 | Hang Zhang] REVERT: Restore row-wise fallback & pos-all fallback removed on 2025-10-18;
#                           - _topk_mean(): default fallback_when_empty=True (legacy)
#                           - _make_masks(): if cross-cam positives empty, fallback to all positives
#                           - Keep valid_mask & nan_to_num() safety reduction
# =============================================================================

import torch
import torch.nn as nn
from typing import Tuple, Optional


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


def _euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, n = x.size(0), y.size(0)
    xx = (x ** 2).sum(1, keepdim=True).expand(m, n)
    yy = (y ** 2).sum(1, keepdim=True).expand(n, m).t()
    dist = (xx + yy - 2.0 * x @ y.t()).clamp_min(1e-12).sqrt()
    return dist


def _topk_mean(
    dist_mat: torch.Tensor,
    mask: torch.Tensor,
    k: int = 5,
    largest: bool = False,
    fallback_when_empty: bool = True,  # REVERT: default True (legacy row-wise fallback)
) -> torch.Tensor:
    """
    Stable top-k mean under masking.

    If a row has no valid entries (mask=False for all columns):
    - When `fallback_when_empty` is True (default), the row falls back to "whole row minus diagonal".
    - When False, the row remains invalid (all fill_val) and should be skipped by a later valid_mask.
    """
    N = dist_mat.size(0)
    device = dist_mat.device
    eye = torch.eye(N, dtype=torch.bool, device=device)

    fill_val = float("-inf") if largest else float("inf")
    work = dist_mat.masked_fill(~mask, fill_val)

    row_has_any = mask.any(dim=1)
    rows_empty = ~row_has_any
    if rows_empty.any():
        if fallback_when_empty:
            # Legacy fallback: use the whole row except diagonal
            fallback_mask = ~eye
            work[rows_empty] = dist_mat[rows_empty].masked_fill(~fallback_mask[rows_empty], fill_val)
        else:
            # Keep invalid
            work[rows_empty] = torch.full_like(work[rows_empty], fill_val)

    k_eff = max(1, min(k, N - 1))
    vals, _ = torch.topk(work, k=k_eff, dim=1, largest=largest)
    is_finite = torch.isfinite(vals)

    safe_sum = (vals * is_finite).sum(dim=1)
    counts = is_finite.sum(dim=1).clamp_min(1)
    mean = safe_sum / counts
    mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
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

    def __init__(
        self,
        margin: float = 0.3,
        use_soft_warmup: bool = True,
        warmup_epochs: int = 10,
        k: int = 5,
        normalize_feature: bool = True,
        cross_cam_pos: bool = True,
        same_cam_neg_boost: float = 1.2,
        alpha: float = 2.0,
        debug_checks: bool = False,
    ):
        super().__init__()
        self.margin = float(margin)
        self.use_soft_warmup = bool(use_soft_warmup)
        self.warmup_epochs = int(warmup_epochs)
        self.k = int(k)
        self.normalize_feature = bool(normalize_feature)
        self.cross_cam_pos = bool(cross_cam_pos)
        self.same_cam_neg_boost = float(same_cam_neg_boost)
        self.alpha = float(alpha)
        self.debug_checks = bool(debug_checks)

        self.rank_hinge = nn.MarginRankingLoss(margin=self.margin, reduction="none")
        self.softplus = nn.Softplus(beta=1.0)

    @torch.no_grad()
    def _make_masks(
        self,
        labels: torch.Tensor,
        camids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Build positive/negative masks with camera-aware option.
        REVERT (2025-10-21): If cross-camera positives are empty for an anchor, fall back to "all positives".
        """
        N = labels.size(0)
        labels = labels.view(-1, 1)
        eye = torch.eye(N, dtype=torch.bool, device=labels.device)

        # Base positives/negatives (ignore diagonal)
        is_pos_all = labels.eq(labels.t())
        is_pos_all = is_pos_all & (~eye)
        is_neg = ~is_pos_all & (~eye)

        same_cam = None
        if camids is not None:
            camids = camids.view(-1, 1)
            same_cam = camids.eq(camids.t())
            if self.cross_cam_pos:
                cross_cam = ~same_cam
                is_pos = is_pos_all & cross_cam  # cross-camera positives only
                # REVERT: pos-all fallback when cross-camera pos is empty for some anchors
                has_pos = is_pos.any(dim=1)
                if (~has_pos).any():
                    rows = (~has_pos).nonzero(as_tuple=False).view(-1)
                    is_pos[rows] = is_pos_all[rows]
            else:
                is_pos = is_pos_all
        else:
            is_pos = is_pos_all

        return is_pos, is_neg, same_cam

    def forward(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        camids: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None
    ):
        # (1) L2 normalization
        x = _l2_normalize(feats) if self.normalize_feature else feats

        if self.debug_checks and (not torch.isfinite(x).all()):
            print("[TripletLossX] Non-finite features detected before distance computation.")

        dist = _euclidean_dist(x, x)
        if self.debug_checks and (not torch.isfinite(dist).all()):
            print("[TripletLossX] Non-finite pairwise distances detected.")

        # Build masks (with pos-all fallback when needed)
        is_pos, is_neg, same_cam = self._make_masks(labels, camids)

        # (4) Camera-aware: boost same-camera negatives (distance shrink for same-cam negatives)
        if same_cam is not None and self.same_cam_neg_boost > 1.0:
            bias_mask = is_neg & same_cam
            neg_bias = torch.ones_like(dist).masked_fill(bias_mask, self.same_cam_neg_boost)
            dist_neg_sel = dist / neg_bias.clamp_min(1e-6)
        else:
            dist_neg_sel = dist

        # (2) Top-k hard mining WITH row-wise fallback enabled (legacy default)
        d_ap = _topk_mean(dist,        is_pos, k=self.k, largest=True,  fallback_when_empty=True)
        d_an = _topk_mean(dist_neg_sel, is_neg, k=self.k, largest=False, fallback_when_empty=True)

        if self.debug_checks:
            if (not torch.isfinite(d_ap).all()) or (not torch.isfinite(d_an).all()):
                print("[TripletLossX] Non-finite d_ap/d_an detected.")

        # Valid anchors (still keep safety mask; with fallbacks most anchors will be valid)
        with torch.no_grad():
            has_pos = is_pos.any(dim=1)
            has_neg = is_neg.any(dim=1)
            valid_mask = (has_pos & has_neg).float()

        # (5) Hardness-aware weighting (masked by valid anchors)
        with torch.no_grad():
            w = torch.sigmoid(self.alpha * (d_ap - d_an))
            w = w / (w.mean().clamp_min(1e-12))
            w = w * valid_mask  # skip invalid anchors if any

        # (3) Margin schedule (warmup)
        if self.use_soft_warmup and (epoch is not None) and (epoch < self.warmup_epochs):
            per_sample = self.softplus(d_ap - d_an)  # softplus(-(d_an - d_ap))
        else:
            y = torch.ones_like(d_an)
            per_sample = self.rank_hinge(d_an, d_ap, y)

        # Safe reduction
        per_sample = torch.nan_to_num(per_sample, nan=0.0, posinf=0.0, neginf=0.0)
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        loss = (w * per_sample).sum() / w.sum().clamp_min(1e-12)

        return loss, d_ap.detach(), d_an.detach()