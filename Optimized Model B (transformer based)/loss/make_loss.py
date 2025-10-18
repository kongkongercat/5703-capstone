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
# [2025-10-14 | Hang Zhang] **Guard switch to avoid affecting other configs.**
#                           - Add cfg.LOSS.PHASED.ENABLE (default False).
#                           - Only when PHASED.ENABLE=True AND epoch is not None, 
#                             run phased scheduling; otherwise use static weights.
# [2025-10-14 | Zeyu Yang] Triplet compatibility and gating:
#                           - Compute Triplet loss only if MODEL.TRIPLET_LOSS_WEIGHT > 0
#                           - Pass (camids, epoch) only when using TripletX; omit for plain Triplet
#                           - Fallback support for implementations without epoch argument
#                           - Added weight gating for both ID and SupCon losses as well
# [2025-10-15 | Hang Zhang] Safety & logging improvements:
#                           - Added _safe_mean() to avoid ZeroDivisionError when using list heads
#                           - Added one-time phase-combo logging (combo + weights) whenever changed
#                           - Removed unused LabelSmoothingCrossEntropy import
# [2025-10-15 | Hang Zhang] Configurable phased schedule via CLI/YAML:
#                           - _phased_selector now reads cfg.LOSS.PHASED:
#                             BOUNDARIES, METRIC_SEQ, W_METRIC_SEQ, W_SUP_SPEC
#                           - Supports 'const:x' and 'linear:x->y' for SupCon weight
#                           - Falls back to original A/B/C if not provided
# [2025-10-15 | Hang Zhang]  **NEW: CLI-switchable feature source (layer) for TripletX & SupCon**
#                           - Read cfg.LOSS.TRIPLETX.FEAT_SRC and cfg.LOSS.SUPCON.FEAT_SRC
#                             ('pre_bn' | 'bnneck') to pick features dynamically.
#                           - Backward compatible: if BNNeck feature is not provided, gracefully
#                             fall back to pre_bn with a one-time warning.
#                           - Extended loss_func signature with optional feat_bn=None to accept
#                             BNNeck feature without breaking old trainers.
# [2025-10-15 | Hang Zhang]  **NEW: Init logging of feature sources**
#                           - Print once at init: the chosen sources for Triplet/TripletX and SupCon.
#                           - If both use the same layer, print a caution about potential gradient interaction.
# [2025-10-17 | Hang Zhang]  **Single-source policy for SupCon (log & behavior)**
#                           - If model returns z_supcon, always use it and IGNORE LOSS.SUPCON.FEAT_SRC.
#                           - Added one-time info log:
#                             "[make_loss][info] [SupCon] using z_supcon from model; LOSS.SUPCON.FEAT_SRC ignored."
#                           - Applied to 'softmax', 'softmax_triplet' and 'random' branches.
# [2025-10-17 | Hang Zhang]  **Stability & clarity**
#                           - _phased_selector now takes (cfg, epoch) explicitly (removed inspect.stack()).
#                           - Added one-time note: "SupConLoss instantiated here (entry should not rebuild)".
#                           - Added one-time note: "CE.FEAT_SRC is removed; CE uses classifier score."
# [2025-10-17 | Hang Zhang]  **Triplet vs TripletX independent FEAT_SRC**
#                           - Read LOSS.TRIPLET.FEAT_SRC and LOSS.TRIPLETX.FEAT_SRC separately.
#                           - Choose FEAT_SRC based on whether TripletX is used in current step.
#                           - TripletX(normalize_feature=False); removed redundant external L2 normalization.
#                           - One-time check log: "[make_loss][check] Metric src=<...>, L2=off".
# [2025-10-18 | Hang Zhang]  **Removed all normalization inside make_loss**
#                           - Deleted F.normalize for Triplet, TripletX, and SupCon (handled internally).
#                           - Ensured no double normalization and unified author name.
# [2025-10-18 | Hang Zhang]  **Baseline-safe ID/Metric list handling & tri_src fix**
#                           - CE: support list outputs with 0.5*(head0 + mean(others)); single Tensor unchanged.
#                           - Triplet: support list features with same 0.5 scheme; single Tensor unchanged.
#                           - Fix: Triplet uses LOSS.TRIPLET.FEAT_SRC; TripletX uses LOSS.TRIPLETX.FEAT_SRC.
# [2025-10-18 | Hang Zhang]  **Compatibility for baseline Triplet**
#                           - Safe fallback for TripletLoss without (camids/epoch) arguments.
# [2025-10-19 | Hang Zhang]  **Remove unused helper & fix CE policy**
#                           - Deleted unused _pick_z() helper.
#                           - Explicit note: CE has no dynamic FEAT_SRC; always uses logits (score).
# ==========================================

import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .supcon_loss import SupConLoss

# Optional: Enhanced TripletX (used in B2/B4)
try:
    from .tripletx_loss import TripletLossX as TripletXLoss
except Exception:
    TripletXLoss = None


# ------------------------------
# Utilities
# ------------------------------
def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_mean(values):
    """Return mean(values) if non-empty else 0.0 (avoid ZeroDivisionError)."""
    return (sum(values) / len(values)) if values else 0.0


def _reduce_triplet_output(out):
    """Return scalar loss if metric loss returns (loss, aux) or a single value."""
    return out[0] if isinstance(out, (list, tuple)) else out


def _unpack_feats_for_sources(feat, feat_bn):
    """Recover (global_pre_bn, bnneck_feat) for flexible feature routing."""
    g, b = None, None
    if isinstance(feat_bn, torch.Tensor):
        b = feat_bn
    if isinstance(feat, (tuple, list)) and len(feat) >= 2 and isinstance(feat[0], torch.Tensor) and isinstance(feat[1], torch.Tensor):
        g, b = feat[0], feat[1]
    elif isinstance(feat, list) and len(feat) >= 1 and isinstance(feat[0], torch.Tensor):
        g = feat[0]
    elif isinstance(feat, torch.Tensor):
        g = feat
    return g, b


def _pick_by_src_with_fallback(src, global_pre_bn, bnneck_feat, once_flag, warn_tag):
    """Return feature by FEAT_SRC with fallback warning."""
    src = (src or "pre_bn").lower()
    if src == "bnneck":
        if isinstance(bnneck_feat, torch.Tensor):
            return bnneck_feat
        if not once_flag.get("warned", False):
            print(f"[make_loss][warn] {warn_tag}: requested 'bnneck' but not provided, fallback to pre_bn.")
            once_flag["warned"] = True
        return global_pre_bn
    return global_pre_bn


# ------------------------------
# Factory
# ------------------------------
def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    use_gpu = torch.cuda.is_available()
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=use_gpu)

    # ---- weights ----
    id_w = _to_float(getattr(cfg.MODEL, "ID_LOSS_WEIGHT", 1.0))
    tri_w = _to_float(getattr(cfg.MODEL, "TRIPLET_LOSS_WEIGHT", 1.0))

    # ---- CE (ID) loss ----
    xent = None
    if getattr(cfg.MODEL, "IF_LABELSMOOTH", "off") == "on":
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print(f"[make_loss] Label smoothing ON, num_classes={num_classes}")
    # CE uses classifier logits (score) directly; no dynamic FEAT_SRC.

    # ---- SupCon (optional; may be disabled in baseline) ----
    supcon_criterion = None
    supcon_enabled = False
    supcon_w = 0.0
    if hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "SUPCON") and bool(getattr(cfg.LOSS.SUPCON, "ENABLE", False)):
        supcon_criterion = SupConLoss(temperature=_to_float(getattr(cfg.LOSS.SUPCON, "T", 0.07)))
        supcon_w = _to_float(getattr(cfg.LOSS.SUPCON, "W", 0.0))
        supcon_enabled = True
        print(f"[make_loss] SupCon enabled: W={supcon_w}, T={getattr(cfg.LOSS.SUPCON, 'T', 0.07)}")

    # ---- Feature sources (defaults) ----
    tri_src_triplet  = str(getattr(getattr(cfg.LOSS, "TRIPLET", {}),  "FEAT_SRC", "pre_bn")).lower()
    tri_src_tripletx = str(getattr(getattr(cfg.LOSS, "TRIPLETX", {}), "FEAT_SRC", "pre_bn")).lower()
    sup_src          = str(getattr(getattr(cfg.LOSS, "SUPCON", {}),   "FEAT_SRC", "bnneck")).lower()
    print("[make_loss][init] Triplet  src:", tri_src_triplet)
    print("[make_loss][init] TripletX src:", tri_src_tripletx)
    print("[make_loss][init] SupCon   src:", sup_src)

    _warn_once_tri = {"warned": False}
    _warn_once_sup = {"warned": False, "model_src_noted": False}

    # ---- Metric loss selection ----
    triplet = None
    use_tx = False
    if "triplet" in str(getattr(cfg.MODEL, "METRIC_LOSS_TYPE", "")).lower() and tri_w > 0:
        use_tx = (
            TripletXLoss is not None
            and hasattr(cfg.LOSS, "TRIPLETX")
            and bool(getattr(cfg.LOSS.TRIPLETX, "ENABLE", False))
        )
        if use_tx:
            triplet = TripletXLoss(
                margin=_to_float(getattr(cfg.LOSS.TRIPLETX, "MARGIN", 0.3)),
                normalize_feature=False,   # no external normalization
                k=int(getattr(cfg.LOSS.TRIPLETX, "K", 4)),
            )
            print("[make_loss] Using TripletX loss (no external normalization).")
        else:
            # baseline TripletLoss: (feat, target)
            if bool(getattr(cfg.MODEL, "NO_MARGIN", False)):
                triplet = TripletLoss()
                print("[make_loss] Using soft (no-margin) Triplet loss")
            else:
                triplet = TripletLoss(_to_float(getattr(cfg.SOLVER, "MARGIN", 0.3)))
                print(f"[make_loss] Using standard Triplet loss (margin={getattr(cfg.SOLVER, 'MARGIN', 0.3)})")

    # -----------------------------
    # Loss function
    # -----------------------------
    def loss_func(score, feat, target, camids=None, z_supcon=None, epoch=None, feat_bn=None):
        total = 0.0

        # ---- CE / ID loss (baseline-safe list handling) ----
        if id_w > 0:
            if isinstance(score, list):
                if xent is not None:
                    id_list = [xent(s, target) for s in score[1:]] if len(score) > 1 else []
                    head0   = xent(score[0], target)
                else:
                    id_list = [F.cross_entropy(s, target) for s in score[1:]] if len(score) > 1 else []
                    head0   = F.cross_entropy(score[0], target)
                ID_LOSS = 0.5 * (_safe_mean(id_list) + head0)
            else:
                ID_LOSS = xent(score, target) if xent else F.cross_entropy(score, target)
            total += id_w * ID_LOSS

        # ---- Prepare feature sources ----
        global_pre_bn, bnneck = _unpack_feats_for_sources(feat, feat_bn)

        # ---- Triplet / TripletX (baseline-safe list handling; no external L2) ----
        if triplet is not None and tri_w > 0:
            # safe core to support TripletX(camids/epoch) and baseline Triplet(no extras)
            def _call_triplet_core(f):
                try:
                    return _reduce_triplet_output(triplet(f, target, camids=camids, epoch=epoch))
                except TypeError:
                    try:
                        return _reduce_triplet_output(triplet(f, target, camids=camids))
                    except TypeError:
                        return _reduce_triplet_output(triplet(f, target))

            if isinstance(feat, list):
                tri_list = [_call_triplet_core(f) for f in (feat[1:] if len(feat) > 1 else [])]
                tri0     = _call_triplet_core(feat[0])
                TRI_LOSS = 0.5 * (_safe_mean(tri_list) + tri0)
            else:
                tri_src = tri_src_tripletx if use_tx else tri_src_triplet
                tri_in  = _pick_by_src_with_fallback(tri_src, global_pre_bn, bnneck, _warn_once_tri, "Triplet")
                TRI_LOSS = _call_triplet_core(tri_in)

            total += tri_w * TRI_LOSS

        # ---- SupCon (kept for compatibility; often disabled in baseline) ----
        if supcon_enabled and supcon_w > 0:
            z = z_supcon if (z_supcon is not None) else _pick_by_src_with_fallback(sup_src, global_pre_bn, bnneck, _warn_once_sup, "SupCon")
            total += supcon_w * supcon_criterion(z, target, camids)

        return total

    return loss_func, center_criterion
