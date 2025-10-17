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
# [2025-10-15 | Team 5703]  **NEW: CLI-switchable feature source (layer) for TripletX & SupCon**
#                           - Read cfg.LOSS.TRIPLETX.FEAT_SRC and cfg.LOSS.SUPCON.FEAT_SRC
#                             ('pre_bn' | 'bnneck') to pick features dynamically.
#                           - Backward compatible: if BNNeck feature is not provided, gracefully
#                             fall back to pre_bn with a one-time warning.
#                           - Extended loss_func signature with optional feat_bn=None to accept
#                             BNNeck feature without breaking old trainers.
# [2025-10-15 | Team 5703]  **NEW: Init logging of feature sources**
#                           - Print once at init: the chosen sources for Triplet/TripletX and SupCon.
#                           - If both use the same layer, print a caution about potential gradient interaction.
# [2025-10-17 | Team 5703]  **Single-source policy for SupCon (log & behavior)**
#                           - If model returns z_supcon, always use it and IGNORE LOSS.SUPCON.FEAT_SRC.
#                           - Added one-time info log:
#                             "[make_loss][info] [SupCon] using z_supcon from model; LOSS.SUPCON.FEAT_SRC ignored."
#                           - Applied to 'softmax', 'softmax_triplet' and 'random' branches.
# [2025-10-17 | Team 5703]  **Stability & clarity**
#                           - _phased_selector now takes (cfg, epoch) explicitly (removed inspect.stack()).
#                           - Added one-time note: "SupConLoss instantiated here (entry should not rebuild)".
#                           - Added one-time note: "CE.FEAT_SRC='<v>' ignored (CE uses classifier score)".
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


def _pick_z(z_supcon, feat):
    """
    Select z_supcon if provided; otherwise fall back to backbone features.
    For transformer_local (feats is list), we use feats[0] = global pre-BN.
    """
    if z_supcon is not None:
        return z_supcon
    return feat[0] if isinstance(feat, list) else feat


def _reduce_triplet_output(out):
    """Return scalar loss if metric loss returns (loss, aux) or a single value."""
    return out[0] if isinstance(out, (list, tuple)) else out


def _unpack_feats_for_sources(feat, feat_bn):
    """
    Try to recover (global_pre_bn, bnneck_feat) from various trainer/model conventions.
    Priority:
      1) Explicit feat_bn argument if tensor.
      2) Tuple-like feat=(global, bn) if both tensors.
      3) For transformer_local: feat is list [global, local1, ...]; bnneck unavailable here unless feat_bn provided.
      4) Fallback: (feat, None)
    """
    g, b = None, None
    if isinstance(feat_bn, torch.Tensor):
        b = feat_bn
    if isinstance(feat, (tuple, list)) and len(feat) >= 2 and isinstance(feat[0], torch.Tensor) and isinstance(feat[1], torch.Tensor):
        g = feat[0]
        b = feat[1]
    elif isinstance(feat, list) and len(feat) >= 1 and isinstance(feat[0], torch.Tensor):
        g = feat[0]
    elif isinstance(feat, torch.Tensor):
        g = feat
    return g, b


def _pick_by_src_with_fallback(src, global_pre_bn, bnneck_feat, once_flag: dict, warn_tag: str):
    """
    Return feature according to src ('pre_bn' | 'bnneck').
    If bnneck is requested but unavailable, fall back to pre_bn with a one-time warning.
    """
    src = (src or "pre_bn").lower()
    if src == "bnneck":
        if isinstance(bnneck_feat, torch.Tensor):
            return bnneck_feat
        if not once_flag.get('warned', False):
            print(f"[make_loss][warn] {warn_tag}: requested FEAT_SRC='bnneck' but BNNeck feature is not provided "
                  f"(trainer/model did not pass it). Falling back to pre_bn.")
            once_flag['warned'] = True
        return global_pre_bn
    return global_pre_bn


# ------------------------------
# Phased scheduling helpers
# ------------------------------
def _supcon_weight_by_epoch(epoch: int) -> float:
    if epoch <= 30:
        return 0.30
    elif epoch <= 60:
        return 0.30 - 0.15 * ((epoch - 30) / 30.0)
    else:
        return 0.0


def _parse_wsup(spec: str, epoch: int, left: int, right: int) -> float:
    try:
        kind, payload = spec.split(":", 1)
        kind = kind.strip().lower()
        if kind == "const":
            return float(payload)
        if kind == "linear":
            v0, v1 = payload.split("->")
            v0, v1 = float(v0), float(v1)
            if right <= left:
                return v1
            t = max(0.0, min(1.0, (epoch - left) / float(right - left)))
            return v0 + (v1 - v0) * t
    except Exception:
        pass
    return 0.0


def _phased_selector(cfg, epoch: int):
    """Explicit cfg-based selector (no inspect)."""
    try:
        phased = cfg.LOSS.PHASED if (cfg is not None and hasattr(cfg, "LOSS")) else None
        if phased is not None and bool(getattr(phased, "ENABLE", False)):
            bounds     = list(getattr(phased, "BOUNDARIES", [30, 60]))
            metrics    = list(getattr(phased, "METRIC_SEQ", ["tripletx", "triplet", "triplet"]))
            w_metric   = list(getattr(phased, "W_METRIC_SEQ", [1.2, 1.0, 1.0]))
            w_sup_spec = list(getattr(phased, "W_SUP_SPEC", ["const:0.30", "linear:0.30->0.15", "const:0.0"]))

            seg_left  = [0] + list(bounds)
            seg_right = list(bounds) + [10**9]

            idx = 0
            for i, (L, R) in enumerate(zip(seg_left, seg_right)):
                if L <= epoch < R:
                    idx = i
                    break

            metric_name = str(metrics[idx]).lower()
            use_tx = (metric_name == "tripletx")
            w_m = float(w_metric[idx])
            w_s = _parse_wsup(str(w_sup_spec[idx]), epoch, seg_left[idx], seg_right[idx])
            return use_tx, w_m, w_s
    except Exception:
        pass

    if epoch <= 30:
        return True, 1.2, 0.30
    elif epoch <= 60:
        return False, 1.0, _supcon_weight_by_epoch(epoch)
    else:
        return False, 1.0, 0.0


# ------------------------------
# Factory
# ------------------------------
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
        supcon_criterion = SupConLoss(temperature=_to_float(getattr(cfg.LOSS.SUPCON, "T", 0.07), 0.07))
        supcon_w = _to_float(getattr(cfg.LOSS.SUPCON, "W", 0.0), 0.0)
        supcon_enabled = True
        print(f"[make_loss] SupCon enabled: W={supcon_w}, T={getattr(cfg.LOSS.SUPCON, 'T', 0.07)}")
        print("[make_loss][init] SupConLoss instantiated here (entry should not rebuild).")

    # NEW: read feature source preferences (with safe defaults reflecting current baseline)
    tri_src = "pre_bn"
    sup_src = "bnneck"
    try:
        if hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "TRIPLETX"):
            tri_src = str(getattr(cfg.LOSS.TRIPLETX, "FEAT_SRC", "pre_bn")).lower()
        if hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "SUPCON"):
            sup_src = str(getattr(cfg.LOSS.SUPCON, "FEAT_SRC", "bnneck")).lower()
    except Exception:
        pass

    # --- one-time display of feature-source config ---
    print("[make_loss][init] Triplet/TripletX feature source:", tri_src)
    print("[make_loss][init] SupCon feature source:", sup_src)
    if tri_src == sup_src:
        print("[make_loss][init] ⚠ Note: Both losses use same feature layer → gradients may interact.")

    # Track warnings & one-time notes
    _warn_once_tri = {"warned": False}
    _warn_once_sup = {"warned": False, "model_src_noted": False}
    _ce_note_once  = {"done": False}

    # --------------------------------
    # Metric loss: Triplet or TripletX
    # --------------------------------
    triplet = None
    USE_TRIPLETX_AVAILABLE = False
    metric_type = str(getattr(cfg.MODEL, "METRIC_LOSS_TYPE", "")).lower()

    if ('triplet' in metric_type) and (tri_w > 0.0):
        USE_TRIPLETX_AVAILABLE = (
            TripletXLoss is not None
            and hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "TRIPLETX")
            and bool(getattr(cfg.LOSS.TRIPLETX, "ENABLE", False))
        )
        if USE_TRIPLETX_AVAILABLE:
            triplet = TripletXLoss(
                margin=_to_float(getattr(cfg.LOSS.TRIPLETX, "MARGIN", 0.3), 0.3),
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
        _static_log_printed = {"done": False}

        def loss_func(score, feat, target, camids=None, z_supcon=None, epoch=None, feat_bn=None):
            total = 0.0

            if not _static_log_printed["done"]:
                combo_name = "NoMetric"
                print(f"[make_loss][phase] sampler=softmax | static combo={combo_name} "
                      f"| id_w={id_w:.3f}, sup_w={(supcon_w if supcon_enabled else 0.0):.3f}")
                _static_log_printed["done"] = True

            # ID loss (CE always uses classifier score; FEAT_SRC is ignored)
            if not _ce_note_once["done"]:
                try:
                    ce_src = str(getattr(cfg.LOSS.CE, "FEAT_SRC", "bnneck"))
                except Exception:
                    ce_src = "bnneck"
                print(f"[make_loss][info] CE.FEAT_SRC='{ce_src}' ignored (CE uses classifier score).")
                _ce_note_once["done"] = True

            if id_w > 0:
                if xent is not None and getattr(cfg.MODEL, "IF_LABELSMOOTH", "off") == 'on':
                    ce = xent(score, target)
                else:
                    ce = F.cross_entropy(score, target)
                total += id_w * ce

            # SupCon (with dynamic source & fallback)
            if supcon_enabled and supcon_w > 0:
                global_pre_bn, bnneck = _unpack_feats_for_sources(feat, feat_bn)
                if z_supcon is not None:
                    if not _warn_once_sup["model_src_noted"]:
                        print("[make_loss][info] [SupCon] using z_supcon from model; LOSS.SUPCON.FEAT_SRC ignored.")
                        _warn_once_sup["model_src_noted"] = True
                    z = z_supcon
                else:
                    sup_input = _pick_by_src_with_fallback(sup_src, global_pre_bn, bnneck,
                                                           _warn_once_sup, "SupCon")
                    z = F.normalize(sup_input, dim=1)
                total += supcon_w * supcon_criterion(z, target, camids)

            return total

    elif sampler == 'softmax_triplet':
        _phase_log_state = {"last": None}

        def loss_func(score, feat, target, camids=None, z_supcon=None, epoch=None, feat_bn=None):
            total = 0.0

            # ID loss (CE always uses classifier score; FEAT_SRC is ignored)
            if not _ce_note_once["done"]:
                try:
                    ce_src = str(getattr(cfg.LOSS.CE, "FEAT_SRC", "bnneck"))
                except Exception:
                    ce_src = "bnneck"
                print(f"[make_loss][info] CE.FEAT_SRC='{ce_src}' ignored (CE uses classifier score).")
                _ce_note_once["done"] = True

            if id_w > 0:
                if isinstance(score, list):
                    if xent is not None and getattr(cfg.MODEL, "IF_LABELSMOOTH", "off") == 'on':
                        id_list = [xent(s, target) for s in score[1:]]
                        head0 = xent(score[0], target)
                    else:
                        id_list = [F.cross_entropy(s, target) for s in score[1:]]
                        head0 = F.cross_entropy(score[0], target)
                    ID_LOSS = 0.5 * (_safe_mean(id_list) + head0)
                else:
                    if xent is not None and getattr(cfg.MODEL, "IF_LABELSMOOTH", "off") == 'on':
                        ID_LOSS = xent(score, target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)
                total += id_w * ID_LOSS

            # phased vs static weights
            phased_on = (
                (epoch is not None)
                and hasattr(cfg, "LOSS")
                and hasattr(cfg.LOSS, "PHASED")
                and bool(getattr(cfg.LOSS.PHASED, "ENABLE", False))
            )
            if phased_on:
                use_tx_phase, w_metric, w_sup = _phased_selector(cfg, int(epoch))
            else:
                use_tx_phase = (TripletXLoss is not None
                                and hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "TRIPLETX")
                                and bool(getattr(cfg.LOSS.TRIPLETX, "ENABLE", False)))
                w_metric = tri_w
                w_sup = supcon_w if supcon_enabled else 0.0

            combo_name = "TripletX" if (use_tx_phase and TripletXLoss is not None) else \
                         ("Triplet" if (triplet is not None) else "NoMetric")
            current_sig = (combo_name, float(w_metric), float(w_sup))
            if _phase_log_state["last"] != current_sig:
                if epoch is None:
                    print(f"[make_loss][phase] sampler=softmax_triplet | static combo={combo_name} "
                          f"| id_w={id_w:.3f}, tri_w={float(w_metric):.3f}, sup_w={float(w_sup):.3f}")
                else:
                    print(f"[make_loss][phase] epoch={int(epoch)} | combo={combo_name} "
                          f"| id_w={id_w:.3f}, tri_w={float(w_metric):.3f}, sup_w={float(w_sup):.3f}")
                _phase_log_state["last"] = current_sig

            # prepare features for dynamic source picking
            global_pre_bn, bnneck = _unpack_feats_for_sources(feat, feat_bn)

            # Triplet / TripletX
            if triplet is not None and w_metric > 0 and global_pre_bn is not None:
                tri_in = _pick_by_src_with_fallback(tri_src, global_pre_bn, bnneck,
                                                    _warn_once_tri, "Triplet/TripletX")
                tri_in = F.normalize(tri_in, dim=1)

                def _call_triplet(f):
                    if use_tx_phase and TripletXLoss is not None:
                        try:
                            return _reduce_triplet_output(triplet(f, target, camids=camids, epoch=epoch))
                        except TypeError:
                            return _reduce_triplet_output(triplet(f, target, camids=camids))
                    else:
                        try:
                            return _reduce_triplet_output(triplet(f, target))
                        except TypeError:
                            return _reduce_triplet_output(triplet(f, target))

                if isinstance(feat, list) and len(feat) > 1 and isinstance(feat[1], torch.Tensor):
                    TRI_LOSS = _call_triplet(tri_in)  # keep global-only by default
                else:
                    TRI_LOSS = _call_triplet(tri_in)

                total += float(w_metric) * TRI_LOSS

            # SupCon
            if supcon_enabled and w_sup > 0:
                if z_supcon is not None:
                    if not _warn_once_sup["model_src_noted"]:
                        print("[make_loss][info] [SupCon] using z_supcon from model; LOSS.SUPCON.FEAT_SRC ignored.")
                        _warn_once_sup["model_src_noted"] = True
                    z = z_supcon
                else:
                    sup_input = _pick_by_src_with_fallback(sup_src, global_pre_bn, bnneck,
                                                           _warn_once_sup, "SupCon")
                    z = F.normalize(sup_input, dim=1)
                total += float(w_sup) * supcon_criterion(z, target, camids)

            return total

    elif sampler == 'random':
        # SSL/SupCon-only mode
        def loss_func(score, feat, target, camids=None, z_supcon=None, epoch=None, feat_bn=None):
            if not (supcon_enabled and supcon_w > 0):
                raise RuntimeError("sampler='random' requires LOSS.SUPCON.ENABLE=True and W>0.")

            global_pre_bn, bnneck = _unpack_feats_for_sources(feat, feat_bn)

            if z_supcon is not None:
                if not _warn_once_sup["model_src_noted"]:
                    print("[make_loss][info] [SupCon] using z_supcon from model; LOSS.SUPCON.FEAT_SRC ignored.")
                    _warn_once_sup["model_src_noted"] = True
                z = z_supcon
            else:
                sup_input = _pick_by_src_with_fallback(sup_src, global_pre_bn, bnneck,
                                                       _warn_once_sup, "SupCon")
                z = F.normalize(sup_input, dim=1)
            return supcon_w * supcon_criterion(z, target, camids)

    else:
        raise ValueError(f"[make_loss] sampler must be one of "
                         f"['softmax', 'softmax_triplet', 'random'], "
                         f"but got '{cfg.DATALOADER.SAMPLER}'")

    return loss_func, center_criterion
