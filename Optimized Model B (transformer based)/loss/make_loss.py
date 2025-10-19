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
#
# [2025-09-22 | Zeyu Yang] Added new branch for sampler == "random" (SupCon-only mode).
#
# [2025-10-06 | Meng Fanyi] Fixed NameError when METRIC_LOSS_TYPE=none (triplet undefined).
#                           Added graceful fallback for SupCon-only (self-supervised) training.
#
# [2025-10-08 | Hang Zhang] Integrated TripletX path (B2/B4) with full argument binding from
#                           cfg.LOSS.TRIPLETX (BNNeck+L2, top-k mining, margin warmup,
#                           camera-aware, hardness-aware).
#                           Unified SupCon usage for B1/B2/B4 with safe z_supcon fallback.
#                           Added sampler='random' branch for B3 (SSL pretraining).
#                           Made triplet optional; added camids/epoch passthrough for TripletX.
#
# [2025-10-11 | Hang Zhang] Compatibility fix (Plan B):
#                           - Conditionally pass camids/epoch only when TRIPLETX.ENABLE=True.
#                           - Keep plain TripletLoss two-arg call in B0.
#                           - Robust handling for tuple/single return from metric loss.
#
# [2025-10-14 | Hang Zhang] Phased-loss scheduling (B2_phased_loss):
#                           - A(0–30): TripletX, w_metric=1.2, w_sup=0.30
#                           - B(30–60): Triplet,  w_metric=1.0, w_sup linearly 0.30→0.15
#                           - C(60–120): Triplet, w_metric=1.0, w_sup=0.0
#                           - If epoch is None → fallback to static cfg weights.
#
# [2025-10-14 | Hang Zhang] Guard switch to avoid affecting other configs.
#                           - Add cfg.LOSS.PHASED.ENABLE (default False).
#                           - Only when PHASED.ENABLE=True AND epoch is not None,
#                             run phased scheduling; otherwise use static weights.
#
# [2025-10-14 | Zeyu Yang] Triplet compatibility and gating:
#                           - Compute Triplet loss only if MODEL.TRIPLET_LOSS_WEIGHT > 0
#                           - Pass (camids, epoch) only when using TripletX; omit for plain Triplet
#                           - Fallback support for implementations without epoch argument
#                           - Added weight gating for both ID and SupCon losses as well
#
# [2025-10-15 | Hang Zhang] Safety & logging improvements:
#                           - Added _safe_mean() to avoid ZeroDivisionError when using list heads
#                           - Added one-time phase-combo logging (combo + weights) whenever changed
#                           - Removed unused LabelSmoothingCrossEntropy import
#
# [2025-10-15 | Hang Zhang] Configurable phased schedule via CLI/YAML:
#                           - _phased_selector now reads cfg.LOSS.PHASED:
#                             BOUNDARIES, METRIC_SEQ, W_METRIC_SEQ, W_SUP_SPEC
#                           - Supports 'const:x' and 'linear:x->y' for SupCon weight
#                           - Falls back to original A/B/C if not provided
#
# [2025-10-15 | Hang Zhang] CLI-switchable feature source (layer) for TripletX & SupCon
#                           - Read cfg.LOSS.TRIPLETX.FEAT_SRC and cfg.LOSS.SUPCON.FEAT_SRC
#                             ('pre_bn' | 'bnneck') to pick features dynamically.
#                           - Backward compatible: if BNNeck feature is not provided, gracefully
#                             fall back to pre_bn with a one-time warning.
#                           - Extended loss_func signature with optional feat_bn=None to accept
#                             BNNeck feature without breaking old trainers.
#
# [2025-10-15 | Hang Zhang] Init logging of feature sources
#                           - Print once at init: only the chosen sources for ENABLED losses.
#                           - Skip disabled losses in the init line for clarity.
#
# [2025-10-17 | Hang Zhang] Single-source policy for SupCon (log & behavior)
#                           - If model returns z_supcon, always use it and IGNORE LOSS.SUPCON.FEAT_SRC.
#                           - Added one-time info log:
#                             "[make_loss][combo] [SupCon] src=model::(z_supcon) | loss_routing=off | L2 in SupConLoss"
#
# [2025-10-17 | Hang Zhang] Stability & clarity
#                           - _supcon_weight_by_epoch takes (cfg, epoch) explicitly.
#                           - Note: "SupConLoss instantiated here (entry should not rebuild)".
#                           - Note: "CE.FEAT_SRC is removed; CE uses classifier score."
#
# [2025-10-17 | Hang Zhang] Triplet vs TripletX independent FEAT_SRC
#                           - Read LOSS.TRIPLET.FEAT_SRC and LOSS.TRIPLETX.FEAT_SRC separately.
#                           - Choose FEAT_SRC based on whether TripletX is used in current step.
#                           - TripletX(normalize_feature=False); removed redundant external L2 normalization.
#                           - One-time check log: "[make_loss][combo] [Metric] ... | ext_L2=off".
#
# [2025-10-18 | Hang Zhang] Removed normalization inside make_loss
#                           - Deleted F.normalize for Triplet, TripletX, and SupCon (handled internally).
#
# [2025-10-18 | Hang Zhang] Baseline-safe ID/Metric list handling & tri_src fix
#                           - CE: support list outputs with 0.5*(head0 + mean(others)); single Tensor unchanged.
#                           - Triplet: support list features with same 0.5 scheme; single Tensor unchanged.
#                           - Fix: Triplet uses LOSS.TRIPLET.FEAT_SRC; TripletX uses LOSS.TRIPLETX.FEAT_SRC.
#
# [2025-10-18 | Hang Zhang] Compatibility for baseline Triplet
#                           - Safe fallback for TripletLoss without (camids/epoch) arguments.
#
# [2025-10-19 | Hang Zhang] Remove unused helper & fix CE policy
#                           - Deleted unused _pick_z() helper.
#                           - Explicit note: CE has no dynamic FEAT_SRC; always uses logits (score).
#
# [2025-10-19 | Hang Zhang] Accurate combo logs (model vs cfg sources)
#                           - SupCon combo logs: model::(z_supcon) vs cfg::<src> with fallback hints.
#                           - Metric combo logs: TripletX/Triplet log once; ext_L2=off; BNNeck-missing warns once.
#
# [2025-10-19 | Hang Zhang] Dynamic metric combo logging on phase switch
#                           - metric_last mechanism: automatically re-log current Metric on switch.
#
# [2025-10-20 | Hang Zhang] Clean init feat_src logs
#                           - Print only enabled losses in [init][feat_src] line.
#                           - Unified author names in all entries.
# ==========================================


import torch
import torch.nn.functional as F

from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .supcon_loss import SupConLoss


# ------------------------------------------------------------
# Optional: Enhanced TripletX (used in B2/B4)
# ------------------------------------------------------------
try:
    from .tripletx_loss import TripletLossX as TripletXLoss
except Exception:
    TripletXLoss = None



# ============================================================
# Utilities
# ============================================================
def _to_float(x, default: float = 0.0):
    """
    Convert x to float with a safe fallback.
    """
    try:
        return float(x)
    except Exception:
        return float(default)



def _safe_mean(values):
    """
    Safe mean: returns 0.0 when the list is empty to avoid ZeroDivisionError.
    """
    return (sum(values) / len(values)) if values else 0.0



def _reduce_triplet_output(out):
    """
    Some triplet implementations return (loss, aux). Normalize to a single scalar.
    """
    return out[0] if isinstance(out, (list, tuple)) else out



def _unpack_feats_for_sources(feat, feat_bn):
    """
    Recover (global_pre_bn, bnneck_feat) for flexible feature routing.

    Expectations:
      - If model returns a list/tuple: feat[0] is global pre-bn, feat[1] is bnneck (when available).
      - If feat_bn is provided explicitly, it overrides missing bnneck from the list.
      - If only a single Tensor is provided, treat it as global pre-bn.
    """
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



def _pick_by_src_with_fallback(src: str,
                               global_pre_bn: torch.Tensor,
                               bnneck_feat: torch.Tensor,
                               once_flag: dict,
                               warn_tag: str):
    """
    Select feature tensor according to FEAT_SRC with a one-time warning fallback.

    Behavior:
      - If src == 'bnneck' but bnneck tensor is missing, fall back to pre_bn and warn once.
      - If src == 'pre_bn', use global_pre_bn silently.
    """
    src = (src or "pre_bn").lower()

    if src == "bnneck":
        if isinstance(bnneck_feat, torch.Tensor):
            return bnneck_feat

        if not once_flag.get("warned", False):
            print(
                f"[make_loss][warn] {warn_tag}: cfg requested FEAT_SRC='bnneck' "
                f"but BNNeck tensor was not provided by model/trainer; "
                f"falling back to 'pre_bn'. (This prints once.)"
            )
            once_flag["warned"] = True

        return global_pre_bn

    # src == 'pre_bn'
    return global_pre_bn



# ============================================================
# Factory
# ============================================================
def make_loss(cfg, num_classes):
    """
    Construct loss function closure and center_criterion (if used).

    This factory:
      - Builds ID (CE), Triplet or TripletX, and SupCon losses according to cfg.
      - Implements phased scheduling (guarded by LOSS.PHASED.ENABLE).
      - Prints concise, accurate logs that reflect only ENABLED losses and their sources.
    """
    # --------------------------------------------------------
    # Read common knobs and instantiate optional criteria
    # --------------------------------------------------------
    sampler = str(getattr(cfg.DATALOADER, "SAMPLER", "softmax_triplet")).lower()

    feat_dim = 2048
    use_gpu = torch.cuda.is_available()
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=use_gpu)

    id_w  = _to_float(getattr(cfg.MODEL, "ID_LOSS_WEIGHT",       1.0))
    tri_w = _to_float(getattr(cfg.MODEL, "TRIPLET_LOSS_WEIGHT",  1.0))



    # --------------------------------------------------------
    # Cross-Entropy (ID) loss
    # --------------------------------------------------------
    xent = None

    if getattr(cfg.MODEL, "IF_LABELSMOOTH", "off") == "on":
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print(f"[make_loss] Label smoothing ON, num_classes={num_classes}")

    # CE uses classifier logits directly (no FEAT_SRC selection).
    ce_src = "bnneck"   # Shown only for display consistency.



    # --------------------------------------------------------
    # SupCon criterion (may be disabled)
    # --------------------------------------------------------
    supcon_enabled = bool(getattr(getattr(cfg.LOSS, "SUPCON", {}), "ENABLE", False))

    supcon_criterion = SupConLoss(
        temperature=_to_float(getattr(cfg.LOSS.SUPCON, "T", 0.07))
    ) if supcon_enabled else None

    supcon_w_static = _to_float(getattr(cfg.LOSS.SUPCON, "W", 0.0))
    sup_src = str(getattr(getattr(cfg.LOSS, "SUPCON", {}), "FEAT_SRC", "bnneck")).lower()

    if supcon_enabled:
        print(f"[make_loss] SupCon enabled: W={supcon_w_static}, T={getattr(cfg.LOSS.SUPCON, 'T', 0.07)}")



    # --------------------------------------------------------
    # Feature-source preferences for metric losses
    # --------------------------------------------------------
    tri_src_triplet  = str(getattr(getattr(cfg.LOSS, "TRIPLET",  {}), "FEAT_SRC", "pre_bn")).lower()
    tri_src_tripletx = str(getattr(getattr(cfg.LOSS, "TRIPLETX", {}), "FEAT_SRC", "pre_bn")).lower()



    # --------------------------------------------------------
    # Init print: only show ENABLED losses to reflect actual plan
    # --------------------------------------------------------
    enabled_srcs = [f"CE={ce_src} (fixed)"]

    if tri_w > 0:
        if bool(getattr(cfg.LOSS.TRIPLETX, "ENABLE", False)):
            enabled_srcs.append(f"TripletX={tri_src_tripletx}")
        elif "triplet" in str(getattr(cfg.MODEL, "METRIC_LOSS_TYPE", "")).lower():
            enabled_srcs.append(f"Triplet={tri_src_triplet}")

    if supcon_enabled:
        enabled_srcs.append(f"SupCon={sup_src}")

    print("[init][feat_src] " + " | ".join(enabled_srcs))



    # --------------------------------------------------------
    # Warn-once state for missing bnneck fallback
    # --------------------------------------------------------
    _warn_once_tri = {"warned": False}
    _warn_once_sup = {"warned": False}



    # --------------------------------------------------------
    # One-time combo log states
    # --------------------------------------------------------
    _combo_logged = {
        "metric_last":  None,   # remember last printed metric kind (Triplet / TripletX)
        "supcon_model": False,  # printed once when using model::z_supcon
        "supcon_cfg":   False,  # printed once when using cfg::<src>
    }



    # --------------------------------------------------------
    # Build metric (Triplet / TripletX) according to cfg
    # --------------------------------------------------------
    triplet = None
    use_tx = False

    if "triplet" in str(getattr(cfg.MODEL, "METRIC_LOSS_TYPE", "")).lower() and tri_w > 0:

        use_tx = (
            TripletXLoss is not None
            and hasattr(cfg.LOSS, "TRIPLETX")
            and bool(getattr(cfg.LOSS.TRIPLETX, "ENABLE", False))
        )

        if use_tx:
            normalize_feature = bool(getattr(cfg.LOSS.TRIPLETX, "NORM_FEAT", True))
            k_value           = int( getattr(cfg.LOSS.TRIPLETX, "K", 4) )

            triplet = TripletXLoss(
                margin=_to_float(getattr(cfg.LOSS.TRIPLETX, "MARGIN", 0.3)),
                normalize_feature=normalize_feature,
                k=k_value,
            )

            print(f"[make_loss] Using TripletX loss (normalize_feature={normalize_feature}, k={k_value}; ext_L2=off)")

        else:
            if bool(getattr(cfg.MODEL, "NO_MARGIN", False)):
                triplet = TripletLoss()
                print("[make_loss] Using soft (no-margin) Triplet loss")
            else:
                triplet = TripletLoss(_to_float(getattr(cfg.SOLVER, "MARGIN", 0.3)))
                print(f"[make_loss] Using standard Triplet loss (margin={getattr(cfg.SOLVER, 'MARGIN', 0.3)})")



    # --------------------------------------------------------
    # Phased-loss switches (guarded)
    # --------------------------------------------------------
    phased_on = (
        hasattr(cfg, "LOSS")
        and hasattr(cfg.LOSS, "PHASED")
        and bool(getattr(cfg.LOSS.PHASED, "ENABLE", False))
    )

    tripletx_end = int(getattr(cfg.LOSS.PHASED, "TRIPLETX_END", 30))   # used only when phased_on



    # --------------------------------------------------------
    # SupCon weight scheduling (guarded by phased_on)
    # --------------------------------------------------------
    sup_w0     = _to_float(getattr(cfg.LOSS.SUPCON, "W0",         supcon_w_static))
    decay_type = str(       getattr(cfg.LOSS.SUPCON, "DECAY_TYPE", "linear")).lower()
    decay_st   = int(       getattr(cfg.LOSS.SUPCON, "DECAY_START", 0))
    decay_ed   = int(       getattr(cfg.LOSS.SUPCON, "DECAY_END",   getattr(cfg.SOLVER, "MAX_EPOCHS", 120)))



    def _supcon_weight_by_epoch(epoch: int) -> float:
        """
        Compute the effective SupCon weight for the given epoch.

        - If phased mode is OFF or epoch is None, return static W (backward-compatible).
        - Otherwise, apply linear / const / exp decay according to cfg.
        """
        if not (phased_on and (epoch is not None)):
            return supcon_w_static

        if decay_type == "linear":
            if epoch <= decay_st:
                return max(0.0, sup_w0)
            if epoch >= decay_ed:
                return 0.0
            frac = float(epoch - decay_st) / float(max(decay_ed - decay_st, 1))
            return max(0.0, sup_w0 * (1.0 - frac))

        elif decay_type == "const":
            return max(0.0, sup_w0)

        elif decay_type == "exp":
            import math
            if epoch < decay_st:
                return max(0.0, sup_w0)
            return max(0.0, sup_w0 * math.exp(-0.05 * (epoch - decay_st)))

        # fallback
        return max(0.0, sup_w0)



    # ========================================================
    # Loss function closure
    # ========================================================
    def loss_func(score,
                  feat,
                  target,
                  camids=None,
                  z_supcon=None,
                  epoch=None,
                  feat_bn=None):
        """
        Compute total loss = ID (optional) + Metric (Triplet/TripletX, optional) + SupCon (optional).

        Parameters:
          - score:   logits or list of logits (for multi-head models)
          - feat:    features (Tensor or list/tuple of features)
          - target:  ground-truth labels (LongTensor)
          - camids:  camera IDs (optional, used by TripletX and SupCon)
          - z_supcon:pre-computed SupCon embedding from model (if provided, overrides FEAT_SRC)
          - epoch:   current epoch index (for phased scheduling; can be None to use static weights)
          - feat_bn: BNNeck feature (optional; used when FEAT_SRC='bnneck' is requested)
        """
        total = 0.0



        # ----------------------------------------------------
        # 1) CE / ID loss
        # ----------------------------------------------------
        if id_w > 0:

            if isinstance(score, list):
                # Multi-head: 0.5 * (head0 + mean(other heads))
                if xent is not None:
                    id_list = [xent(s, target) for s in score[1:]] if len(score) > 1 else []
                    head0   = xent(score[0], target)
                else:
                    id_list = [F.cross_entropy(s, target) for s in score[1:]] if len(score) > 1 else []
                    head0   = F.cross_entropy(score[0], target)

                ID_LOSS = 0.5 * (_safe_mean(id_list) + head0)

            else:
                # Single head
                ID_LOSS = xent(score, target) if xent else F.cross_entropy(score, target)

            total += id_w * ID_LOSS



        # ----------------------------------------------------
        # Prepare feature sources for metric / SupCon
        # ----------------------------------------------------
        global_pre_bn, bnneck = _unpack_feats_for_sources(feat, feat_bn)



        # ----------------------------------------------------
        # 2) Metric: Triplet or TripletX (skipped when sampler=='random')
        # ----------------------------------------------------
        phase_use_tx = use_tx

        if phased_on and (epoch is not None) and (TripletXLoss is not None):
            # Only switch if TripletX path is available and phased mode is enabled.
            phase_use_tx = (int(epoch) <= int(tripletx_end))

        if (sampler != "random") and (triplet is not None) and (tri_w > 0):

            # Print combo line when metric kind changes (Triplet <-> TripletX)
            current_metric = "TripletX" if phase_use_tx else "Triplet"

            if _combo_logged.get("metric_last") != current_metric:

                chosen_src = (tri_src_tripletx if phase_use_tx else tri_src_triplet)
                maybe_fb   = " | fallback=pre_bn_if_bnneck_missing" if chosen_src == "bnneck" else ""
                print(f"[make_loss][combo] [Metric] {current_metric} src=cfg::{chosen_src}{maybe_fb} | ext_L2=off")

                _combo_logged["metric_last"] = current_metric



            def _call_triplet_core(f):
                """
                Call triplet loss with a signature compatible path.
                Try (f, target, camids, epoch) -> then (f, target, camids) -> then (f, target).
                """
                try:
                    return _reduce_triplet_output(triplet(f, target, camids=camids, epoch=epoch))
                except TypeError:
                    try:
                        return _reduce_triplet_output(triplet(f, target, camids=camids))
                    except TypeError:
                        return _reduce_triplet_output(triplet(f, target))



            # Multi-head features: 0.5*(head0 + mean(other heads))
            if isinstance(feat, list):
                tri_list = [_call_triplet_core(f) for f in (feat[1:] if len(feat) > 1 else [])]
                tri0     = _call_triplet_core(feat[0])
                TRI_LOSS = 0.5 * (_safe_mean(tri_list) + tri0)

            else:
                tri_src = (tri_src_tripletx if phase_use_tx else tri_src_triplet)
                tri_in  = _pick_by_src_with_fallback(tri_src, global_pre_bn, bnneck, _warn_once_tri, "Triplet")
                TRI_LOSS = _call_triplet_core(tri_in)

            total += tri_w * TRI_LOSS



        # ----------------------------------------------------
        # 3) SupCon (if enabled and effective weight > 0)
        # ----------------------------------------------------
        if supcon_enabled:

            eff_w = _supcon_weight_by_epoch(epoch)

            if eff_w > 0.0:

                # Use model-provided z_supcon first (single-source policy)
                if z_supcon is not None:

                    if not _combo_logged["supcon_model"]:
                        print("[make_loss][combo] [SupCon] src=model::(z_supcon) | loss_routing=off | L2 in SupConLoss")
                        _combo_logged["supcon_model"] = True

                    z = z_supcon

                else:
                    # Otherwise, route according to cfg FEAT_SRC with fallback
                    if not _combo_logged["supcon_cfg"]:
                        maybe_fb = " | fallback=pre_bn_if_bnneck_missing" if sup_src == "bnneck" else ""
                        print(f"[make_loss][combo] [SupCon] src=cfg::{sup_src}{maybe_fb} | loss_routing=on | L2 in SupConLoss")
                        _combo_logged["supcon_cfg"] = True

                    z = _pick_by_src_with_fallback(sup_src, global_pre_bn, bnneck, _warn_once_sup, "SupCon")

                total += eff_w * supcon_criterion(z, target, camids)



        # ----------------------------------------------------
        # Return combined loss
        # ----------------------------------------------------
        return total



    # --------------------------------------------------------
    # Return closure and center criterion
    # --------------------------------------------------------
    return loss_func, center_criterion
