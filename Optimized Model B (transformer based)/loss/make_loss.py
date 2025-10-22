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
#                           - B(30–60): Triplet,  w_metric=1.0, w_sup linearly 0.30->0.15
#                           - C(60–120): Triplet, w_metric=1.0, w_sup=0.0
#                           - If epoch is None -> fallback to static cfg weights.
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
#                             "[make_loss][combo] [SupCon] src=model::(z_supcon) | loss_routing=off | L2 in SupConLoss".
#
# [2025-10-17 | Hang Zhang] Stability & clarity
#                           - _supcon_weight_by_epoch takes (cfg, epoch) explicitly.
#                           - Note: "SupConLoss instantiated here (entry should not rebuild)".
#                           - Note: "CE.FEAT_SRC is removed; CE uses classifier score.".
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
# [2025-10-20 | Hang Zhang] CE combo line (one-time) and wording
#                           - Added one-time CE combo line: "[ID] CE src=model::logits | norm=none | label_smoothing=on/off".
#
# [2025-10-20 | Hang Zhang] SupCon schedule checkpoint logging
#                           - Log when SupCon decay starts/phase switches/weight drops to zero or re-enables.
#                           - New helpers: _phase_tag; rounded weight snapshot for stable comparison.
#
# [2025-10-20 | Hang Zhang] ACTIVE combo snapshot
#                           - Print a concise "[combo][ACTIVE][<phase>] ..." line only when the active set changes.
#
# [2025-10-21 | Hang Zhang] Clarified Metric logging vs TripletX (no behavior change)
#                           - If TRIPLETX.ENABLE=True but Metric is gated off by weight/config:
#                               * Print one-time info: TripletX in cfg but gated (w=0 or METRIC_LOSS_TYPE!=triplet).
#                               * ACTIVE line shows "Metric(off, TripletX:on, w=0)" or
#                                 "Metric(off, TripletX:on, cfg.METRIC_LOSS_TYPE!=triplet)".
#
# [2025-10-21 | Hang Zhang] **FIX**: Decouple TripletX weight from baseline triplet
#                           - Read TripletX weight from cfg.LOSS.TRIPLETX.W (tx_w).
#
# [2025-10-22 | Hang Zhang] **Brightness-aware SupCon boosting (optional)**                 # [NEW]
#                           - loss_func accepts optional dark_mask/dark_weight.              # [NEW]
#                           - SupCon recomposed as scale*(all + beta*dark).                  # [NEW]
#
# [2025-10-22 | Hang Zhang] **Brightness-aware Metric split by subset (minimal change)**    # [NEW]
#                           - If dark_mask is provided:                                      # [NEW]
#                               * normal subset -> Triplet (weight=tri_w)                    # [NEW]
#                               * dark   subset -> TripletX (weight=tx_w)                    # [NEW]
#                             Subset sizes <2 are safely skipped; otherwise phased behavior  # [NEW]
#                             remains default when dark_mask is None.                        # [NEW]
#
# [2025-10-23 | Hang Zhang] **FIX**: Read SUPCON.DECAY_END with fallback to SOLVER.MAX_EPOCHS
#                           - decay_ed = cfg.LOSS.SUPCON.DECAY_END if set else cfg.SOLVER.MAX_EPOCHS.
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


# ============================================================
# Utilities
# ============================================================

def _to_float(x, default: float = 0.0):
    """Safe float cast with fallback."""
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_mean(values):
    """Safe mean: returns 0.0 when the list is empty to avoid ZeroDivisionError."""
    return (sum(values) / len(values)) if values else 0.0


def _reduce_triplet_output(out):
    """Some triplet implementations return (loss, aux). Normalize to a single scalar."""
    return out[0] if isinstance(out, (list, tuple)) else out


def _unpack_feats_for_sources(feat, feat_bn):
    """Recover (global_pre_bn, bnneck_feat) for flexible feature routing.

    Expectations:
      - If model returns a list/tuple: feat[0] is global pre-bn, feat[1] is bnneck (when available).
      - If feat_bn is provided explicitly, it overrides missing bnneck from the list.
      - If only a single Tensor is provided, treat it as global pre-bn.
    """
    g, b = None, None

    if isinstance(feat_bn, torch.Tensor):
        b = feat_bn

    if (
        isinstance(feat, (tuple, list))
        and len(feat) >= 2
        and isinstance(feat[0], torch.Tensor)
        and isinstance(feat[1], torch.Tensor)
    ):
        g, b = feat[0], feat[1]

    elif isinstance(feat, list) and len(feat) >= 1 and isinstance(feat[0], torch.Tensor):
        g = feat[0]

    elif isinstance(feat, torch.Tensor):
        g = feat

    return g, b


def _pick_by_src_with_fallback(src: str, global_pre_bn: torch.Tensor, bnneck_feat: torch.Tensor, once_flag: dict, warn_tag: str):
    """Select tensor by FEAT_SRC with one-time fallback warning."""
    src = (src or "pre_bn").lower()

    if src == "bnneck":
        if isinstance(bnneck_feat, torch.Tensor):
            return bnneck_feat
        if not once_flag.get("warned", False):
            print(
                f"[make_loss][warn] {warn_tag}: cfg requested FEAT_SRC='bnneck' but BNNeck tensor was not provided; "
                f"falling back to 'pre_bn'. (This prints once.)"
            )
            once_flag["warned"] = True
        return global_pre_bn

    return global_pre_bn  # src == 'pre_bn'


# ============================================================
# Factory
# ============================================================

def make_loss(cfg, num_classes):
    """Construct loss function closure and center_criterion.

    Includes:
      - CE (ID), Triplet/TripletX, SupCon.
      - Phased scheduling guarded by LOSS.PHASED.ENABLE.
      - Concise, accurate logs reflecting only ENABLED losses.
    """
    sampler = str(getattr(cfg.DATALOADER, "SAMPLER", "softmax_triplet")).lower()

    feat_dim = 2048
    use_gpu = torch.cuda.is_available()
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=use_gpu)

    id_w  = _to_float(getattr(cfg.MODEL, "ID_LOSS_WEIGHT", 1.0))
    tri_w = _to_float(getattr(cfg.MODEL, "TRIPLET_LOSS_WEIGHT", 1.0))  # baseline Triplet weight
    tx_w  = _to_float(getattr(getattr(cfg.LOSS, "TRIPLETX", {}), "W", 0.0))  # TripletX weight (FIX)

    # --------------------------------------------------------
    # Cross-Entropy (ID) loss
    # --------------------------------------------------------
    xent = None
    if getattr(cfg.MODEL, "IF_LABELSMOOTH", "off") == "on":
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print(f"[make_loss] Label smoothing ON, num_classes={num_classes}")

    # CE uses logits directly (no FEAT_SRC selection).
    ce_src = "bnneck"  # shown for display consistency in init line

    # --------------------------------------------------------
    # SupCon criterion (may be disabled)
    # --------------------------------------------------------
    supcon_enabled = bool(getattr(getattr(cfg.LOSS, "SUPCON", {}), "ENABLE", False))
    supcon_criterion = SupConLoss(temperature=_to_float(getattr(cfg.LOSS.SUPCON, "T", 0.07))) if supcon_enabled else None
    supcon_w_static = _to_float(getattr(cfg.LOSS.SUPCON, "W", 0.0))
    sup_src = str(getattr(getattr(cfg.LOSS, "SUPCON", {}), "FEAT_SRC", "bnneck")).lower()

    if supcon_enabled:
        print(f"[make_loss] SupCon enabled: W={supcon_w_static}, T={getattr(cfg.LOSS.SUPCON, 'T', 0.07)}")

    # --------------------------------------------------------
    # Feature-source preferences for metric losses
    # --------------------------------------------------------
    tri_src_triplet = str(getattr(getattr(cfg.LOSS, "TRIPLET", {}), "FEAT_SRC", "pre_bn")).lower()
    tri_src_tripletx = str(getattr(getattr(cfg.LOSS, "TRIPLETX", {}), "FEAT_SRC", "pre_bn")).lower()

    # NEW: detect cfg TripletX enable for logging clarity
    tripletx_cfg_enabled = bool(getattr(getattr(cfg.LOSS, "TRIPLETX", {}), "ENABLE", False))
    metric_cfg_triplet_mode = ("triplet" in str(getattr(cfg.MODEL, "METRIC_LOSS_TYPE", "")).lower())

    # --------------------------------------------------------
    # Init print: only show ENABLED losses to reflect actual plan
    # --------------------------------------------------------
    enabled_srcs = [f"CE={ce_src} (fixed)"]
    if tripletx_cfg_enabled and tx_w > 0:
        enabled_srcs.append(f"TripletX={tri_src_tripletx}")
    elif tri_w > 0 and "triplet" in str(getattr(cfg.MODEL, "METRIC_LOSS_TYPE", "")).lower():
        enabled_srcs.append(f"Triplet={tri_src_triplet}")
    if supcon_enabled:
        enabled_srcs.append(f"SupCon={sup_src}")
    print("[init][feat_src] " + " | ".join(enabled_srcs))

    if tripletx_cfg_enabled and (tx_w == 0.0 or not metric_cfg_triplet_mode):
        reason = "w=0" if tx_w == 0.0 else "cfg.METRIC_LOSS_TYPE!=triplet"
        print(f"[make_loss][info] TripletX enabled in cfg but gated off ({reason}); Metric will be off.")

    # --------------------------------------------------------
    # Warn-once state for missing bnneck fallback
    # --------------------------------------------------------
    _warn_once_tri = {"warned": False}
    _warn_once_sup = {"warned": False}

    # --------------------------------------------------------
    # One-time combo log states
    # --------------------------------------------------------
    _combo_logged = {
        "metric_last": None,
        "supcon_model": False,
        "supcon_cfg": False,
        "id_once": False,
        "sup_w_last": None,
        "phase_last": None,
        "active_last": None,
    }

    # --------------------------------------------------------
    # Build metric (Triplet / TripletX) according to cfg
    # --------------------------------------------------------
    triplet = None
    use_tx = False
    # [NEW] Build both variants for split-by-subset; keep original 'triplet' for legacy path
    triplet_plain = None  # TripletLoss
    tripletx_obj = None   # TripletXLoss

    if metric_cfg_triplet_mode:
        # TripletX branch (if available & enabled)
        if TripletXLoss is not None and tripletx_cfg_enabled:
            use_tx = True
            normalize_feature = bool(getattr(cfg.LOSS.TRIPLETX, "NORM_FEAT", True))
            k_value = int(getattr(cfg.LOSS.TRIPLETX, "K", 4))
            tripletx_obj = TripletXLoss(
                margin=_to_float(getattr(cfg.LOSS.TRIPLETX, "MARGIN", 0.3)),
                normalize_feature=normalize_feature,
                k=k_value,
            )
            print(f"[make_loss] Using TripletX loss (normalize_feature={normalize_feature}, k={k_value}; ext_L2=off)")
        # Plain Triplet
        if bool(getattr(cfg.MODEL, "NO_MARGIN", False)):
            triplet_plain = TripletLoss()
            if not use_tx:
                triplet = triplet_plain
                print("[make_loss] Using soft (no-margin) Triplet loss")
        else:
            triplet_plain = TripletLoss(_to_float(getattr(cfg.SOLVER, "MARGIN", 0.3)))
            if not use_tx:
                triplet = triplet_plain
                print(f"[make_loss] Using standard Triplet loss (margin={getattr(cfg.SOLVER, 'MARGIN', 0.3)})")
        # For legacy path keep 'triplet' pointing to current phase choice (TripletX or Triplet)
        if use_tx:
            triplet = tripletx_obj

    # --------------------------------------------------------
    # Phased-loss switches (guarded)
    # --------------------------------------------------------
    phased_on = (
        hasattr(cfg, "LOSS")
        and hasattr(cfg.LOSS, "PHASED")
        and bool(getattr(cfg.LOSS.PHASED, "ENABLE", False))
    )
    tripletx_end = int(getattr(getattr(cfg.LOSS, "PHASED", {}), "TRIPLETX_END", 30))

    # --------------------------------------------------------
    # SupCon weight scheduling (guarded by phased_on)
    # --------------------------------------------------------
    sup_w0 = _to_float(getattr(getattr(cfg.LOSS, "SUPCON", {}), "W0", supcon_w_static))
    decay_type = str(getattr(getattr(cfg.LOSS, "SUPCON", {}), "DECAY_TYPE", "linear")).lower()
    decay_st = int(getattr(getattr(cfg.LOSS, "SUPCON", {}), "DECAY_START", 0))
    decay_ed = int(getattr(getattr(cfg.LOSS, "SUPCON", {}), "DECAY_END",
                           getattr(getattr(cfg, "SOLVER", {}), "MAX_EPOCHS", 120)))

    def _supcon_weight_by_epoch(epoch: int) -> float:
        """Compute the effective SupCon weight for the given epoch."""
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
        return max(0.0, sup_w0)

    def _phase_tag(epoch: int) -> str:
        """Return human-readable phase tag (A/B/C or 'static')."""
        if not (phased_on and (epoch is not None)):
            return "static"
        e = int(epoch)
        if e <= int(tripletx_end):
            return "A"
        if e <= int(decay_ed):
            return "B"
        return "C"

    # --------------------------------------------------------
    # ACTIVE combo logger
    # --------------------------------------------------------
    def _log_active_combo_if_changed(
        phase_tag: str,
        id_on: bool,
        metric_kind: str,
        metric_src: str,
        metric_w: float,
        sup_on: bool,
        sup_src_desc: str,
        sup_w_eff: float,
        label_smoothing_on: bool,
        *,
        tripletx_cfg_enabled: bool,
        metric_cfg_triplet_mode: bool,
    ):
        """Print compact ACTIVE combo only when the signature changes."""
        if metric_kind == "None":
            if tripletx_cfg_enabled and not metric_cfg_triplet_mode:
                metric_text = "Metric(off, TripletX:on, cfg.METRIC_LOSS_TYPE!=triplet)"
            elif tripletx_cfg_enabled and metric_w == 0.0:
                metric_text = "Metric(off, TripletX:on, w=0)"
            else:
                metric_text = "Metric(off)"
        else:
            metric_text = f"Metric={metric_kind}[w={metric_w:.3f}, src={metric_src}]"

        sig = (
            phase_tag,
            ("ID", round(float(id_w), 6), bool(label_smoothing_on)) if id_on else None,
            metric_text,
            ("S", round(float(sup_w_eff), 6), sup_src_desc) if sup_on else None,
        )

        last_sig = _combo_logged["active_last"]
        last_phase = _combo_logged.get("phase_last", None)

        if last_sig != sig or last_phase != phase_tag:
            parts = [f"[combo][ACTIVE][{phase_tag}]"]
            parts.append(f"ID(w={id_w:.3f}, sm={'on' if label_smoothing_on else 'off'})" if id_on else "ID(off)")
            parts.append(metric_text)
            parts.append(f"SupCon[w={sup_w_eff:.3f}, src={sup_src_desc}]" if sup_on else "SupCon(off)")
            print(" ".join(parts))

            if last_phase is not None and last_phase != phase_tag:
                last_metric_text = "-"
                if isinstance(last_sig, tuple) and len(last_sig) >= 3:
                    last_metric_text = last_sig[2] if isinstance(last_sig[2], str) else "-"
                print(f"[phase] switch {last_phase} -> {phase_tag} | {last_metric_text} -> {metric_text}")

            _combo_logged["active_last"] = sig
            _combo_logged["phase_last"] = phase_tag

    # ========================================================
    # Loss function closure
    # ========================================================
    # [NEW] Accept brightness hints. When provided:
    #       - Metric is split: normal->Triplet, dark->TripletX
    #       - SupCon is boosted on dark subset
    def loss_func(score, feat, target, camids=None, z_supcon=None, epoch=None, feat_bn=None,
                  dark_mask=None, dark_weight=None):
        """Compute total loss = ID + Metric + SupCon (gated by weights/enables)."""
        total = 0.0

        # --- CE / ID loss ---
        if id_w > 0 and not _combo_logged["id_once"]:
            print(
                "[make_loss][combo] [ID] CE src=model::logits | norm=none | label_smoothing=%s"
                % ("on" if xent is not None else "off")
            )
            _combo_logged["id_once"] = True

        if id_w > 0:
            if isinstance(score, list):
                if xent is not None:
                    id_list = [xent(s, target) for s in score[1:]] if len(score) > 1 else []
                    head0 = xent(score[0], target)
                else:
                    id_list = [F.cross_entropy(s, target) for s in score[1:]] if len(score) > 1 else []
                    head0 = F.cross_entropy(score[0], target)
                ID_LOSS = 0.5 * (_safe_mean(id_list) + head0)
            else:
                ID_LOSS = xent(score, target) if xent else F.cross_entropy(score, target)
            total += id_w * ID_LOSS
            # (Optional) per-sample dark weighting for CE could be added here if your CE supports reduction='none'

        # --- Prepare features ---
        global_pre_bn, bnneck = _unpack_feats_for_sources(feat, feat_bn)

        # --- SupCon effective weight & phase ---
        phase = _phase_tag(epoch)
        eff_w = _supcon_weight_by_epoch(epoch)
        eff_w_r = round(float(eff_w), 6)

        if supcon_enabled:
            last_w = _combo_logged["sup_w_last"]
            last_phase = _combo_logged["phase_last"]
            if last_w is None:
                print(f"[make_loss][sup] phase={phase} | effW={eff_w_r:.6f}")
            else:
                if (last_w > 0 and eff_w_r == 0.0):
                    print(f"[make_loss][sup] phase={phase} | SupCon disabled (effW {last_w:.6f} -> 0)")
                elif (last_w == 0.0 and eff_w_r > 0):
                    print(f"[make_loss][sup] phase={phase} | SupCon re-enabled (effW 0 -> {eff_w_r:.6f})")
                elif (last_phase != phase):
                    print(f"[make_loss][sup] phase switch {last_phase} -> {phase} | effW={eff_w_r:.6f}")
            _combo_logged["sup_w_last"] = eff_w_r
            _combo_logged["phase_last"] = phase
        else:
            if _combo_logged["phase_last"] is None:
                _combo_logged["phase_last"] = phase

        # --- Metric: Triplet/TripletX ---
        phase_use_tx = False
        metric_w_eff_for_log = 0.0  # for logger

        if metric_cfg_triplet_mode and ((triplet_plain is not None) or (tripletx_obj is not None)):
            # Default legacy: choose by phase (single metric)
            phase_use_tx = (TripletXLoss is not None and tripletx_cfg_enabled)
            if phased_on and (epoch is not None) and (TripletXLoss is not None):
                phase_use_tx = (int(epoch) <= int(tripletx_end))

            # === Split-by-subset when dark_mask is provided ===
            did_split = False
            if isinstance(dark_mask, torch.Tensor) and (sampler != "random"):
                dm = dark_mask.to(global_pre_bn.device, dtype=torch.bool)
                idx_dark = torch.nonzero(dm, as_tuple=False).view(-1)
                idx_norm = torch.nonzero(~dm, as_tuple=False).view(-1)

                def _call_triplet_plain(feats, tgt):
                    try:
                        return _reduce_triplet_output(triplet_plain(feats, tgt))
                    except TypeError:
                        return _reduce_triplet_output(triplet_plain(feats, tgt))

                def _call_tripletx(feats, tgt, cams, ep):
                    try:
                        return _reduce_triplet_output(tripletx_obj(feats, tgt, camids=cams, epoch=ep))
                    except TypeError:
                        try:
                            return _reduce_triplet_output(tripletx_obj(feats, tgt, camids=cams))
                        except TypeError:
                            return _reduce_triplet_output(tripletx_obj(feats, tgt))

                # Select sources per metric kind
                tri_in_norm = _pick_by_src_with_fallback(tri_src_triplet,  global_pre_bn, bnneck, _warn_once_tri, "Triplet")
                tri_in_dark = _pick_by_src_with_fallback(tri_src_tripletx, global_pre_bn, bnneck, _warn_once_tri, "TripletX")

                L_tri_norm = torch.tensor(0.0, device=global_pre_bn.device)
                L_trix_dark = torch.tensor(0.0, device=global_pre_bn.device)

                ok_norm = idx_norm.numel() >= 2 and (tri_w > 0.0) and (triplet_plain is not None)
                ok_dark = idx_dark.numel() >= 2 and (tx_w > 0.0) and (tripletx_obj is not None)

                if ok_norm:
                    feats_n = tri_in_norm[idx_norm] if tri_in_norm.dim() > 1 else tri_in_norm
                    tgt_n = target[idx_norm]
                    L_tri_norm = _call_triplet_plain(feats_n, tgt_n)

                if ok_dark:
                    feats_d = tri_in_dark[idx_dark] if tri_in_dark.dim() > 1 else tri_in_dark
                    tgt_d = target[idx_dark]
                    cams_d = camids[idx_dark] if camids is not None else None
                    L_trix_dark = _call_tripletx(feats_d, tgt_d, cams_d, epoch)

                if ok_norm or ok_dark:
                    total += (tri_w * L_tri_norm) + (tx_w * L_trix_dark)
                    metric_w_eff_for_log = (tri_w if ok_norm else 0.0) + (tx_w if ok_dark else 0.0)
                    # mark split mode for logging
                    phase_use_tx = False
                    did_split = True

            if not did_split and (sampler != "random"):
                # Legacy single-path (unchanged): phase-based metric
                metric_w_eff = (tx_w if phase_use_tx else tri_w)
                if metric_w_eff > 0.0 and (triplet is not None):
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
                        tri0 = _call_triplet_core(feat[0])
                        TRI_LOSS = 0.5 * (_safe_mean(tri_list) + tri0)
                    else:
                        tri_src = (tri_src_tripletx if phase_use_tx else tri_src_triplet)
                        tri_in = _pick_by_src_with_fallback(tri_src, global_pre_bn, bnneck, _warn_once_tri, "Triplet")
                        TRI_LOSS = _call_triplet_core(tri_in)
                    total += metric_w_eff * TRI_LOSS
                    metric_w_eff_for_log = metric_w_eff

        # --- SupCon ---
        if supcon_enabled and eff_w > 0.0:
            if z_supcon is not None:
                if not _combo_logged["supcon_model"]:
                    print("[make_loss][combo] [SupCon] src=model::(z_supcon) | loss_routing=off | L2 in SupConLoss")
                    _combo_logged["supcon_model"] = True
                z = z_supcon
            else:
                if not _combo_logged["supcon_cfg"]:
                    maybe_fb = " | fallback=pre_bn_if_bnneck_missing" if sup_src == "bnneck" else ""
                    print(f"[make_loss][combo] [SupCon] src=cfg::{sup_src}{maybe_fb} | loss_routing=on | L2 in SupConLoss")
                    _combo_logged["supcon_cfg"] = True
                z = _pick_by_src_with_fallback(sup_src, global_pre_bn, bnneck, _warn_once_sup, "SupCon")

            SUP_LOSS = supcon_criterion(z, target, camids)

            # Brightness-aware SupCon boost (optional)
            if isinstance(dark_mask, torch.Tensor):
                dm = dark_mask.to(z.device, dtype=torch.bool)
                idx_dark = torch.nonzero(dm, as_tuple=False).view(-1)
                L_sup_dark = torch.tensor(0.0, device=z.device)
                if idx_dark.numel() >= 2:
                    z_dark = z[idx_dark]
                    tgt_dark = target[idx_dark]
                    cams_dark = camids[idx_dark] if camids is not None else None
                    L_sup_dark = supcon_criterion(z_dark, tgt_dark, cams_dark)
                Nv = int(target.shape[0])
                Nd = int(idx_dark.numel())
                Nn = max(Nv - Nd, 0)
                gamma = 1.8
                beta = (gamma - 1.0) * (Nd / max(Nv, 1))
                scale = Nv / max(Nn + gamma * Nd, 1)
                SUP_LOSS = scale * (SUP_LOSS + beta * L_sup_dark)

            total += eff_w * SUP_LOSS

        # --- ACTIVE combo log ---
        metric_kind = "None"
        metric_src_used = "-"
        if sampler != "random":
            if isinstance(dark_mask, torch.Tensor):
                metric_kind = "Split"  # Triplet(normal) + TripletX(dark)
                metric_src_used = f"Tri:{tri_src_triplet}|TX:{tri_src_tripletx}"
            elif (triplet is not None) and (metric_w_eff_for_log > 0.0):
                metric_kind = "TripletX" if phase_use_tx else "Triplet"
                metric_src_used = (tri_src_tripletx if phase_use_tx else tri_src_triplet)

        sup_on = (supcon_enabled and eff_w > 0.0)
        sup_src_desc = "model::z_supcon" if (z_supcon is not None) else f"cfg::{sup_src}"

        _log_active_combo_if_changed(
            phase_tag=phase,
            id_on=(id_w > 0),
            metric_kind=metric_kind,
            metric_src=metric_src_used,
            metric_w=metric_w_eff_for_log,
            sup_on=sup_on,
            sup_src_desc=sup_src_desc,
            sup_w_eff=eff_w if sup_on else 0.0,
            label_smoothing_on=(xent is not None),
            tripletx_cfg_enabled=tripletx_cfg_enabled,
            metric_cfg_triplet_mode=metric_cfg_triplet_mode,
        )

        return total

    return loss_func, center_criterion
