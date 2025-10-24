import torch


# =============================================================================
# File: make_optimizer.py
# Purpose: Build optimizers for TransReID + TinyCLIP + AFEM framework.
#
# Change Log
# [2025-10-25 | Hang Zhang] **TinyCLIP LR-Scaling Update**
#     - Added MODEL.CLIP_SEPARATE_LR and MODEL.CLIP_LR_SCALE configs.
#     - Automatically detects TinyCLIP parameters (name contains "clip_model").
#     - Applies scaled learning rate: lr = BASE_LR * CLIP_LR_SCALE.
#     - Keeps bias and classifier LR logic unchanged.
#     - Prints debug info on optimizer creation for transparency.
# =============================================================================


def make_optimizer(cfg, model, center_criterion):
    """
    Build two optimizers:
    (1) Main optimizer for all model parameters (TransReID backbone, AFEM, BNNeck,
        classifiers, TinyCLIP, etc.), respecting per-parameter learning rate logic.
    (2) Center-loss optimizer for the center criterion (if enabled).

    Extended functionality:
    - If cfg.MODEL.CLIP_SEPARATE_LR=True, parameters whose names contain "clip_model"
      are assigned a scaled learning rate = BASE_LR * CLIP_LR_SCALE.
      This allows gentle fine-tuning of TinyCLIP without destabilizing pretrained weights.
    """

    params = []

    # -------------------------------------------------------------------------
    # Global solver parameters
    # -------------------------------------------------------------------------
    base_lr = cfg.SOLVER.BASE_LR
    base_wd = cfg.SOLVER.WEIGHT_DECAY
    bias_lr_factor = cfg.SOLVER.BIAS_LR_FACTOR
    bias_wd = cfg.SOLVER.WEIGHT_DECAY_BIAS
    large_fc_lr = cfg.SOLVER.LARGE_FC_LR

    # -------------------------------------------------------------------------
    # TinyCLIP-specific scaling parameters
    # -------------------------------------------------------------------------
    clip_separate = getattr(cfg.MODEL, "CLIP_SEPARATE_LR", True)
    clip_scale = getattr(cfg.MODEL, "CLIP_LR_SCALE", 0.2)

    # -------------------------------------------------------------------------
    # Parameter grouping and learning-rate assignment
    # -------------------------------------------------------------------------
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue  # skip frozen layers

        # Default learning rate and weight decay
        lr = base_lr
        weight_decay = base_wd

        # (1) Bias parameters: usually double LR, custom weight decay
        if "bias" in key:
            lr = base_lr * bias_lr_factor
            weight_decay = bias_wd

        # (2) Fully-connected classifier heads (ArcFace / CosFace / Linear)
        if large_fc_lr and ("classifier" in key or "arcface" in key):
            lr = base_lr * 2
            print("[make_optimizer] Using 2× learning rate for classifier:", key)

        # (3) TinyCLIP branch (vision encoder)
        if clip_separate and "clip_model" in key:
            lr = base_lr * clip_scale

        # Append to optimizer parameter list
        params.append({
            "params": [value],
            "lr": lr,
            "weight_decay": weight_decay
        })

    # -------------------------------------------------------------------------
    # Build main optimizer
    # -------------------------------------------------------------------------
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(
            params,
            momentum=cfg.SOLVER.MOMENTUM
        )
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(
            params,
            lr=base_lr,
            weight_decay=base_wd
        )
    else:
        # e.g. "Adam"
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    # -------------------------------------------------------------------------
    # Build center-loss optimizer (unchanged)
    # -------------------------------------------------------------------------
    optimizer_center = torch.optim.SGD(
        center_criterion.parameters(),
        lr=cfg.SOLVER.CENTER_LR
    )

    # -------------------------------------------------------------------------
    # Debug print for transparency
    # -------------------------------------------------------------------------
    print(f"[make_optimizer] BASE_LR={base_lr} | "
          f"CLIP_SEPARATE_LR={clip_separate} | "
          f"CLIP_LR_SCALE={clip_scale}")

    return optimizer, optimizer_center
