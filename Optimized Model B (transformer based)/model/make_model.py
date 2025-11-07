# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

"""
Merged architecture overview:

┌───────────────────────────────────────────────────────────────────────────────┐
│ cfg.MODEL.USE_CLIP = False                                                   │ → Original TransReID + JPM
│ cfg.MODEL.USE_CLIP = True,  CLIP_IMPL="open_clip", CLIP_FUSION="minimal"     │ → OpenCLIP fusion (global concat + Linear; locals unchanged)
│ cfg.MODEL.USE_CLIP = True,  CLIP_IMPL="open_clip", CLIP_FUSION="quickwin"    │ → OpenCLIP fusion (global + 4×locals with LN + gate + projection)
│ cfg.MODEL.USE_CLIP = True,  CLIP_IMPL="hf"                                   │ → TinyCLIP fusion (global-only; optional AFEM + sem_refine)
└───────────────────────────────────────────────────────────────────────────────┘

Switch summary:
  MODEL.USE_CLIP            Enable/disable CLIP fusion (bool)
  MODEL.CLIP_IMPL           "open_clip" | "hf"
  MODEL.CLIP_FUSION         "minimal" | "quickwin"    (only for CLIP_IMPL="open_clip")

  # open_clip specific:
  MODEL.CLIP_BACKBONE       e.g., "ViT-B-16"
  MODEL.CLIP_PRETRAIN       e.g., "laion2b_s34b_b88k"

  # hf (TinyCLIP) specific:
  MODEL.CLIP_HF_ID          e.g., "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
  MODEL.CLIP_INPUT_SIZE     e.g., (224, 224) (optional override)
  MODEL.CLIP_FINETUNE       bool (enable vision tower finetuning)
  MODEL.CLIP_USE_AFEM       bool (enable AFEM refinement)
  MODEL.CLIP_USE_SEM_REFINE bool (enable semantics refine projection)

Both branches retain their respective:
  – preprocessing (ImageNet→CLIP normalization)
  – resize logic (OpenCLIP: bicubic; TinyCLIP: bilinear)
  – fusion order (LN + gate + projection for quickwin; concat + Linear for minimal)
  – AFEM/sem_refine modules (TinyCLIP) exactly as implemented
  – SupCon head (UN-normalized output; L2 inside SupConLoss)
  – Option-B returns (global + 4 locals) and forward signatures unchanged

CLI quick examples:
  # 1) Baseline (no CLIP)
  --opts MODEL.USE_CLIP False

  # 2) OpenCLIP minimal (global only)
  --opts MODEL.USE_CLIP True MODEL.CLIP_IMPL open_clip MODEL.CLIP_FUSION minimal \
        MODEL.CLIP_BACKBONE ViT-B-16 MODEL.CLIP_PRETRAIN laion2b_s34b_b88k

  # 3) OpenCLIP quick-win (global + 4×locals with LN+gate+proj)
  --opts MODEL.USE_CLIP True MODEL.CLIP_IMPL open_clip MODEL.CLIP_FUSION quickwin \
        MODEL.CLIP_BACKBONE ViT-B-16 MODEL.CLIP_PRETRAIN laion2b_s34b_b88k

  # 4) TinyCLIP (HF) global-only + AFEM + sem_refine
  --opts MODEL.USE_CLIP True MODEL.CLIP_IMPL hf MODEL.CLIP_HF_ID wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M \
        MODEL.CLIP_INPUT_SIZE '(224,224)' MODEL.CLIP_FINETUNE True \
        MODEL.CLIP_USE_AFEM True MODEL.CLIP_USE_SEM_REFINE True
"""

# =============================================================================
# File: make_model.py
# Purpose: Build backbone/transformer/transformer_local models with optional
#          SupCon head and CLI-switchable feature sources for TripletX/SupCon.
#
# Default layering (unchanged if no CLI overrides are given):
#   - CE / Margin-Softmax head: BNNeck feature (feat_bn)
#   - Triplet / TripletX:      pre-BN global feature (global_feat)
#   - SupCon:                  BNNeck feature (feat_bn)
#
# Training-time returns (backbone/transformer):
#   - always return 4 items: (cls_score, global_feat, feat_bn, z_supcon)
#     (z_supcon=None if SupCon disabled)
#
# Training-time returns (transformer_local):
#   - always return 4 items: (scores, feats, feat_bn_global_or_list, z_supcon)
#
# Test-time behavior is unchanged (controlled by cfg.TEST.NECK_FEAT).
#
# Change Log
# [2025-09-12 | Zhang Hang & Meng Fanyi] Added SupCon projection head (->128) to Backbone,
#                                       build_transformer, build_transformer_local; training
#                                       forwards return extra L2-normalized z_supcon when enabled.
# [2025-09-15 | Zhang Hang]            Added load_param() to build_transformer_local for
#                                       evaluation/test checkpoint loading.
# [2025-10-15 | Zhang Hang]            Added CLI-switchable feature source for TripletX/SupCon
#                                       (pre_bn vs bnneck) while keeping default layering intact.
# [2025-10-17 | Hang Zhang]            **Stability Fix**
#                                       - Unified forward() return signatures (always 4 items).
#                                       - Added default z_supcon=None when disabled.
# [2025-10-17 | Hang Zhang]            **Compatibility Fix**
#                                       - Added full classifier initialization for Arcface/Cosface/
#                                         AMSoftmax/CircleLoss in Backbone to prevent AttributeError
#                                         when COS_LAYER=True.
# [2025-10-19 | Hang Zhang]            **Normalize-at-loss policy (SupCon)**
#                                       - Removed F.normalize on z_supcon in all models.
#                                       - SupCon head now outputs raw (unnormalized) embeddings.
#                                       - L2-normalization is centralized in SupConLoss.forward().
#                                       - Added one-time init log when SupCon head is enabled.
# [2025-10-19 | Hang Zhang]            **Model-side SupCon source logging (once per model)**
#                                       - On first forward that builds z_supcon, print:
#                                         "[make_model][combo] [SupCon] src=model::<pre_bn|bnneck> | head=2xLinear(->128) | output=UN-normalized (L2 in SupConLoss)"
#                                       - No behavior change; only logging for clarity.
# [2025-10-19 | Hang Zhang]            **Classification head label passing fix**
#                                       - For margin-softmax heads (arcface/cosface/amsoftmax/circle),
#                                         always pass `label` to classifier; Linear head does not.
# [2025-10-21 | Hang Zhang]            Added self.bottleneck_3.bias.requires_grad_(False),
#                                       self.bottleneck_3.apply(weights_init_kaiming)
# [2025-10-22 | Hang Zhang]            **Option-B (BNNeck parity)**
#                                       - For build_transformer_local (JPM=True), training forward now returns
#                                         feat_bn as a list: [feat_bn_global, local_bn1, local_bn2, local_bn3, local_bn4].
#                                       - Backbone/build_transformer keep a single BNNeck tensor for feat_bn.
#                                       - Enables Triplet/TripletX to aggregate "global + 4×local BNNeck"
#                                         mirroring the pre_bn path (structure parity).
# [2025-10-23 | Hang Zhang]            **CLIP fusion (minimal, JPM=True)**
#                                       - Added optional frozen CLIP visual encoder to build_transformer_local.
#                                       - Fused CLIP only into GLOBAL pre-BN (concat+Linear); 4 local branches unchanged.
#                                       - Controlled by MODEL.USE_CLIP, with MODEL.CLIP_BACKBONE / CLIP_PRETRAIN (optional).
#                                       - Disabled -> exact parity with previous behavior (bitwise-compatible paths).
#                                       - Verified run: veri776_20251023-0731_ce_triplet_clip_seed0_deit_run
# [2025-10-24 | Hang Zhang]            **CLIP fusion quick-win (global + 4×local)**
#                                       - Add LayerNorm-based alignment and scalar gating (alpha) for global fusion.
#                                       - Connect CLIP to 4 local branches via a shared lightweight fusion head.
#                                       - Keep forward signatures and loss usage unchanged; SupCon optional.
#                                       - Backward-compatible: when MODEL.USE_CLIP=False, behavior is identical.
#                                       - Verified run: veri776_20251023-1618_ce_triplet_clip_seed0_deit_run
# [2025-10-24 | Hang Zhang]            **TinyCLIP (HF) global-only fusion**
#                                       - Integrated HuggingFace TinyCLIP-ViT as optional CLIP_IMPL='hf'.
#                                       - Added AFEM & sem_refine modules; position embedding resize via bicubic.
#                                       - Controlled by MODEL.CLIP_HF_ID / MODEL.CLIP_INPUT_SIZE / MODEL.CLIP_USE_AFEM / MODEL.CLIP_USE_SEM_REFINE.
#                                       - Verified run: veri776_20251024-0912_ce_triplet_hfclip_seed0_deit_run
# [2025-11-04 | Hang Zhang]            **Merged CLIP switch framework**
#                                       - Unified prior CLIP variants (minimal, quick-win, TinyCLIP) under config switches:
#                                         MODEL.USE_CLIP, MODEL.CLIP_IMPL, MODEL.CLIP_FUSION.
#                                       - No behavior change for existing runs; backward-compatible reproductions confirmed.
#                                       - Merge source: [2025-10-23 minimal] + [2025-10-24 quickwin/hf], both previously run.
#                                       - Purpose: enable runtime selection without code duplication.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# (backbones / vit factory)
from .backbones.resnet import ResNet, Bottleneck
from .backbones.vit_pytorch import (
    vit_base_patch16_224_TransReID,
    vit_small_patch16_224_TransReID,
    deit_small_patch16_224_TransReID
)
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def shuffle_unit(features, shift, group, begin=1):
    """Token shift + patch shuffle for JPM path."""
    batchsize = features.size(0)
    dim = features.size(-1)
    feature_random = torch.cat(
        [features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1
    )
    x = feature_random
    try:
        x = x.view(batchsize, group, -1, dim)
    except Exception:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)
    return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias, 0.0)


def _pick_feat_by_src(src: str, global_feat: torch.Tensor, bn_feat: torch.Tensor) -> torch.Tensor:
    """Select feature by name: 'pre_bn' | 'bnneck' (default to pre_bn)."""
    return bn_feat if src == "bnneck" else global_feat


# -----------------------------------------------------------------------------
# HF TinyCLIP helpers (vision tower only)
# -----------------------------------------------------------------------------
class SafeTinyClipEmbeddings(nn.Module):
    """
    Patch for HF TinyCLIP to avoid calling position_embedding(Parameter) as a module.
    - Keep original conv patch embedding + class token.
    - Add positional parameter by broadcast.
    - No extra interpolate here (we resize pos_embed outside once).
    """
    def __init__(self, orig_emb_module):
        super().__init__()
        self.patch_embedding = orig_emb_module.patch_embedding
        self.class_embedding = orig_emb_module.class_embedding
        self.position_embedding = orig_emb_module.position_embedding

    def forward(self, pixel_values, interpolate_pos_encoding: bool = False):
        x = self.patch_embedding(pixel_values)        # (B, C, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)              # (B, Gh*Gw, C)
        cls = self.class_embedding
        if cls.dim() == 1:
            cls = cls.unsqueeze(0).unsqueeze(0)
        cls = cls.expand(x.size(0), -1, -1)           # (B,1,C)
        x = torch.cat([cls, x], dim=1)                # (B, 1+Gh*Gw, C)

        pe = self.position_embedding
        if pe.dim() == 2:
            pe = pe.unsqueeze(0)                      # (1, L, C)
        x = x + pe
        return x


def _hf_get_pos_embed(clip_vision):
    """Return (pos_embed [1,N,C], patch_size:int) in a HF/TinyCLIP-safe way."""
    pe_obj = clip_vision.embeddings.position_embedding
    if hasattr(pe_obj, "weight"):
        pe = pe_obj.weight
    else:
        pe = pe_obj
    if pe.dim() == 2:
        pe = pe.unsqueeze(0)
    patch = int(getattr(clip_vision.config, "patch_size"))
    return pe, patch


@torch.no_grad()
def _hf_resize_pos_embed(clip_vision, new_hw):
    """
    One-time resize for TinyCLIP position embeddings to arbitrary input size.
    - Split CLS + grid, bicubic on grid, concat back.
    - Replace embeddings.position_embedding with resized nn.Parameter([N,C]).
    """
    pe, patch = _hf_get_pos_embed(clip_vision)  # [1,N,C]
    _, N, C = pe.shape
    cls_tok = pe[:, :1, :]
    grid = pe[:, 1:, :]
    gh = gw = int((N - 1) ** 0.5)
    if gh * gw != (N - 1):
        raise ValueError(f"Non-square grid: N-1={N-1}")
    grid = grid.reshape(1, gh, gw, C).permute(0, 3, 1, 2)  # [1,C,gh,gw]

    new_h, new_w = new_hw
    new_gh = new_h // patch
    new_gw = new_w // patch
    grid = F.interpolate(grid, size=(new_gh, new_gw), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_gh * new_gw, C)
    pe_new = torch.cat([cls_tok, grid], dim=1)  # [1, 1+new_gh*new_gw, C]

    pe_new_param = nn.Parameter(pe_new.squeeze(0))  # [N,C]
    clip_vision.embeddings.position_embedding = pe_new_param
    print(f"[make_model][_hf_resize_pos_embed] {gh}x{gw} -> {new_gh}x{new_gw} tokens "
          f"({N-1} -> {1 + new_gh*new_gw})")


# -----------------------------------------------------------------------------
# AFEM (for TinyCLIP semantics refinement)
# -----------------------------------------------------------------------------
class AFEM(nn.Module):
    """
    Adaptive Fine-grained Enhancement Module:
      T' = T + W_g * f_theta(T)
    where W_g is learnable per-group weight.
    """
    def __init__(self, dim: int, groups: int = 32):
        super().__init__()
        assert dim % groups == 0, "AFEM: dim must be divisible by groups"
        self.dim = dim
        self.groups = groups
        self.group_dim = dim // groups
        self.linear = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
        )
        self.group_weight = nn.Parameter(torch.randn(groups))
        nn.init.normal_(self.group_weight, mean=0.0, std=1.0)

    def forward(self, ts: torch.Tensor) -> torch.Tensor:
        base = self.linear(ts)                      # [B,D]
        B, D = base.shape
        g = base.view(B, self.groups, self.group_dim)
        w = self.group_weight.view(1, self.groups, 1)
        weighted = (w * g).view(B, D)
        return ts + weighted


# -----------------------------------------------------------------------------
# ResNet Backbone
# -----------------------------------------------------------------------------
class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super().__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride, block=Bottleneck, layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            raise ValueError(f'unsupported backbone! got {model_name}')

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print(f'Loading pretrained ImageNet model......from {model_path}')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        # ID head
        self.ID_LOSS_TYPE = getattr(cfg.MODEL, "ID_LOSS_TYPE", "softmax")
        if self.ID_LOSS_TYPE == 'arcface':
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # SupCon
        self.supcon_enabled = (
            hasattr(cfg, "LOSS")
            and hasattr(cfg.LOSS, "SUPCON")
            and getattr(cfg.LOSS.SUPCON, "ENABLE", False)
        )
        if self.supcon_enabled:
            self.supcon_feat_src = getattr(cfg.LOSS.SUPCON, "FEAT_SRC", "bnneck")
            self.supcon_head = nn.Sequential(
                nn.Linear(self.in_planes, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 128)
            )
            self._supcon_log_done = False
            print("[make_model][init] SupCon head enabled (Backbone): output UN-normalized; L2 in SupConLoss.")

    def forward(self, x, label=None):
        x = self.base(x)
        global_feat = self.gap(x).view(x.size(0), -1)
        feat = global_feat if self.neck == 'no' else self.bottleneck(global_feat)

        z_supcon = None
        if self.training and self.supcon_enabled:
            sup_in = _pick_feat_by_src(self.supcon_feat_src, global_feat, feat)
            if not self._supcon_log_done:
                print(f"[make_model][combo] [SupCon] src=model::{self.supcon_feat_src} | head=2xLinear(->128) | "
                      f"output=UN-normalized (L2 in SupConLoss)")
                self._supcon_log_done = True
            z_supcon = self.supcon_head(sup_in)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat, feat, z_supcon
        else:
            return feat if self.neck_feat == 'after' else global_feat


# -----------------------------------------------------------------------------
# ViT/DeiT Transformer (global only)
# -----------------------------------------------------------------------------
class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super().__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print(f'using Transformer_type: {cfg.MODEL.TRANSFORMER_TYPE} as a backbone')

        if not cfg.MODEL.SIE_CAMERA: camera_num = 0
        if not cfg.MODEL.SIE_VIEW:   view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN,
            sie_xishu=cfg.MODEL.SIE_COE,
            camera=camera_num,
            view=view_num,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE
        )
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print(f'Loading pretrained ImageNet model......from {model_path}')

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                         s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # SupCon
        self.supcon_enabled = (
            hasattr(cfg, "LOSS") and hasattr(cfg.LOSS, "SUPCON")
            and getattr(cfg.LOSS.SUPCON, "ENABLE", False)
        )
        if self.supcon_enabled:
            self.supcon_feat_src = getattr(cfg.LOSS.SUPCON, "FEAT_SRC", "bnneck")
            self.supcon_head = nn.Sequential(
                nn.Linear(self.in_planes, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 128)
            )
            self._supcon_log_done = False
            print("[make_model][init] SupCon head enabled (Transformer): output UN-normalized; L2 in SupConLoss.")

    def forward(self, x, label=None, cam_label=None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)
        feat = self.bottleneck(global_feat)

        z_supcon = None
        if self.training and self.supcon_enabled:
            sup_in = _pick_feat_by_src(self.supcon_feat_src, global_feat, feat)
            if not self._supcon_log_done:
                print(f"[make_model][combo] [SupCon] src=model::{self.supcon_feat_src} | head=2xLinear(->128) | "
                      f"output=UN-normalized (L2 in SupConLoss)")
                self._supcon_log_done = True
            z_supcon = self.supcon_head(sup_in)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat, feat, z_supcon
        else:
            return feat if self.neck_feat == 'after' else global_feat


# -----------------------------------------------------------------------------
# ViT/DeiT Transformer with JPM locals + CLIP fusion (switchable)
# -----------------------------------------------------------------------------
class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super().__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.rearrange = rearrange

        print(f'using Transformer_type: {cfg.MODEL.TRANSFORMER_TYPE} as a backbone')

        if not cfg.MODEL.SIE_CAMERA: camera_num = 0
        if not cfg.MODEL.SIE_VIEW:   view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN,
            sie_xishu=cfg.MODEL.SIE_COE,
            local_feature=cfg.MODEL.JPM,
            camera=camera_num,
            view=view_num,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            drop_path_rate=cfg.MODEL.DROP_PATH
        )

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print(f'Loading pretrained ImageNet model......from {model_path}')

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))
        self.b2 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))

        # ---------------- Classifiers ----------------
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
            # Global head may be margin-softmax; locals use linear heads.
            if self.ID_LOSS_TYPE == 'arcface':
                self.classifier = Arcface(self.in_planes, self.num_classes,
                                          s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            elif self.ID_LOSS_TYPE == 'cosface':
                self.classifier = Cosface(self.in_planes, self.num_classes,
                                          s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            elif self.ID_LOSS_TYPE == 'amsoftmax':
                self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                            s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            else:
                self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                             s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            for h in [self.classifier_1, self.classifier_2, self.classifier_3, self.classifier_4]:
                h.apply(weights_init_classifier)
        else:
            # All linear heads (global + locals)
            self.classifier   = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            for h in [self.classifier, self.classifier_1, self.classifier_2, self.classifier_3, self.classifier_4]:
                h.apply(weights_init_classifier)

        # ---------------- BNNecks ----------------
        self.bottleneck   = nn.BatchNorm1d(self.in_planes); self.bottleneck.bias.requires_grad_(False);   self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes); self.bottleneck_1.bias.requires_grad_(False); self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes); self.bottleneck_2.bias.requires_grad_(False); self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes); self.bottleneck_3.bias.requires_grad_(False); self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes); self.bottleneck_4.bias.requires_grad_(False); self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        self.shift_num = cfg.MODEL.SHIFT_NUM
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH

        # ---------------- Common mean/std buffers (ImageNet) ----------------
        im_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        im_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imnet_mean", im_mean, persistent=False)
        self.register_buffer("imnet_std",  im_std,  persistent=False)

        # ---------------- CLIP switches ----------------
        self.use_clip   = bool(getattr(cfg.MODEL, "USE_CLIP", False))
        self.clip_impl  = str(getattr(cfg.MODEL, "CLIP_IMPL", "open_clip")).lower()
        self.clip_fuse  = str(getattr(cfg.MODEL, "CLIP_FUSION", "minimal")).lower()

        # ---------------- OPEN_CLIP path ----------------
        if self.use_clip and self.clip_impl == "open_clip":
            try:
                import open_clip
            except Exception as e:
                raise ImportError("MODEL.CLIP_IMPL='open_clip' requires `open_clip` package.") from e

            clip_backbone = getattr(cfg.MODEL, "CLIP_BACKBONE", "ViT-B-16")
            clip_pretrain = getattr(cfg.MODEL, "CLIP_PRETRAIN", "laion2b_s34b_b88k")

            self.clip_model = open_clip.create_model(clip_backbone, pretrained=clip_pretrain)
            for p in self.clip_model.parameters():
                p.requires_grad = False
            self.clip_model.eval()

            size = getattr(self.clip_model.visual, "image_size", 224)
            if isinstance(size, int):
                size = (size, size)
            self.clip_input_size = tuple(size)  # e.g. (224,224)

            # CLIP mean/std (OpenCLIP defaults)
            clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
            self.register_buffer("clip_mean", clip_mean, persistent=False)
            self.register_buffer("clip_std",  clip_std,  persistent=False)

            if self.clip_fuse == "minimal":
                # Minimal: global-only concat+Linear; locals unchanged
                self.fuse_proj_global_min = nn.Linear(
                    self.in_planes + self.clip_model.visual.output_dim, self.in_planes
                )
                print(f"[make_model][init] CLIP(open_clip/minimal) global-only; expected_size={self.clip_input_size}")
            elif self.clip_fuse == "quickwin":
                # Quick-win: global + 4×locals with LN + gate + projection
                self.fuse_ln_global_backbone = nn.LayerNorm(self.in_planes)
                self.fuse_ln_global_clip     = nn.LayerNorm(self.clip_model.visual.output_dim)
                self.fuse_gate_global = nn.Sequential(
                    nn.Linear(self.in_planes + self.clip_model.visual.output_dim, self.in_planes // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.in_planes // 4, 1),
                    nn.Sigmoid()
                )
                self.fuse_proj_global = nn.Linear(
                    self.in_planes + self.clip_model.visual.output_dim, self.in_planes
                )
                # Shared local head
                self.fuse_ln_local_bb = nn.LayerNorm(self.in_planes)
                self.fuse_ln_local_cf = nn.LayerNorm(self.clip_model.visual.output_dim)
                self.fuse_gate_local = nn.Sequential(
                    nn.Linear(self.in_planes + self.clip_model.visual.output_dim, self.in_planes // 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.in_planes // 4, 1),
                    nn.Sigmoid()
                )
                self.fuse_proj_local = nn.Linear(
                    self.in_planes + self.clip_model.visual.output_dim, self.in_planes
                )
                print(f"[make_model][init] CLIP(open_clip/quickwin) global+4×local with LN+gate; expected_size={self.clip_input_size}")
            else:
                raise ValueError(f"Unknown MODEL.CLIP_FUSION='{self.clip_fuse}' (expected 'minimal' or 'quickwin').")

        # ---------------- HF TinyCLIP path ----------------
        if self.use_clip and self.clip_impl == "hf":
            try:
                from transformers import CLIPModel, CLIPImageProcessor
            except Exception as e:
                raise ImportError("MODEL.CLIP_IMPL='hf' requires `transformers` package.") from e

            hf_id     = getattr(cfg.MODEL, "CLIP_HF_ID", "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M")
            local_dir = getattr(cfg.MODEL, "CLIP_LOCAL_PATH", "").strip()
            model_id_or_path = local_dir if local_dir else hf_id
            print(f"[make_model][init] Loading TinyCLIP (HF) from: {model_id_or_path}")

            self.hf_processor = CLIPImageProcessor.from_pretrained(model_id_or_path, use_fast=True)
            hf_clip = CLIPModel.from_pretrained(model_id_or_path)

            self.clip_model = hf_clip.vision_model  # vision only
            self.clip_model.embeddings = SafeTinyClipEmbeddings(self.clip_model.embeddings)
            self.clip_model.config.interpolate_pos_encoding = True

            # Input size (default from processor, overridable by cfg.MODEL.CLIP_INPUT_SIZE)
            default_h = self.hf_processor.size.get("shortest_edge", None)
            if default_h is None:
                default_h = self.hf_processor.size.get("height", 224)
            self.clip_input_size = tuple(getattr(cfg.MODEL, "CLIP_INPUT_SIZE", (default_h, default_h)))
            self.clip_target_size = (int(self.clip_input_size[0]), int(self.clip_input_size[1]))

            # HF mean/std
            clip_mean = torch.tensor(self.hf_processor.image_mean).view(1, 3, 1, 1)
            clip_std  = torch.tensor(self.hf_processor.image_std).view(1, 3, 1, 1)
            self.register_buffer("clip_mean", clip_mean, persistent=False)
            self.register_buffer("clip_std",  clip_std,  persistent=False)

            # Finetune control
            self.clip_finetune = bool(getattr(cfg.MODEL, "CLIP_FINETUNE", True))
            for p in self.clip_model.parameters():
                p.requires_grad = self.clip_finetune

            self.clip_output_dim = int(hf_clip.config.vision_config.hidden_size)

            # Fusion: global-only (Tu) + optional AFEM refinement (Ts')
            self.fuse_fc = nn.Linear(self.in_planes + self.clip_output_dim, self.in_planes)
            self.afem = AFEM(dim=self.clip_output_dim, groups=32)
            self.sem_refine_proj = nn.Linear(self.clip_output_dim, self.in_planes)

            # One-time PE resize flag
            self._clip_pos_resized = False

            # Ablation switches
            self.clip_use_afem = bool(getattr(cfg.MODEL, "CLIP_USE_AFEM", True))
            self.clip_use_sem_refine = bool(getattr(cfg.MODEL, "CLIP_USE_SEM_REFINE", True))
            mode_str = ("CLIP fusion mode: AFEM + sem_refine_proj (full)" if self.clip_use_sem_refine and self.clip_use_afem
                        else "CLIP fusion mode: sem_refine_proj only (no AFEM)" if self.clip_use_sem_refine
                        else "CLIP fusion mode: fuse_fc only (no refine branch)")
            print("[make_model][clip][hf] " + mode_str +
                  f" | input={self.clip_target_size}, proj_dim={self.clip_output_dim}, finetune={self.clip_finetune}")

        # ---------------- SupCon head ----------------
        self.supcon_enabled = (
            hasattr(cfg, "LOSS")
            and hasattr(cfg.LOSS, "SUPCON")
            and getattr(cfg.LOSS.SUPCON, "ENABLE", False)
        )
        if self.supcon_enabled:
            self.supcon_feat_src = getattr(cfg.LOSS.SUPCON, "FEAT_SRC", "bnneck")
            self.supcon_head = nn.Sequential(
                nn.Linear(self.in_planes, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 128)
            )
            self._supcon_log_done = False
            print("[make_model][init] SupCon head enabled (TransformerLocal): output UN-normalized; L2 in SupConLoss.")

    def _encode_openclip(self, x_img):
        # ImageNet denorm -> CLIP norm -> bicubic resize -> encode
        x_denorm = x_img * self.imnet_std + self.imnet_mean
        x_clip = (x_denorm - self.clip_mean) / self.clip_std
        h, w = self.clip_input_size
        if x_clip.shape[-2:] != (h, w):
            x_clip = F.interpolate(x_clip, size=(h, w), mode='bicubic', align_corners=False)
        with torch.no_grad():
            feat_clip = self.clip_model.encode_image(x_clip)  # [B, D_clip]
        return feat_clip

    def _fuse_quickwin_global(self, global_feat, feat_clip):
        bb_g  = self.fuse_ln_global_backbone(global_feat)
        cf_g  = self.fuse_ln_global_clip(feat_clip)
        cat_g = torch.cat([bb_g, cf_g], dim=1)
        a_g   = self.fuse_gate_global(cat_g)           # [B,1]
        fused_in_g = torch.cat([(1.0 - a_g) * bb_g, a_g * cf_g], dim=1)
        return self.fuse_proj_global(fused_in_g)

    def _fuse_quickwin_local(self, local_feat, feat_clip):
        bb  = self.fuse_ln_local_bb(local_feat)
        cf  = self.fuse_ln_local_cf(feat_clip)
        cat = torch.cat([bb, cf], dim=1)
        a   = self.fuse_gate_local(cat)                # [B,1]
        fused_in = torch.cat([(1.0 - a) * bb, a * cf], dim=1)
        return self.fuse_proj_local(fused_in)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        # Keep raw input for CLIP vision preprocessing
        x_img = x

        # Backbone with JPM tokens
        features = self.base(x, cam_label=cam_label, view_label=view_label)
        b1_feat = self.b1(features)
        global_feat = b1_feat[:, 0]  # CLS token

        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x_tokens = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x_tokens = features[:, 1:]

        def extract_local(start, end):
            lf = x_tokens[:, start:end]
            lf = self.b2(torch.cat((token, lf), dim=1))
            return lf[:, 0]

        local_feat_1 = extract_local(0, patch_length)
        local_feat_2 = extract_local(patch_length, patch_length * 2)
        local_feat_3 = extract_local(patch_length * 2, patch_length * 3)
        local_feat_4 = extract_local(patch_length * 3, patch_length * 4)

        # ---------------- CLIP fusion (if enabled) ----------------
        if self.use_clip and self.clip_impl == "open_clip":
            feat_clip = self._encode_openclip(x_img)

            if self.clip_fuse == "minimal":
                # Minimal: global-only concat+Linear; locals unchanged
                pre_bn_global = self.fuse_proj_global_min(torch.cat([global_feat, feat_clip], dim=1))
            else:
                # Quick-win: global + 4×locals with LN+gate+proj
                pre_bn_global = self._fuse_quickwin_global(global_feat, feat_clip)
                local_feat_1  = self._fuse_quickwin_local(local_feat_1, feat_clip)
                local_feat_2  = self._fuse_quickwin_local(local_feat_2, feat_clip)
                local_feat_3  = self._fuse_quickwin_local(local_feat_3, feat_clip)
                local_feat_4  = self._fuse_quickwin_local(local_feat_4, feat_clip)

        elif self.use_clip and self.clip_impl == "hf":
            # HF TinyCLIP: global-only (Tu + Ts'), locals unchanged
            x_unnorm = x_img * self.imnet_std + self.imnet_mean
            x_clip = (x_unnorm - self.clip_mean) / self.clip_std
            th, tw = self.clip_input_size
            if x_clip.shape[-2:] != (th, tw):
                x_clip = F.interpolate(x_clip, size=(th, tw), mode='bilinear', align_corners=False)

            if not getattr(self, "_clip_pos_resized", False):
                _hf_resize_pos_embed(self.clip_model, (th, tw))
                if hasattr(self.clip_model, "config") and hasattr(self.clip_model.config, "image_size"):
                    self.clip_model.config.image_size = th
                if hasattr(self.clip_model, "embeddings") and hasattr(self.clip_model.embeddings, "image_size"):
                    self.clip_model.embeddings.image_size = th
                self._clip_pos_resized = True
                print(f"[make_model][clip][hf] resized TinyCLIP pos_embed to {th}x{tw} (once)")

            out = self.clip_model(pixel_values=x_clip)

            def _clip_pool(outputs):
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    return outputs.pooler_output
                if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    xx = outputs.last_hidden_state
                    return xx[:, 0] if xx.dim() == 3 else xx
                if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                    xx = outputs[0]
                    return xx[:, 0] if (torch.is_tensor(xx) and xx.dim() == 3) else xx
                if torch.is_tensor(outputs):
                    return outputs[:, 0] if outputs.dim() == 3 else outputs
                raise RuntimeError("Unsupported TinyCLIP outputs structure.")

            ts_raw = _clip_pool(out)  # [B, D_clip]
            Tu = self.fuse_fc(torch.cat([global_feat, ts_raw], dim=1))  # [B,C]
            if self.clip_use_sem_refine:
                ts_ref = self.afem(ts_raw) if self.clip_use_afem else ts_raw
                Ts_prime = self.sem_refine_proj(ts_ref)
            else:
                Ts_prime = torch.zeros_like(Tu)
            pre_bn_global = Tu + Ts_prime

        else:
            # No CLIP: keep original global pre-BN
            pre_bn_global = global_feat

        # ---------------- BN necks ----------------
        feat_bn_global = self.bottleneck(pre_bn_global)
        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        # ---------------- SupCon ----------------
        z_supcon = None
        if self.training and self.supcon_enabled:
            sup_in = feat_bn_global if self.supcon_feat_src == "bnneck" else pre_bn_global
            if not self._supcon_log_done:
                print(f"[make_model][combo] [SupCon] src=model::{self.supcon_feat_src} | head=2xLinear(->128) | "
                      f"output=UN-normalized (L2 in SupConLoss)")
                self._supcon_log_done = True
            z_supcon = self.supcon_head(sup_in)

        # ---------------- Heads & returns ----------------
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_bn_global, label)
            else:
                cls_score = self.classifier(feat_bn_global)
            scores = [
                cls_score,
                self.classifier_1(local_feat_1_bn),
                self.classifier_2(local_feat_2_bn),
                self.classifier_3(local_feat_3_bn),
                self.classifier_4(local_feat_4_bn)
            ]
            feats = [pre_bn_global, local_feat_1, local_feat_2, local_feat_3, local_feat_4]
            feats_bn = [feat_bn_global, local_feat_1_bn, local_feat_2_bn, local_feat_3_bn, local_feat_4_bn]
            return scores, feats, feats_bn, z_supcon
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat_bn_global,
                     local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4],
                    dim=1
                )
            else:
                return torch.cat(
                    [pre_bn_global,
                     local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4],
                    dim=1
                )

    def load_param(self, trained_path: str, strict: bool = False):
        import os
        assert os.path.isfile(trained_path), f"Weight file not found: {trained_path}"
        state = torch.load(trained_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        clean = {k.replace("module.", ""): v for k, v in state.items()}
        incompatible = self.load_state_dict(clean, strict=strict)
        missing = getattr(incompatible, "missing_keys", [])
        unexpected = getattr(incompatible, "unexpected_keys", [])
        print(f"[build_transformer_local.load_param] loaded '{trained_path}' | "
              f"missing={len(missing)} unexpected={len(unexpected)} strict={strict}")
        if missing:
            print("  missing:", missing[:12], "..." if len(missing) > 12 else "")
        if unexpected:
            print("  unexpected:", unexpected[:12], "..." if len(unexpected) > 12 else "")


# -----------------------------------------------------------------------------
# Factory & API
# -----------------------------------------------------------------------------
__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}


def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(
                num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE
            )
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
