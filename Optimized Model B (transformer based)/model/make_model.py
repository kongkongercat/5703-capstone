# model/make_model.py
# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .backbones.resnet import ResNet, Bottleneck
from .backbones.vit_pytorch import (
    vit_base_patch16_224_TransReID,
    vit_small_patch16_224_TransReID,
    deit_small_patch16_224_TransReID
)
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

# HF TinyCLIP (vision + projection) for image embeddings
from transformers import CLIPModel, CLIPImageProcessor  # HF-only path


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
# To switch by CLI:
#   --opts LOSS.TRIPLETX.FEAT_SRC bnneck|pre_bn LOSS.SUPCON.FEAT_SRC bnneck|pre_bn
#
# Training-time returns (backbone/transformer):
#   - always return 4 items: (cls_score, global_feat, feat_bn, z_supcon)
#     (z_supcon=None if SupCon disabled)
#
# Training-time returns (transformer_local):
#   - always return 4 items: (scores, feats, feat_bn_list_or_tensor, z_supcon)
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
# [2025-10-19 | Hang Zhang]           **Model-side SupCon source logging (once per model)**
#                                       - On first forward that builds z_supcon, print:
#                                         "[make_model][combo] [SupCon] src=model::<pre_bn|bnneck> | head=2xLinear(->128) | output=UN-normalized (L2 in SupConLoss)"
# [2025-10-19 | Hang Zhang]            **Classification head label passing fix**
#                                       - For margin-softmax heads (arcface/cosface/amsoftmax/circle),
#                                         always pass `label` to classifier; Linear head does not.
# [2025-10-21 | Hang Zhang]      add self.bottleneck_3.bias.requires_grad_(False), self.bottleneck_3.apply(weights_init_kaiming)
# [2025-10-22 | Hang Zhang]      **Option-B (BNNeck parity)**
#                                 - For build_transformer_local (JPM=True), training forward now returns
#                                   feat_bn as a list: [feat_bn_global, local_bn1, local_bn2, local_bn3, local_bn4].
#                                 - Backbone/build_transformer keep a single BNNeck tensor for feat_bn.
#                                 - Enables Triplet/TripletX to aggregate "global + 4×local BNNeck"
#                                   mirroring the pre_bn path (structure parity).
# [2025-10-24 | Hang Zhang]      **TinyCLIP + AFEM (HF-only)**
#                                 - Switched to HuggingFace CLIPModel (vision + projection).
#                                 - Fuses `image_embeds` (projection_dim=768) with global Transformer feat.
#                                 - Position embedding resize for arbitrary CLIP_INPUT_SIZE.
#                                 - AFEM module for semantic refinement prior to projection.
#                                 - Optional ε-level fine-tuning toggled by cfg.MODEL.CLIP_FINETUNE.
# =============================================================================


# ----------------------------- Utilities -------------------------------------
def shuffle_unit(features, shift, group, begin=1):
    """Token shift + patch shuffle for JPM path."""
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift
    feature_random = torch.cat(
        [features[:, begin - 1 + shift:], features[:, begin:begin - 1 + shift]], dim=1
    )
    x = feature_random
    # Patch shuffle
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
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def _pick_feat_by_src(src: str, global_feat: torch.Tensor, bn_feat: torch.Tensor) -> torch.Tensor:
    """Choose feature by name: 'pre_bn' or 'bnneck' (default to pre_bn)."""
    return bn_feat if src == "bnneck" else global_feat


# --------- HF TinyCLIP positional embedding resize (Vision tower only) -------
def _hf_get_pos_embed(clip_vision):
    """
    Returns (pos_embed[1,N,C], patch_size:int) for HF CLIP vision tower.
    """
    pe = clip_vision.embeddings.position_embedding.weight  # [1,N,C] or [N,C]
    if pe.dim() == 2:
        pe = pe.unsqueeze(0)
    patch = int(clip_vision.config.patch_size)
    return pe, patch


@torch.no_grad()
def _hf_resize_pos_embed(clip_vision, new_hw):
    """
    Resize TinyCLIP / HF CLIP vision tower positional embeddings
    so it can run at a new spatial size (e.g. 256x256 or 320x320).

    Steps:
    1. Take the original position_embedding (CLS + grid tokens).
    2. Bicubic-interpolate the grid part from old (gh x gw) to new (new_gh x new_gw).
    3. Concat CLS token back.
    4. Replace clip_vision.embeddings.position_embedding with a NEW nn.Parameter
       that has the correct new length.
    """
    # Get original positional embedding as [1, N, C]
    pe, patch = _hf_get_pos_embed(clip_vision)  # pe: [1, N, C]
    _, N, C = pe.shape

    # Split CLS token (index 0) and patch grid tokens (1:)
    cls_tok = pe[:, :1, :]      # [1,1,C]
    grid    = pe[:, 1:, :]      # [1,(N-1),C]

    # Figure out original grid size (should be square for TinyCLIP pretrain)
    gh = gw = int((N - 1) ** 0.5)
    assert gh * gw == (N - 1), f"Non-square grid: N-1={N-1}"

    # Reshape grid tokens to [1, C, gh, gw] so we can spatially interpolate
    grid = grid.reshape(1, gh, gw, C).permute(0, 3, 1, 2)  # -> [1, C, gh, gw]

    # Compute target grid size from desired new_hw and patch size
    new_h, new_w = new_hw                # e.g. (320,320)
    new_gh = new_h // patch             # e.g. 320//32 = 10
    new_gw = new_w // patch             #      320//32 = 10

    # Bicubic interpolate positional grid to new grid resolution
    grid = F.interpolate(
        grid,
        size=(new_gh, new_gw),
        mode="bicubic",
        align_corners=False
    )  # -> [1, C, new_gh, new_gw]

    # Flatten back to tokens: [1, new_gh*new_gw, C]
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_gh * new_gw, C)

    # Concat CLS token back in front → [1, 1+new_gh*new_gw, C]
    pe_new = torch.cat([cls_tok, grid], dim=1)  # [1, new_N, C]

    # IMPORTANT:
    # Replace clip_vision.embeddings.position_embedding with a fresh nn.Parameter
    # whose length matches the new patch grid. We CANNOT just .copy_() into the old
    # param because token counts don't match (e.g. 50 -> 101).
    clip_vision.embeddings.position_embedding = nn.Parameter(
        pe_new.squeeze(0)  # HF CLIP vision stores it as [N, C] (no batch dim)
    )




# --------------------------- AFEM (TinyCLIP semantics) -----------------------
class AFEM(nn.Module):
    """
    Adaptive Fine-grained Enhancement Module for CLIP semantics.
    Input : [B, D_sem]  TinyCLIP semantic vector
    Output: [B, D_sem]  refined semantics T_s'
    """
    def __init__(self, dim: int, groups: int = 32):
        super().__init__()
        assert dim % groups == 0, "AFEM: 'dim' must be divisible by 'groups'"
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
        base = self.linear(ts)                           # f_linear(T_s)
        B, D = base.shape
        g = base.view(B, self.groups, self.group_dim)    # [B,G,d]
        w = self.group_weight.view(1, self.groups, 1)
        weighted = (w * g).view(B, D)                    # Σ w_i * group_i
        return base + weighted                           # residual add


# --------------------------- ResNet Backbone ---------------------------------
class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
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

        # --- COS_LAYER-aware classifier init ---
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

        # SupCon projection head (default source = BNNeck)
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
            print("[make_model][init] SupCon head enabled (Backbone): output is UN-normalized; "
                  "L2 normalization is handled in SupConLoss.")

    def forward(self, x, label=None):
        x = self.base(x)
        global_feat = self.gap(x).view(x.size(0), -1)
        feat = global_feat if self.neck == 'no' else self.bottleneck(global_feat)

        z_supcon = None
        if self.training and self.supcon_enabled:
            sup_in = _pick_feat_by_src(self.supcon_feat_src, global_feat, feat)
            if not self._supcon_log_done:
                print(f"[make_model][combo] [SupCon] src=model::{self.supcon_feat_src} "
                      f"| head=2xLinear(->128) | output=UN-normalized (L2 in SupConLoss)")
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


# ----------------------- ViT/DeiT Transformer (global) -----------------------
class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print(f'using Transformer_type: {cfg.MODEL.TRANSFORMER_TYPE} as a backbone')

        if not cfg.MODEL.SIE_CAMERA:
            camera_num = 0
        if not cfg.MODEL.SIE_VIEW:
            view_num = 0

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

        # SupCon head
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
            print("[make_model][init] SupCon head enabled (Transformer): output is UN-normalized; "
                  "L2 normalization is handled in SupConLoss.")


# ------------------- ViT/DeiT Transformer (with JPM local) -------------------
class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.rearrange = rearrange

        print(f'using Transformer_type: {cfg.MODEL.TRANSFORMER_TYPE} as a backbone')

        if not cfg.MODEL.SIE_CAMERA:
            camera_num = 0
        if not cfg.MODEL.SIE_VIEW:
            view_num = 0

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

        # Classifiers
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        self.shift_num = cfg.MODEL.SHIFT_NUM
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH

        # ---------------- TinyCLIP + AFEM integration (HF-only, vision-only branch) ----------------
        self.use_clip = getattr(cfg.MODEL, "USE_CLIP", False)
        if self.use_clip:
            hf_id = getattr(cfg.MODEL, "CLIP_HF_ID", "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M")
            local_path = getattr(cfg.MODEL, "CLIP_LOCAL_PATH", "").strip()
            model_id_or_path = local_path if local_path else hf_id
            print(f"[make_model][init] Loading TinyCLIP (HF) from: {model_id_or_path}")

            # (1) Load processor and TinyCLIP vision-only branch
            self.hf_processor = CLIPImageProcessor.from_pretrained(model_id_or_path, use_fast=True)
            hf_clip = CLIPModel.from_pretrained(model_id_or_path)
            self.clip_model = hf_clip.vision_model        # ← keep vision-only (no text encoder)
            self.clip_impl = "hf"

            # (2) Enable runtime interpolation for arbitrary input sizes (224→256/320)
            self.clip_model.config.interpolate_pos_encoding = True

            # (3) Configurable input size (fallback to processor default)
            default_h = self.hf_processor.size.get("shortest_edge", None)
            if default_h is None:
                default_h = self.hf_processor.size.get("height", 224)
            self.clip_input_size = tuple(getattr(cfg.MODEL, "CLIP_INPUT_SIZE", (default_h, default_h)))

            # (4) Register normalization buffers for ImageNet ↔ CLIP conversion
            im_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            im_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            clip_mean = torch.tensor(self.hf_processor.image_mean).view(1, 3, 1, 1)
            clip_std  = torch.tensor(self.hf_processor.image_std).view(1, 3, 1, 1)
            self.register_buffer("imnet_mean", im_mean, persistent=False)
            self.register_buffer("imnet_std",  im_std,  persistent=False)
            self.register_buffer("clip_mean",  clip_mean, persistent=False)
            self.register_buffer("clip_std",   clip_std,  persistent=False)

            # (5) ε-level fine-tuning control (only affects the vision branch)
            self.clip_finetune = bool(getattr(cfg.MODEL, "CLIP_FINETUNE", True))
            for p in self.clip_model.parameters():
                p.requires_grad = self.clip_finetune

            # (6) Output dimension from vision_config.hidden_size (not projection_dim)
            self.clip_output_dim = int(hf_clip.config.vision_config.hidden_size)

           
            # (7) Fusion heads (paper-aligned):
            # Tu = FC([Ta ⊕ Ts_raw]) where Ta∈R^{in_planes}, Ts_raw∈R^{clip_output_dim}
            self.fuse_fc = nn.Linear(self.in_planes + self.clip_output_dim, self.in_planes)

            # Ts' = Proj(AFEM(Ts_raw))
            self.afem = AFEM(dim=self.clip_output_dim, groups=32)
            self.sem_refine_proj = nn.Linear(self.clip_output_dim, self.in_planes)


            print(f"[make_model][init] TinyCLIP (HF) ready. input={self.clip_input_size}, "
                  f"proj_dim={self.clip_output_dim}, AFEM(groups=32), finetune={self.clip_finetune}.")
            
            # (8) Target CLIP input size we actually want to run at (e.g. 256 or 320).
            # Priority:
            #   - cfg.MODEL.CLIP_INPUT_SIZE if given, e.g. (256,256) or (320,320)
            #   - else fall back to processor's default square size
            #
            # IMPORTANT: this becomes the "real" spatial size we feed TinyCLIP,
            # and we will also resize TinyCLIP's positional embedding to match.
            if hasattr(cfg.MODEL, "CLIP_INPUT_SIZE"):
                self.clip_target_size = tuple(cfg.MODEL.CLIP_INPUT_SIZE)
            else:
                # fallback: use whatever was inferred earlier for self.clip_input_size
                # (you already computed self.clip_input_size above)
                self.clip_target_size = tuple(self.clip_input_size)

            # Safety: force it to be (H,W) both ints
            self.clip_target_size = (int(self.clip_target_size[0]), int(self.clip_target_size[1]))

            # (9) We'll only resize TinyCLIP positional embeddings ONCE per model init/first forward.
            self._clip_pos_resized = False


        # SupCon head
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
            print("[make_model][init] SupCon head enabled (TransformerLocal).")

    def forward(self, x, label=None, cam_label=None, view_label=None):
        x_img = x
        features = self.base(x, cam_label=cam_label, view_label=view_label)
        b1_feat = self.b1(features)
        global_feat = b1_feat[:, 0]

        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]

        def extract_local(start, end):
            lf = x[:, start:end]
            lf = self.b2(torch.cat((token, lf), dim=1))
            return lf[:, 0]

        local_feat_1 = extract_local(0, patch_length)
        local_feat_2 = extract_local(patch_length, patch_length * 2)
        local_feat_3 = extract_local(patch_length * 2, patch_length * 3)
        local_feat_4 = extract_local(patch_length * 3, patch_length * 4)
        
        if self.use_clip:
            # 1. Use the exact same input image tensor the main backbone saw
            #    (already augmented and resized to e.g. 320×320 by the dataloader / main pipeline).
            x_clip = x_img  # [B,3,H,W]

            # 2. Convert ImageNet-style normalization -> CLIP-style normalization.
            #    First undo ImageNet norm, then apply CLIP mean/std.
            x_unnorm = x_clip * self.imnet_std + self.imnet_mean
            x_clip = (x_unnorm - self.clip_mean) / self.clip_std  # now matches CLIP pixel_values space

            # 3. Force TinyCLIP to run at our desired resolution (e.g. 256x256 or 320x320),
            #    not just the pretraining 224x224.
            target_h, target_w = self.clip_target_size
            if x_clip.shape[-2:] != (target_h, target_w):
                x_clip = F.interpolate(
                    x_clip,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )

            # 4. On the first forward only:
            #    - Resize TinyCLIP's positional embeddings to the new patch grid.
            #    - Patch TinyCLIP's internal "image_size" bookkeeping so it stops assuming 224.
            if not self._clip_pos_resized:
                _hf_resize_pos_embed(self.clip_model, (target_h, target_w))

                # Sync model metadata so HF's forward() won't complain about 224x224 mismatch.
                if hasattr(self.clip_model, "config") and hasattr(self.clip_model.config, "image_size"):
                    # vision_model.config.image_size (int)
                    self.clip_model.config.image_size = target_h  # e.g. 320
                if hasattr(self.clip_model, "embeddings") and hasattr(self.clip_model.embeddings, "image_size"):
                    # CLIPVisionEmbeddings.image_size (int cached at init time)
                    self.clip_model.embeddings.image_size = target_h

                self._clip_pos_resized = True
                print(f"[make_model][clip] resized TinyCLIP pos_embed to {target_h}x{target_w} (patch OK)")

            # 5. Run TinyCLIP's vision tower at this higher resolution.
            #    We explicitly set interpolate_pos_encoding=True to bypass HF's strict size check
            #    and allow arbitrary spatial size.
            def _clip_vision_pool(outputs):
                """
                Safely extract a pooled visual embedding [B, C] from various HF return types.
                Priority:
                - pooler_output
                - last_hidden_state[:,0]
                - first tensor in tuple/list
                """
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    return outputs.pooler_output
                if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    xx = outputs.last_hidden_state  # [B, N, C]
                    return xx[:, 0] if xx.dim() == 3 else xx
                if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                    xx = outputs[0]
                    return xx[:, 0] if (torch.is_tensor(xx) and xx.dim() == 3) else xx
                if torch.is_tensor(outputs):
                    return outputs[:, 0] if outputs.dim() == 3 else outputs
                raise RuntimeError("Unsupported CLIP vision outputs structure.")

            # New: try to pass interpolate_pos_encoding=True (newer HF).
            # Fallback: call without it (older HF).
            try:
                out_clip = self.clip_model(pixel_values=x_clip, interpolate_pos_encoding=True)
            except TypeError:
                out_clip = self.clip_model(pixel_values=x_clip)

            # Raw TinyCLIP semantic embedding (CLS / pooled visual token), shape [B, clip_output_dim]
            ts_raw = _clip_vision_pool(out_clip)

            # 6. Fuse TinyCLIP semantics with our transformer global feature via AFEM (+ projection heads).
            #
            #   Tu      = fuse_fc([Ta ⊕ Ts_raw])
            #   Ts_ref  = AFEM(Ts_raw)
            #   Ts'     = sem_refine_proj(Ts_ref)
            #   T_final = Tu + Ts'
            #
            # where:
            #   Ta          : our global_feat from the ViT/DeiT branch  (dim = in_planes)
            #   Ts_raw      : TinyCLIP pooled vision feature            (dim = clip_output_dim)
            #   fuse_fc     : Linear(in_planes + clip_output_dim -> in_planes)
            #   AFEM        : learns channel-group reweighting on Ts_raw
            #   sem_refine_proj: Linear(clip_output_dim -> in_planes)
            #
            Tu = self.fuse_fc(torch.cat([global_feat, ts_raw], dim=1))          # [B, in_planes]
            Ts_prime = self.sem_refine_proj(self.afem(ts_raw))                  # [B, in_planes]

            # final fused global feature replaces the old global_feat for downstream heads/losses
            global_feat = Tu + Ts_prime

        # pre-BN feature for downstream heads/losses (works with or without CLIP)
        pre_bn_global = global_feat

        # BNNecks
        feat_bn_global = self.bottleneck(pre_bn_global)
        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        # SupCon feature
        z_supcon = None
        if self.training and self.supcon_enabled:
            sup_in = feat_bn_global if self.supcon_feat_src == "bnneck" else pre_bn_global
            if not self._supcon_log_done:
                print(f"[make_model][combo] [SupCon] src=model::{self.supcon_feat_src}")
                self._supcon_log_done = True
            z_supcon = self.supcon_head(sup_in)

        if self.training:
            cls_score = self.classifier(feat_bn_global)
            scores = [
                cls_score,
                self.classifier_1(local_feat_1_bn),
                self.classifier_2(local_feat_2_bn),
                self.classifier_3(local_feat_3_bn),
                self.classifier_4(local_feat_4_bn)
            ]
            feats = [pre_bn_global, local_feat_1, local_feat_2, local_feat_3, local_feat_4]
            feats_bn = [feat_bn_global, local_feat_1_bn, local_feat_2_bn,
                        local_feat_3_bn, local_feat_4_bn]
            return scores, feats, feats_bn, z_supcon
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat_bn_global,
                     local_feat_1_bn / 4, local_feat_2_bn / 4,
                     local_feat_3_bn / 4, local_feat_4_bn / 4],
                    dim=1
                )
            else:
                return torch.cat(
                    [pre_bn_global,
                     local_feat_1 / 4, local_feat_2 / 4,
                     local_feat_3 / 4, local_feat_4 / 4],
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


# ------------------------------ Factory & API --------------------------------
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
