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
#   - without SupCon: (cls_score, global_feat, feat_bn)
#   - with    SupCon: (cls_score, global_feat, feat_bn, z_supcon)
#
# Training-time returns (transformer_local):
#   - without SupCon: (scores, feats, feat_bn_global)
#   - with    SupCon: (scores, feats, feat_bn_global, z_supcon)
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
# =============================================================================

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
    """Pick feature by source string: 'pre_bn' | 'bnneck' (default to pre_bn)."""
    return bn_feat if src == "bnneck" else global_feat


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

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # SupCon projection head (default source = BNNeck to keep current behavior)
        self.supcon_enabled = (
            hasattr(cfg, "LOSS")
            and hasattr(cfg.LOSS, "SUPCON")
            and getattr(cfg.LOSS.SUPCON, "ENABLE", False)
        )
        if self.supcon_enabled:
            self.supcon_feat_src = getattr(cfg.LOSS.SUPCON, "FEAT_SRC", "bnneck")  # default: bnneck
            self.supcon_head = nn.Sequential(
                nn.Linear(self.in_planes, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 128)
            )

    def forward(self, x, label=None):
        x = self.base(x)
        global_feat = F.avg_pool2d(x, x.shape[2:4]).view(x.size(0), -1)  # pre-BN
        feat = global_feat if self.neck == 'no' else self.bottleneck(global_feat)  # BNNeck

        if self.training:
            cls_score = self.arcface(feat, label) if self.cos_layer else self.classifier(feat)

            if self.supcon_enabled:
                sup_in = _pick_feat_by_src(self.supcon_feat_src, global_feat, feat)
                z_supcon = F.normalize(self.supcon_head(sup_in), dim=1)
                # training-time returns with SupCon
                return cls_score, global_feat, feat, z_supcon
            else:
                # training-time returns without SupCon
                return cls_score, global_feat, feat
        else:
            return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            self.state_dict()[k].copy_(v)
        print(f'Loading pretrained model from {trained_path}')

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for k, v in param_dict.items():
            self.state_dict()[k].copy_(v)
        print(f'Loading pretrained model for finetuning from {model_path}')


# ----------------------- ViT/DeiT Transformer (global) -----------------------
class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
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

        self.gap = nn.AdaptiveAvgPool2d(1)

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

        # SupCon projection head (default source = BNNeck to keep current behavior)
        self.supcon_enabled = (
            hasattr(cfg, "LOSS")
            and hasattr(cfg.LOSS, "SUPCON")
            and getattr(cfg.LOSS.SUPCON, "ENABLE", False)
        )
        if self.supcon_enabled:
            self.supcon_feat_src = getattr(cfg.LOSS.SUPCON, "FEAT_SRC", "bnneck")  # default: bnneck
            self.supcon_head = nn.Sequential(
                nn.Linear(self.in_planes, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 128)
            )

    def forward(self, x, label=None, cam_label=None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)  # pre-BN token-pooled feat
        feat = self.bottleneck(global_feat)  # BNNeck

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            if self.supcon_enabled:
                sup_in = _pick_feat_by_src(self.supcon_feat_src, global_feat, feat)
                z_supcon = F.normalize(self.supcon_head(sup_in), dim=1)
                return cls_score, global_feat, feat, z_supcon
            else:
                return cls_score, global_feat, feat
        else:
            return feat if self.neck_feat == 'after' else global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for k, v in param_dict.items():
            self.state_dict()[k.replace('module.', '')].copy_(v)
        print(f'Loading pretrained model from {trained_path}')

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for k, v in param_dict.items():
            self.state_dict()[k].copy_(v)
        print(f'Loading pretrained model for finetuning from {model_path}')


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

        # SupCon projection head (default source = BNNeck to keep current behavior)
        self.supcon_enabled = (
            hasattr(cfg, "LOSS")
            and hasattr(cfg.LOSS, "SUPCON")
            and getattr(cfg.LOSS.SUPCON, "ENABLE", False)
        )
        if self.supcon_enabled:
            self.supcon_feat_src = getattr(cfg.LOSS.SUPCON, "FEAT_SRC", "bnneck")  # default: bnneck
            self.supcon_head = nn.Sequential(
                nn.Linear(self.in_planes, 2048),
                nn.ReLU(inplace=True),
                nn.Linear(2048, 128)
            )

    def forward(self, x, label=None, cam_label=None, view_label=None):
        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch (pre-BN)
        b1_feat = self.b1(features)  # [B, tokens, C]
        global_feat = b1_feat[:, 0]

        # JPM branch
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

        # BN for global (used by CE / optional for TripletX if chosen)
        feat_bn_global = self.bottleneck(global_feat)
        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat_bn_global, label)
                scores = [cls_score]
            else:
                cls_score = self.classifier(feat_bn_global)
                scores = [
                    cls_score,
                    self.classifier_1(local_feat_1_bn),
                    self.classifier_2(local_feat_2_bn),
                    self.classifier_3(local_feat_3_bn),
                    self.classifier_4(local_feat_4_bn)
                ]

            # pre-BN feats for metric learning (global + 4 locals)
            feats = [global_feat, local_feat_1, local_feat_2, local_feat_3, local_feat_4]

            if self.supcon_enabled:
                sup_in = _pick_feat_by_src(self.supcon_feat_src, global_feat, feat_bn_global)
                z_supcon = F.normalize(self.supcon_head(sup_in), dim=1)
                return scores, feats, feat_bn_global, z_supcon
            else:
                return scores, feats, feat_bn_global
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat_bn_global,
                     local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4],
                    dim=1
                )
            else:
                return torch.cat(
                    [global_feat,
                     local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4],
                    dim=1
                )

    # Added load_param() for evaluation checkpoint loading
    def load_param(self, trained_path: str, strict: bool = False):
        import os
        assert os.path.isfile(trained_path), f"Weight file not found: {trained_path}"
        state = torch.load(trained_path, map_location="cpu")

        # compatible formats
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]

        # strip DataParallel prefix
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
