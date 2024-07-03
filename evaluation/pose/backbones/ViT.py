# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Object detection and instance segmentation with XCiT backbone
Based on mmdet, timm and DeiT code bases
https://github.com/open-mmlab/mmdetection
https://github.com/rwightman/pytorch-image-models/tree/master/timm
https://github.com/facebookresearch/deit/
"""
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner import load_state_dict
from mmengine.utils import to_2tuple

from mmpose.registry import MODELS

from .base_backbone import BaseBackbone
from .utils import get_state_dict

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from timm.models.vision_transformer import _cfg, Mlp
from timm.models.registry import register_model
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
import numpy as np
from .utils import load_checkpoint
# Copyright (c) OpenMMLab. All rights reserved.


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=[224, 224], patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches_w = img_size[0] // patch_size
        self.num_patches_h = img_size[1] // patch_size

        num_patches = self.num_patches_w * self.num_patches_h
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)
        return x

# class PatchEmbed(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, img_size=[224, 224], patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         self.num_patches_w = img_size[0] // patch_size
#         self.num_patches_h = img_size[1] // patch_size

#         num_patches = self.num_patches_w * self.num_patches_h
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            
#     def forward(self, x, mask=None):
#         B, C, H, W = x.shape
#         return self.proj(x)


@MODELS.register_module()
class ViT(BaseBackbone):
    """
    Based on timm and DeiT code bases
    https://github.com/rwightman/pytorch-image-models/tree/master/timm
    https://github.com/facebookresearch/deit/
    """

    def __init__(self, img_size=[224, 224], patch_size=16, in_chans=3, num_classes=80, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 out_indices=[2, 5, 8, 11], abs = True, pretrained = None, init_cfg = None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            cls_attn_layers: (int) Depth of Class attention layers
            use_pos: (bool) whether to use positional encoding
            eta: (float) layerscale initialization value
            tokens_norm: (bool) Whether to normalize all tokens or just the cls_token in the CA
            out_indices: (list) Indices of layers from which FPN features are extracted
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.img_size = img_size
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches +1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.out_indices = out_indices
        if abs:
            self.build_2d_sincos_position_embedding()
        self.pretrained = pretrained
        self.init_cfg = init_cfg
        # self.apply(self._init_weights)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def init_weights(self):
        """Initialize the weights in backbone."""

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            state_dict = get_state_dict(
                self.init_cfg['checkpoint'], map_location='cpu')
            logger.info(f'Load pre-trained model for '
                        f'{self.__class__.__name__} from original repo')
            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    logger.info(msg=f'Resize the pos_embed shape from '
                                f'{state_dict["pos_embed"].shape} to '
                                f'{self.pos_embed.shape}')            
                    state_dict['pos_embed'] = self.resize_pos_embed(state_dict['pos_embed'])
            # mis, une = load_state_dict(self, state_dict, strict=False, logger=logger)
            # load_state_dict(self, state_dict, strict=False, logger=logger)
            # logger.warn(msg=f'missing key {mis}')
            # logger.warn(msg=f'unexpected key {une}')
            mis, une = self.load_state_dict(state_dict, False)
            logger.info(msg=f'missing key {mis}')
            logger.info(msg=f'unexpected key {une}')

        else:
            super(ViT, self).init_weights()

    def resize_pos_embed(self, pos_embed_old):
        N = pos_embed_old.shape[1] - 1
        class_pos_embed = pos_embed_old[:, 0]
        patch_pos_embed = pos_embed_old[:, 1:]
        dim = pos_embed_old.shape[-1]
        w0 = self.img_size[0] // self.patch_embed.patch_size
        h0 = self.img_size[1] // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )        
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

 

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding
        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        # x = x + self.interpolate_pos_encoding(x, w, h)
        x = x + self.pos_embed

        return self.pos_drop(x)



    def forward_features(self, x):
        B, C, H, W = x.shape
        H, W = H // self.patch_embed.patch_size, W // self.patch_embed.patch_size
        x  = self.prepare_tokens(x)

        for i, layer in enumerate(self.blocks):
            x = layer(x)
        x = self.norm(x)
        x = x[:, 1:]
        x = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        return x


    def forward(self, x):
        x = self.forward_features(x)
        # print(x.shape)
        outs = []
        outs.append(x)
        return outs

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h = self.patch_embed.num_patches_h
        w = self.patch_embed.num_patches_w
        # h = w = 14
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        # assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False