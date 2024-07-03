import torch
import torch.nn as nn
from functools import partial
from einops import repeat, rearrange
import math
from timm.models.layers import trunc_normal_
# from timm.models.vision_transformer import Block
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
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ViTEncoderProjPredHeadMultiNoClsD3Momentum(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=8, dim=128, mlp_dim=2048, projector_depth=3, predictor_depth=2, drop_path_rate=0.1):
        super().__init__()
        self.base_encoder = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads, drop_path_rate=drop_path_rate)
        self.momentum_encoder = VisionTransformer(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, depth=depth, num_heads=num_heads)

        self.predictor_patch = self._build_mlp(predictor_depth, dim, mlp_dim, dim)
        self.projector_patch = self._build_mlp(projector_depth, embed_dim, mlp_dim, dim)
        self.predictor_divide_32 = self._build_mlp(predictor_depth, dim, mlp_dim, dim)
        self.projector_divide_32 = self._build_mlp(projector_depth, embed_dim, mlp_dim, dim)
        self.predictor_divide_112 = self._build_mlp(predictor_depth, dim, mlp_dim, dim)
        self.projector_divide_112 = self._build_mlp(projector_depth, embed_dim, mlp_dim, dim)
        self.length = img_size // patch_size 

        self.projector_divide_momentum_32 = self._build_mlp(projector_depth, embed_dim, mlp_dim, dim)
        self.projector_divide_momentum_112 = self._build_mlp(projector_depth, embed_dim, mlp_dim, dim)
        self.projector_patch_momentum = self._build_mlp(projector_depth, embed_dim, mlp_dim, dim)
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False
        for param_b, param_m in zip(self.projector_patch.parameters(), self.projector_patch_momentum.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False
        for param_b, param_m in zip(self.projector_divide_32.parameters(), self.projector_divide_momentum_32.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False
        for param_b, param_m in zip(self.projector_divide_112.parameters(), self.projector_divide_momentum_112.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False
        self.apply(self._init_weights)

    def forward(self, x1, x2, patch_indexs1, patch_indexs2, patch_indexs3, patch_indexs4, patch_indexs5, patch_indexs6, divide_size_32, divide_size_112, m):
        z1 = self.base_encoder(x1)
        z2 = self.base_encoder(x2)
        c = z1.shape[2]
        # divide_size_32_select 
        dz1 = self.token_pool(z1[:,1:], divide_size_32)
        dz2 = self.token_pool(z2[:,1:], divide_size_32)
        dz1 = torch.gather(dz1, dim=1, index=patch_indexs3.unsqueeze(-1).repeat(1, 1, c))
        dz2 = torch.gather(dz2, dim=1, index=patch_indexs4.unsqueeze(-1).repeat(1, 1, c))
        dz1 = rearrange(dz1, 'b n c ->(b n) c') 
        dz2 = rearrange(dz2, 'b n c ->(b n) c') 

        # divide_size_112_select 
        lz1 = self.token_pool(z1[:,1:], divide_size_112)
        lz2 = self.token_pool(z2[:,1:], divide_size_112)
        lz1 = torch.gather(lz1, dim=1, index=patch_indexs5.unsqueeze(-1).repeat(1, 1, c))
        lz2 = torch.gather(lz2, dim=1, index=patch_indexs6.unsqueeze(-1).repeat(1, 1, c))
        lz1 = rearrange(lz1, 'b n c ->(b n) c') 
        lz2 = rearrange(lz2, 'b n c ->(b n) c') 

        # patch_select 
        z1 = torch.gather(z1[:, 1:], dim=1, index=patch_indexs1.unsqueeze(-1).repeat(1, 1, c))
        z2 = torch.gather(z2[:, 1:], dim=1, index=patch_indexs2.unsqueeze(-1).repeat(1, 1, c))
        z1 = rearrange(z1, 'b n c ->(b n) c') 
        z2 = rearrange(z2, 'b n c ->(b n) c') 
        # projectior
        dz1 = self.projector_divide_32(dz1)
        dz2 = self.projector_divide_32(dz2)
        lz1 = self.projector_divide_112(lz1)
        lz2 = self.projector_divide_112(lz2)
        z1 = self.projector_patch(z1)
        z2 = self.projector_patch(z2)
        # predictor
        dp1 = self.predictor_divide_32(dz1)
        dp2 = self.predictor_divide_32(dz2)
        lp1 = self.predictor_divide_112(lz1)
        lp2 = self.predictor_divide_112(lz2)  
        p1 = self.predictor_patch(z1)
        p2 = self.predictor_patch(z2)
 
        with torch.no_grad():
            self._update_momentum_encoder(m)
            mz1 = self.momentum_encoder(x1)
            mz2 = self.momentum_encoder(x2) 

            mdz1 = self.token_pool(mz1[:,1:], divide_size_32)
            mdz2 = self.token_pool(mz2[:,1:], divide_size_32)
            mdz1 = torch.gather(mdz1, dim=1, index=patch_indexs3.unsqueeze(-1).repeat(1, 1, c))
            mdz2 = torch.gather(mdz2, dim=1, index=patch_indexs4.unsqueeze(-1).repeat(1, 1, c))
            mdz1 = rearrange(mdz1, 'b n c ->(b n) c') 
            mdz2 = rearrange(mdz2, 'b n c ->(b n) c') 

            mlz1 = self.token_pool(mz1[:,1:], divide_size_112)
            mlz2 = self.token_pool(mz2[:,1:], divide_size_112)
            mlz1 = torch.gather(mlz1, dim=1, index=patch_indexs5.unsqueeze(-1).repeat(1, 1, c))
            mlz2 = torch.gather(mlz2, dim=1, index=patch_indexs6.unsqueeze(-1).repeat(1, 1, c))
            mlz1 = rearrange(mlz1, 'b n c ->(b n) c') 
            mlz2 = rearrange(mlz2, 'b n c ->(b n) c') 

            mz1 = torch.gather(mz1[:, 1:], dim=1, index=patch_indexs1.unsqueeze(-1).repeat(1, 1, c))
            mz2 = torch.gather(mz2[:, 1:], dim=1, index=patch_indexs2.unsqueeze(-1).repeat(1, 1, c))
            mz1 = rearrange(mz1, 'b n c ->(b n) c') 
            mz2 = rearrange(mz2, 'b n c ->(b n) c') 

            mdz1 = self.projector_divide_momentum_32(mdz1)
            mdz2 = self.projector_divide_momentum_32(mdz2)
            mlz1 = self.projector_divide_momentum_112(mlz1)
            mlz2 = self.projector_divide_momentum_112(mlz2)
            mz1 = self.projector_patch_momentum(mz1)
            mz2 = self.projector_patch_momentum(mz2)

        return p1, p2, mz1, mz2, dp1, dp2, mdz1, mdz2, lp1, lp2, mlz1, mlz2
        # return cp1, cp2, cz1, cz2

    def token_pool(self, features, divide_size):
        B = features.shape[0]
        features = rearrange(features, 'b (h w) c -> b h w c', h = self.length)   # (batch_size, npatch, out_dim)
        pool_features = rearrange(features, 'b (h1 d1) (w1 d2) c -> (b h1 w1)  c (d1 d2)', d1=divide_size, d2=divide_size) 
        pool_features = nn.AdaptiveAvgPool1d(1)(pool_features)
        pool_features = torch.flatten(pool_features,1)
        pool_features = rearrange(pool_features, '(b n) c -> b n c', b = B)
        return pool_features

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
        for param_b, param_m in zip(self.projector_patch.parameters(), self.projector_patch_momentum.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
        for param_b, param_m in zip(self.projector_divide_32.parameters(), self.projector_divide_momentum_32.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)
        for param_b, param_m in zip(self.projector_divide_112.parameters(), self.projector_divide_momentum_112.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.build_2d_sincos_position_embedding()
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
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
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)
    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x
    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)
    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output
    
    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h = self.patch_embed.grid_size
        w = self.patch_embed.grid_size
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


