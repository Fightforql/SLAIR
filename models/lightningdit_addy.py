"""
Lightning DiT's codes are built from original DiT & SiT.
(https://github.com/facebookresearch/DiT; https://github.com/willisma/SiT)
It demonstrates that a advanced DiT together with advanced diffusion skills
could also achieve a very promising result with 1.35 FID on ImageNet 256 generation.

Enjoy everyone, DiT strikes back!

by Maple (Jingfeng Yao) from HUST-VL
"""

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import PatchEmbed, Mlp
from models.swiglu_ffn import SwiGLUFFN 
from models.pos_embed import VisionRotaryEmbeddingFast
from models.rmsnorm import RMSNorm

@torch.compile
def modulate(x, shift, scale):
    if scale.ndim == 2:
        scale = scale.unsqueeze(1)
    if shift is not None and shift.ndim == 2:
        shift = shift.unsqueeze(1)
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift

class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            q = rope(q)
            k = rope(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t: A 1-D Tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
        Returns:
            An (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return embedding
    
    @torch.compile
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class YToTNet(nn.Module):
    """
    Estimate timestep t from conditioning input y (B, C, H, W).
    Outputs t with shape (B, 1). Resolution-agnostic via global pooling.
    """
    def __init__(self, in_channels: int, base_channels: int = 64, num_layers: int = 4):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            out_channels = base_channels if i == 0 else base_channels * (2 ** (i - 1))
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels))
            layers.append(nn.SiLU())
            channels = out_channels
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, 1),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(y)
        return self.head(feat)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    Same as DiT.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    @torch.compile
    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        labels = labels.to(self.embedding_table.weight.device)
        embeddings = self.embedding_table(labels)
        return embeddings


class SimpleConditionUNet(nn.Module):
    """
    Lightweight UNet for processing condition c to extract multi-resolution features.
    Uses depthwise separable convolutions and bottleneck structure to reduce parameters.
    """
    def __init__(self, hidden_size, num_scales=3, reduction=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_scales = num_scales
        self.reduction = reduction  # Channel reduction factor for bottleneck
        reduced_dim = hidden_size // reduction
        
        def depthwise_separable_conv(in_dim, out_dim, kernel=3, stride=1, padding=1):
            """Depthwise separable convolution: much fewer parameters than standard conv"""
            return nn.Sequential(
                # Depthwise convolution
                nn.Conv2d(in_dim, in_dim, kernel, stride=stride, padding=padding, groups=in_dim, bias=False),
                nn.GroupNorm(min(8, in_dim), in_dim),
                nn.SiLU(),
                # Pointwise convolution
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.GroupNorm(min(8, out_dim), out_dim),
                nn.SiLU(),
            )
        
        # Encoder: simplified with bottleneck and depthwise separable convs
        # Level 1: Full resolution (with bottleneck)
        self.enc1 = depthwise_separable_conv(hidden_size, reduced_dim)
        self.downsample1 = nn.Conv2d(reduced_dim, reduced_dim, 3, stride=2, padding=1, groups=reduced_dim)  # Depthwise
        
        # Level 2: Half resolution
        self.enc2 = depthwise_separable_conv(reduced_dim, reduced_dim)
        self.downsample2 = nn.Conv2d(reduced_dim, reduced_dim, 3, stride=2, padding=1, groups=reduced_dim)  # Depthwise
        
        # Middle (bottleneck) - minimal processing
        self.mid = depthwise_separable_conv(reduced_dim, reduced_dim)
        
        # Decoder: upsampling with skip connections
        # Level 2 -> Level 1
        self.upsample1 = nn.ConvTranspose2d(reduced_dim, reduced_dim, 2, stride=2, groups=reduced_dim)  # Depthwise
        self.dec1 = depthwise_separable_conv(reduced_dim * 2, reduced_dim)  # Skip connection
        
        # Level 1 -> Full resolution
        self.upsample2 = nn.ConvTranspose2d(reduced_dim, reduced_dim, 2, stride=2, groups=reduced_dim)  # Depthwise
        self.dec2 = depthwise_separable_conv(reduced_dim * 2, reduced_dim)  # Skip connection
        
        # Compress input c_spatial to half channels
        self.input_compress = nn.Conv2d(hidden_size, hidden_size // 2, 1)
        
        # Final projection to half of hidden_size (UNet will output half, input provides other half)
        self.final_proj = nn.Conv2d(reduced_dim, hidden_size // 2, 1)

    def forward(self, c, ph, pw):
        """
        Args:
            c: (N, T, D) condition tensor, where T = ph * pw
            ph, pw: spatial dimensions
        Returns:
            full_seq: (N, T, D) processed condition tensor
        """
        # Reshape c from (N, T, D) to (N, D, H, W)
        B, T, D = c.shape
        assert T == ph * pw, f"Token count mismatch: T={T} != ph*pw={ph*pw}"
        c_spatial = c.view(B, ph, pw, D).permute(0, 3, 1, 2)  # (N, D, ph, pw)
        
        # Encoder: Full resolution -> Half resolution
        enc1 = self.enc1(c_spatial)  # (N, reduced_dim, ph, pw)
        enc_down1 = self.downsample1(enc1)  # (N, reduced_dim, ph//2, pw//2)
        
        # Encoder: Half resolution -> Quarter resolution
        enc2 = self.enc2(enc_down1)  # (N, reduced_dim, ph//2, pw//2)
        enc_down2 = self.downsample2(enc2)  # (N, reduced_dim, ph//4, pw//4)
        
        # Middle (bottleneck)
        mid = self.mid(enc_down2)  # (N, reduced_dim, ph//4, pw//4)
        
        # Decoder: Quarter -> Half resolution
        mid_up = self.upsample1(mid)  # (N, reduced_dim, ph//2, pw//2)
        # Ensure dimensions match for skip connection
        if mid_up.shape[2:] != enc2.shape[2:]:
            mid_up = F.interpolate(mid_up, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([enc2, mid_up], dim=1)  # Skip connection from enc2
        dec1 = self.dec1(dec1)  # (N, reduced_dim, ph//2, pw//2)
        
        # Decoder: Half -> Full resolution
        dec_up = self.upsample2(dec1)  # (N, reduced_dim, ph, pw)
        # Ensure dimensions match for skip connection
        if dec_up.shape[2:] != enc1.shape[2:]:
            dec_up = F.interpolate(dec_up, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([enc1, dec_up], dim=1)  # Skip connection from enc1
        dec2 = self.dec2(dec2)  # (N, reduced_dim, ph, pw)
        
        # Compress input c_spatial to half channels
        c_compressed = self.input_compress(c_spatial)  # (N, D//2, ph, pw)
        
        # Final projection to half channels
        full_unet = self.final_proj(dec2)  # (N, D//2, ph, pw)
        
        # Interleave channels: odd positions (0,2,4,...) from compressed input,
        # even positions (1,3,5,...) from UNet output
        # This preserves positional embedding structure (height/width encoding alternates)
        full = torch.zeros(B, D, ph, pw, device=c_spatial.device, dtype=c_spatial.dtype)
        full[:, 0::2, :, :] = c_compressed  # Odd indices: compressed input
        full[:, 1::2, :, :] = full_unet     # Even indices: UNet output
        
        full_seq = full.permute(0, 2, 3, 1).contiguous().view(B, ph * pw, D)
        
        return full_seq

class LightningDiTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False, 
        use_rmsnorm=False,
        wo_shift=False,
        mlp_drop=0.0,
        **block_kwargs
    ):
        super().__init__()
        
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
            
        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )
        
        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=mlp_drop
            )
            
        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    @torch.compile
    def forward(self, x, c, feat_rope=None):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
            
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

class FinalLayer(nn.Module):
    """
    The final layer of LightningDiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        condition_dropout_prob=0.5,
        num_classes=10,
        learn_sigma=False,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        wo_shift=False,
        use_checkpoint=False,
        attn_drop=0.0,
        proj_drop=0.0,
        mlp_drop=0.0,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels if not learn_sigma else in_channels * 2
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        self.condition_dropout_prob = condition_dropout_prob
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.y_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # y_embedder_weight / pos_embed_weight removed: never used in forward (would trigger DDP unused-param error)
        num_patches = self.x_embedder.num_patches
        
        # Simple UNet for multi-resolution condition processing
        self.condition_unet = SimpleConditionUNet(hidden_size, num_scales=3)
        # Fixed sin-cos pos embeds as buffers so DDP does not expect gradients from them
        # mlp_head removed: never used in forward (triggered DDP unused-param indices 8,9)

        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            half_head_dim = hidden_size // num_heads // 2
            hw_seq_len = input_size // patch_size
            self.feat_rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.feat_rope = None

        self.blocks = nn.ModuleList([
            LightningDiTBlock(hidden_size, 
                     num_heads, 
                     mlp_ratio=mlp_ratio, 
                     use_qknorm=use_qknorm, 
                     use_swiglu=use_swiglu, 
                     use_rmsnorm=use_rmsnorm,
                     wo_shift=wo_shift,
                     mlp_drop=mlp_drop,
                     attn_drop=attn_drop,
                     proj_drop=proj_drop,
                     ) for _ in range(depth)
        ])
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))
        self.register_buffer("pos_embed_y", torch.zeros(1, num_patches, hidden_size))
        # skip_linears removed: forward() no longer uses skip connections (branch was commented out)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        w = self.y_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.y_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder_t.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, h=None, w=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H*p, W*p)

        h, w: number of patches along height / width (needed for non-square inputs)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if h is None or w is None:
            h = w = int(round(x.shape[1] ** 0.5))
        assert h * w == x.shape[1], \
            f"Patch count mismatch: h*w={h*w} != x.shape[1]={x.shape[1]}"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t=None, y=None, inputs=None, label=None):
        """
        Forward pass of LightningDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        use_checkpoint: boolean to toggle checkpointing

        Supports training-free multi-resolution inference via bicubic interpolation of
        the fixed sin-cos positional embeddings when the input spatial size differs from
        the training-time size.
        """
        use_checkpoint = self.use_checkpoint

        # Derive patch-grid dims from the actual input; enables arbitrary-resolution inference
        _, _, H, W = x.shape
        ph = H // self.patch_size
        pw = W // self.patch_size

        # Recompute positional embeddings for the current spatial resolution
        pos_embed   = torch.from_numpy(get_2d_sincos_pos_embed(self.hidden_size, (ph, pw))).float().unsqueeze(0).to(x.device)

        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = ph * pw
        t = self.t_embedder(t)               # (N, D)

        y = self.condition_unet(self.y_embedder(y), ph, pw) + pos_embed  # (N, T, D)
        # label_embed = self.label_embedder(label, self.training)
            # Expand label_embed to match y's shape: (N, D) -> (N, 1, D)
        # y = self.y_norm(y)

        # print(label_embed.shape, t.shape, y.shape)
        # y = self.y_embedder_t(y, self.training)    # (N, D)
        # c = t.unsqueeze(1) + y#  + label_embed.unsqueeze(1)                                 # (N, T, D)
        
        # Process c through UNet to get multi-resolution features for degradation analysis
        c_multi_scale = y + t.unsqueeze(1) # List of (N, T_i, D) at different scales
        # Store multi-scale features for potential use in degradation analysis
        # c_multi_scale[0]: full resolution (N, T, D)
        # c_multi_scale[1]: half resolution (N, T//4, D)
        # c_multi_scale[2]: quarter resolution (N, T//4, D)
        # Use the full-resolution output as the main condition (can be enhanced with multi-scale info)

        # Get RoPE callable for the current (ph, pw) resolution.
        # for_hw() returns self when size matches training, or a lightweight
        # _DynamicRopeFn with recomputed frequencies otherwise.
        feat_rope = self.feat_rope.for_hw(ph, pw) if self.feat_rope is not None else None

        skips = []
        for i, block in enumerate(self.blocks):
            if use_checkpoint:
                x = checkpoint(block, x, c_multi_scale, feat_rope, use_reentrant=True)
            else:
                x = block(x, c_multi_scale, feat_rope)
            # if i < self.depth // 2:
            #     if use_checkpoint:
            #         x = checkpoint(block, x, c, self.feat_rope, use_reentrant=True)
            #     else:
            #         x = block(x, c, self.feat_rope)
            #     skips.append(x)
            # else:
            #     skip_x = skips.pop()
            #     x = torch.cat([x, skip_x], dim=-1)
            #     x = self.skip_linears[self.depth - 1 - i](x)
            #     if use_checkpoint:
            #         x = checkpoint(block, x, c, self.feat_rope, use_reentrant=True)
            #     else:
            #         x = block(x, c, self.feat_rope)

        x = self.final_layer(x, c_multi_scale)                   # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, h=ph, w=pw)           # (N, out_channels, H, W)

        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y=None, inputs=None, cfg_scale=1.0, cfg_interval=None, cfg_interval_start=None, label=None):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)

        if inputs is not None:
            y = inputs
        # Zero out the second half of y for unconditional generation
        if len(y) == len(x):
            y_cond = y[: len(y) // 2]
        else:
            y_cond = y
        y_uncond = torch.zeros_like(y_cond)
        combined_y = torch.cat([y_cond, y_uncond], dim=0)

        # label_embedder removed; CFG uses only y (image condition)
        model_out = self.forward(combined, t, y=combined_y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        
        if cfg_interval is True:
            timestep = t[0]
            if timestep < cfg_interval_start:
                half_eps = cond_eps

        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, tuple):
        grid_h_size = grid_size[0]  # height (ph)
        grid_w_size = grid_size[1]  # width (pw)
        grid_h = np.arange(grid_h_size, dtype=np.float32)
        grid_w = np.arange(grid_w_size, dtype=np.float32)
    else:
        grid_h_size = grid_w_size = grid_size
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w / grid_w_size * 45, grid_h / grid_h_size * 45)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_h_size, grid_w_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                             LightningDiT Configs                              #
#################################################################################

def LightningDiT_XL_1(**kwargs):
    return LightningDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def LightningDiT_XL_2(**kwargs):
    return LightningDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def LightningDiT_L_2(**kwargs):
    return LightningDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def LightningDiT_B_1(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def LightningDiT_B_2(**kwargs):
    return LightningDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def LightningDiT_1p0B_1(**kwargs):
    return LightningDiT(depth=24, hidden_size=1536, patch_size=1, num_heads=24, **kwargs)

def LightningDiT_1p0B_2(**kwargs):
    return LightningDiT(depth=24, hidden_size=1536, patch_size=2, num_heads=24, **kwargs)

def LightningDiT_1p6B_1(**kwargs):
    return LightningDiT(depth=28, hidden_size=1792, patch_size=1, num_heads=28, **kwargs)

def LightningDiT_1p6B_2(**kwargs):
    return LightningDiT(depth=28, hidden_size=1792, patch_size=2, num_heads=28, **kwargs)

LightningDiT_models = {
    'LightningDiT-B/1': LightningDiT_B_1, 'LightningDiT-B/2': LightningDiT_B_2,
    'LightningDiT-L/2': LightningDiT_L_2,
    'LightningDiT-XL/1': LightningDiT_XL_1, 'LightningDiT-XL/2': LightningDiT_XL_2,
    'LightningDiT-1p0B/1': LightningDiT_1p0B_1, 'LightningDiT-1p0B/2': LightningDiT_1p0B_2,
    'LightningDiT-1p6B/1': LightningDiT_1p6B_1, 'LightningDiT-1p6B/2': LightningDiT_1p6B_2,
}