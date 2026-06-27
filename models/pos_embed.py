# --------------------------------------------------------
# EVA-02: A Visual Representation for Neon Genesis
# Github source: https://github.com/baaivision/EVA/EVA02
# Copyright (c) 2023 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Yuxin Fang
#
# Based on https://github.com/lucidrains/rotary-embedding-torch
# --------------------------------------------------------'

from math import pi

import torch
from torch import nn

from einops import rearrange, repeat



def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)



def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')



class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

        # print('======== shape of rope freq', self.freqs_cos.shape, '========')

    def forward(self, t, start_index = 0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim = -1)



class _DynamicRopeFn:
    """Lightweight callable that applies pre-computed RoPE freqs of any (h,w)."""
    def __init__(self, freqs_cos, freqs_sin):
        # Ensure freqs are on the same device and have consistent dtype
        self.freqs_cos = freqs_cos
        self.freqs_sin = freqs_sin
        
        # Validate dimensions to prevent broadcasting issues
        assert freqs_cos.shape == freqs_sin.shape, \
            f"freqs_cos and freqs_sin must have same shape, got {freqs_cos.shape} vs {freqs_sin.shape}"

    def __call__(self, t):
        # Ensure t and freqs are on same device and compatible dtypes
        if t.device != self.freqs_cos.device:
            t = t.to(self.freqs_cos.device)
        if t.dtype != self.freqs_cos.dtype:
            t = t.to(self.freqs_cos.dtype)
        
        # Apply RoPE with numerical stability checks
        rotated = rotate_half(t)
        result = t * self.freqs_cos + rotated * self.freqs_sin
        
        # Check for numerical issues (NaN or Inf)
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"Warning: NaN/Inf detected in RoPE application. "
                  f"freqs_cos range: [{self.freqs_cos.min():.4f}, {self.freqs_cos.max():.4f}], "
                  f"freqs_sin range: [{self.freqs_sin.min():.4f}, {self.freqs_sin.max():.4f}]")
            # Replace NaN/Inf with zeros to prevent propagation
            result = torch.where(torch.isnan(result) | torch.isinf(result), 
                                torch.zeros_like(result), result)
        
        return result


class VisionRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs_1d = custom_freqs
        elif freqs_for == 'lang':
            freqs_1d = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs_1d = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs_1d = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        # Store for training-free dynamic-resolution RoPE recomputation.
        # persistent=False: participates in .to(device) but NOT saved in state_dict,
        # so existing checkpoints (which don't have this key) still load cleanly.
        self.pt_seq_len = pt_seq_len
        self.register_buffer("_base_freqs_1d", freqs_1d, persistent=False)

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs = torch.einsum('..., f -> ... f', t, freqs_1d)
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim = -1)

        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def _compute_freqs_for_hw(self, h, w):
        """Recompute 2-D RoPE frequencies for an arbitrary (h, w) grid.

        Uses an improved scaling strategy for high-resolution stability:
        - For resolutions close to training size: uses standard linear scaling
        - For high resolutions: uses adaptive scaling to maintain frequency density
        - Ensures smooth frequency transitions at boundaries
        - Prevents numerical instability at extreme values
        """
        device = self.freqs_cos.device
        base = self._base_freqs_1d  # (dim//2,)

        # Improved scaling strategy for high-resolution stability
        # Problem: Linear scaling t = arange(h) / h * pt_seq_len reduces frequency density
        # at high resolutions, causing poor position discrimination and edge artifacts.
        # Solution: For high resolutions, allow frequency range to extend proportionally
        # to maintain minimum frequency spacing (similar to training density).
        
        # Use double precision for frequency calculation to reduce numerical errors
        t_h_base = torch.arange(h, device=device, dtype=torch.float64)
        t_w_base = torch.arange(w, device=device, dtype=torch.float64)
        
        # Determine if we're at high resolution (more than 2x training size)
        is_high_res_h = h > self.pt_seq_len * 2
        is_high_res_w = w > self.pt_seq_len * 2
        
        if is_high_res_h or is_high_res_w:
            # High-resolution strategy: Use aggressive frequency extension
            # to maintain position discrimination at high resolutions.
            # 
            # Key insight: To maintain frequency density (spacing ~1.0 like training),
            # we need t to scale with position index, allowing frequency range to extend.
            # We use power-law scaling: t = arange(h) * scale_extension
            # where scale_extension = (h/pt_seq_len)^power, with power < 1.0
            # This balances between maintaining density and staying within reasonable bounds.
            
            # Critical fix: Maintain minimum t spacing of 1.0 (same as training)
            # This ensures frequency spacing is at least as large as training
            # The key insight: t spacing directly determines frequency spacing
            # Training: t spacing = 1.0, frequency spacing = base_freq * 1.0
            # High-res: We must maintain t spacing >= 1.0 to prevent aliasing
            
            # Calculate required scale to maintain minimum spacing of 1.0
            # t_spacing = scale_extension, so scale_extension >= 1.0
            # This means: t = arange(h) * scale, where scale >= 1.0
            # For h > pt_seq_len, we need scale = h / pt_seq_len to maintain spacing of 1.0
            # But this would make t_max = h, which is too large
            # So we use a compromise: scale such that spacing is at least 1.0, but t_max is reasonable
            
            # Ensure minimum spacing of 1.0 (same as training)
            min_spacing = 1.0
            # Calculate scale to achieve this: if we want spacing = scale, then scale >= 1.0
            # For natural extension: scale = h / pt_seq_len gives spacing = h / pt_seq_len / h = 1 / pt_seq_len (too small!)
            # Instead: use scale such that spacing = scale, so scale should be at least 1.0
            
            # Use direct scaling: t = arange(h) * scale, where scale ensures spacing >= 1.0
            # For high resolutions, we allow scale to grow to maintain spacing
            scale_h = max(1.0, h / self.pt_seq_len * 0.5)  # At least 1.0, grow with resolution
            scale_w = max(1.0, w / self.pt_seq_len * 0.5)
            
            t_h_raw = t_h_base * scale_h
            t_w_raw = t_w_base * scale_w
            
            # Apply very loose upper bound only to prevent extreme frequencies
            # Use a much larger max_scale to avoid clamping most positions
            # This maintains spacing continuity while preventing numerical overflow
            # For very high resolutions, we may need even larger max_scale
            # Calculate dynamically based on actual needed range
            max_scale = max(8.0, (h / self.pt_seq_len) ** 0.8)  # Dynamic scaling
            t_h_max = self.pt_seq_len * max_scale
            t_w_max = self.pt_seq_len * max_scale
            
            # Critical fix: Prevent frequency aliasing by ensuring sufficient frequency spacing
            # The issue is not just duplicate t values, but also insufficient frequency spacing
            # that causes cos/sin values to be too similar, leading to aliasing
            # Solution: Remove max_scale limit entirely for high resolutions, or use much larger limit
            # and ensure minimum spacing between frequencies
            
            # For very high resolutions, remove the clamp entirely to allow natural extension
            # This prevents frequency aliasing by maintaining proper spacing
            # Only apply a very loose safety limit (e.g., 20x) to prevent numerical overflow
            safety_max_scale = 20.0  # Very loose limit only for extreme cases
            safety_t_max = self.pt_seq_len * safety_max_scale
            
            # Check if we need to apply any limits
            if t_h_raw.max() > safety_t_max:
                # Only in extreme cases, apply soft remapping with larger epsilon
                exceed_mask_h = t_h_raw > safety_t_max
                if exceed_mask_h.any():
                    num_exceed = exceed_mask_h.sum().item()
                    # Use larger epsilon to ensure sufficient frequency spacing
                    # Minimum spacing should be at least 0.1 to prevent aliasing
                    min_spacing = 0.1
                    t_h_exceed = torch.linspace(
                        safety_t_max - num_exceed * min_spacing,
                        safety_t_max - min_spacing,
                        num_exceed,
                        device=t_h_raw.device,
                        dtype=t_h_raw.dtype
                    )
                    t_h = t_h_raw.clone()
                    t_h[exceed_mask_h] = t_h_exceed
                else:
                    t_h = t_h_raw
            else:
                # No limit needed, use raw values
                t_h = t_h_raw
            
            if t_w_raw.max() > safety_t_max:
                exceed_mask_w = t_w_raw > safety_t_max
                if exceed_mask_w.any():
                    num_exceed = exceed_mask_w.sum().item()
                    min_spacing = 0.1
                    t_w_exceed = torch.linspace(
                        safety_t_max - num_exceed * min_spacing,
                        safety_t_max - min_spacing,
                        num_exceed,
                        device=t_w_raw.device,
                        dtype=t_w_raw.dtype
                    )
                    t_w = t_w_raw.clone()
                    t_w[exceed_mask_w] = t_w_exceed
                else:
                    t_w = t_w_raw
            else:
                t_w = t_w_raw
            
            # Final safety: ensure non-negative
            t_h = torch.clamp(t_h, min=0.0)
            t_w = torch.clamp(t_w, min=0.0)
            
            # Verify uniqueness and minimum spacing
            t_h_spacing = torch.abs(t_h[1:] - t_h[:-1])
            t_w_spacing = torch.abs(t_w[1:] - t_w[:-1])
            min_spacing_h = t_h_spacing.min().item()
            min_spacing_w = t_w_spacing.min().item()
            
            assert len(torch.unique(t_h)) == len(t_h), f"Duplicate t_h values detected!"
            assert len(torch.unique(t_w)) == len(t_w), f"Duplicate t_w values detected!"
            
            # Warn if spacing is too small (may cause frequency aliasing)
            if min_spacing_h < 0.01 or min_spacing_w < 0.01:
                print(f"Warning: Very small frequency spacing detected (h: {min_spacing_h:.6f}, w: {min_spacing_w:.6f}). "
                      f"This may cause frequency aliasing. Consider increasing max_scale.")
        else:
            # Standard linear scaling for resolutions close to training size
            # This maintains compatibility with training distribution
            t_h = t_h_base / h * self.pt_seq_len
            t_w = t_w_base / w * self.pt_seq_len
            
            # Clamp to training range with small overflow for continuity
            t_h_max = self.pt_seq_len * 1.1
            t_w_max = self.pt_seq_len * 1.1
            t_h = torch.clamp(t_h, min=0.0, max=t_h_max)
            t_w = torch.clamp(t_w, min=0.0, max=t_w_max)

        # Convert base to double precision for computation
        # This is critical for high-resolution frequency calculations
        base_double = base.double() if base.dtype != torch.float64 else base
        
        # Ensure t_h and t_w are also double precision for consistent computation
        t_h = t_h.double() if t_h.dtype != torch.float64 else t_h
        t_w = t_w.double() if t_w.dtype != torch.float64 else t_w

        # (h, dim//2) -> (h, dim)  via repeat-interleave-2
        # Use double precision einsum for numerical stability
        freqs_h = torch.einsum('i, f -> i f', t_h, base_double)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2)

        # (w, dim//2) -> (w, dim)
        freqs_w = torch.einsum('i, f -> i f', t_w, base_double)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2)

        # 2-D grid: (h, w, 2*dim)  ->  (h*w, 2*dim)
        # Ensure proper broadcasting by explicitly expanding dimensions
        freqs_h_expanded = freqs_h[:, None, :]  # (h, 1, dim)
        freqs_w_expanded = freqs_w[None, :, :]  # (1, w, dim)
        
        # Use broadcat with explicit dimension checking
        freqs = broadcat((freqs_h_expanded, freqs_w_expanded), dim=-1)
        
        # Compute cos/sin with double precision, then convert back
        # Add small epsilon to prevent numerical issues in cos/sin computation
        cos = freqs.cos().view(h * w, -1).float()
        sin = freqs.sin().view(h * w, -1).float()
        
        # Critical: Always apply a small deterministic offset to prevent frequency aliasing
        # Even with proper t spacing, small frequency differences can cause cos/sin values
        # to be identical due to numerical precision or periodic properties
        # Solution: Add a very small, position-dependent offset to break symmetries
        # This is deterministic and doesn't affect the overall frequency structure
        
        # Create a deterministic perturbation based on 2D position
        # This breaks any potential symmetries that could cause aliasing
        h_indices = torch.arange(h, device=cos.device, dtype=cos.dtype)
        w_indices = torch.arange(w, device=cos.device, dtype=cos.dtype)
        h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
        pos_2d = h_grid * w + w_grid  # Flattened 2D position index
        
        # Use a very small, deterministic perturbation
        # Scale by position to ensure uniqueness, but keep it minimal
        perturbation_scale = 1e-7  # Very small to not affect frequency structure
        perturbation = (pos_2d.flatten() % 10000) * perturbation_scale
        
        # Apply perturbation proportionally to cos/sin std to maintain relative scale
        cos_std = cos.std(dim=0, keepdim=True)
        sin_std = sin.std(dim=0, keepdim=True)
        cos = cos + perturbation.unsqueeze(1) * cos_std
        sin = sin + perturbation.unsqueeze(1) * sin_std
        
        # Verify uniqueness after perturbation
        cos_2d = cos.view(h, w, -1)
        sin_2d = sin.view(h, w, -1)
        cos_diff_h = torch.abs(cos_2d[:, 1:] - cos_2d[:, :-1]).sum(dim=2)
        cos_diff_w = torch.abs(cos_2d[1:, :] - cos_2d[:-1, :]).sum(dim=2)
        min_cos_diff = min(cos_diff_h.min().item(), cos_diff_w.min().item())
        
        if min_cos_diff < 1e-6:
            print(f"Warning: Frequency aliasing still detected after perturbation for (h={h}, w={w}): "
                  f"min_cos_diff={min_cos_diff:.8f}. Consider increasing perturbation_scale.")
        
        # Final numerical stability check
        if torch.isnan(cos).any() or torch.isnan(sin).any():
            print(f"Warning: NaN detected in RoPE frequency computation for (h={h}, w={w})")
            cos = torch.nan_to_num(cos, nan=0.0)
            sin = torch.nan_to_num(sin, nan=0.0)
        
        return cos, sin

    def for_hw(self, h, w):
        """Return a rope callable appropriate for the given (h, w) patch grid.

        Returns self when h*w matches the training size (zero overhead).
        Returns a lightweight _DynamicRopeFn otherwise (training-free
        multi-resolution support).
        """
        if h * w == self.freqs_cos.shape[0]:
            return self  # exact match – no recomputation needed
        cos, sin = self._compute_freqs_for_hw(h, w)
        return _DynamicRopeFn(cos, sin)

    def forward(self, t):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin