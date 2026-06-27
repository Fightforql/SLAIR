import torch
import torch.nn as nn


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_out = None
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = self.norm1(x_in)
        x = x * torch.sigmoid(x)
        x = self.conv1(x)
        if self.conv_out is not None:
            x_in = self.conv_out(x_in)
        return x + x_in


class Fuse_sft_block_RRDB(nn.Module):
    def __init__(self, in_ch, out_ch, layer, num_block=1, num_grow_ch=16):
        super().__init__()
        if layer == 0:
            ratio = 1
        elif layer <= 1:
            ratio = 2
        else:
            ratio = 4

        self.in_ch = in_ch

        self.enc_denoise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // ratio, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch // ratio, in_ch, kernel_size=3, stride=1, padding=1),
        )

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6, affine=False)

        self.enc_gate = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

        self.cross_attention = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6),
            nn.Sigmoid(),
        )

        self.adaln_proj = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, stride=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0),
        )

        self.encode_enc_1 = ResBlock(2 * in_ch, in_ch)
        self.encode_enc_3 = ResBlock(in_ch, out_ch)

        self._init_weights()

    def _init_weights(self):
        """Initialize newly added fusion parameters with Xavier initialization."""
        for m in self.enc_denoise.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        last_conv = None
        for m in reversed(list(self.enc_gate.modules())):
            if isinstance(m, nn.Conv2d):
                last_conv = m
                break

        for m in self.enc_gate.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if m is last_conv:
                        nn.init.constant_(m.bias, -5.0)
                    else:
                        nn.init.zeros_(m.bias)

        for m in self.cross_attention.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.adaln_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, enc_feat, dec_feat, w=1.0):
        enc_feat_denoised = self.enc_denoise(enc_feat)
        enc_feat_transformed = self.conv(enc_feat_denoised)

        gate_input = torch.cat([enc_feat_transformed, dec_feat], dim=1)
        enc_gate = self.enc_gate(gate_input)
        enc_feat_gated = enc_feat_transformed * enc_gate

        combined_for_attn = torch.cat([enc_feat_gated, dec_feat], dim=1)
        attention_map = self.cross_attention(combined_for_attn)

        enc_feat_attended = enc_feat_gated * attention_map
        dec_feat_enhanced = dec_feat * (1.0 + (1.0 - attention_map) * 0.3)

        combined_feat_2d = torch.cat([enc_feat_attended, dec_feat_enhanced], dim=1)
        adaln_params = self.adaln_proj(combined_feat_2d)
        gamma, beta = adaln_params.chunk(2, dim=1)

        enc_feat_modulated = self.norm(enc_feat_attended) * (1 + gamma) + beta
        combined_feat = torch.cat([enc_feat_modulated, dec_feat_enhanced], dim=1)

        out = self.encode_enc_1(combined_feat)
        out = self.encode_enc_3(out)

        return w * out + dec_feat
