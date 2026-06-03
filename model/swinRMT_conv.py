# -*- coding: utf-8 -*-
"""
UNet + Swin-Style Retention (RetNet)
- 输入:  x: Tensor [B, in_ch, H, W]
- 输出:  logits: Tensor [B, num_classes, H, W]（内部不做 sigmoid/softmax）
- 兼容：保留原先并行式 RetNet 插入点；新增 ret_replace_at 用 RetNet 替换 block 内第二个卷积
"""

from __future__ import annotations
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 可选：用于 profile（没有安装 fvcore 也不影响训练）
try:
    from fvcore.nn import parameter_count, FlopCountAnalysis
except Exception:  # pragma: no cover
    parameter_count = FlopCountAnalysis = None


# =========================
# 工具 & 基础
# =========================
class GroupNorm1dPerHead(nn.Module):
    def __init__(self, num_heads: int, channels_per_head: int, eps: float = 1e-5):
        super().__init__()
        self.gn = nn.GroupNorm(
            num_groups=num_heads,
            num_channels=num_heads * channels_per_head,
            eps=eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, Cph]
        B, H, T, Cph = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(B, H * Cph, T)
        x = self.gn(x)
        x = x.view(B, H, Cph, T).permute(0, 1, 3, 2).contiguous()
        return x


def window_partition(x: torch.Tensor, window_size: int):
    # x: [B, C, H, W]
    B, C, H, W = x.shape
    assert H % window_size == 0 and W % window_size == 0, "H/W must be divisible by window_size"
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    # -> [B, C, nH, Wh, nW, Ww] -> [B*nW, C, Wh, Ww]
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return x


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    # windows: [B*nW, C, Wh, Ww]
    BnW, C, Wh, Ww = windows.shape
    nH, nW = H // window_size, W // window_size
    B = BnW // (nH * nW)
    x = windows.view(B, nH, nW, C, Wh, Ww).permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, H, W)
    return x


def build_swin_attn_mask(H: int, W: int, window_size: int, shift_size: int, device):
    """Swin 的跨窗屏蔽 mask（仅 shift>0 时需要）"""
    if shift_size == 0:
        return None

    img_mask = torch.zeros((1, 1, H, W), device=device)  # [1,1,H,W]
    cnt = 0
    h_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    w_slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    for h in h_slices:
        for w in w_slices:
            img_mask[:, :, h, w] = cnt
            cnt += 1

    rolled_mask = torch.roll(img_mask, shifts=(-shift_size, -shift_size), dims=(2, 3))
    mask_windows = window_partition(rolled_mask, window_size)  # [nW,1,Wh,Ww]
    mask_windows = mask_windows.view(-1, window_size * window_size)  # [nW, Tw]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, Tw, Tw]
    attn_mask = attn_mask.ne(0).to(torch.float32)  # 不同标号=1，同标号=0
    return attn_mask.masked_fill(attn_mask > 0, float("-inf"))  # [nW, Tw, Tw]


# =========================
# 窗口 RetNet（自动 padding）
# =========================
class MultiScaleRetention2DWindowed(nn.Module):
    """
    窗口 RetNet（可选 shift），自动右/下 padding 到 window_size 倍数；结束后裁剪回原尺寸。
    """
    def __init__(
        self,
        in_ch: int,
        num_heads: int = 4,
        qk_dim: int = 64,
        window_size: int = 8,
        shift_size: int = 0,
        downsample: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert in_ch % num_heads == 0, "in_ch must be divisible by num_heads"
        assert 0 <= shift_size < window_size, "shift_size must be in [0, window_size)"
        self.h = num_heads
        self.qk_dim = qk_dim
        self.v_dim = 2 * qk_dim
        self.downsample = downsample
        self.window_size = window_size
        self.shift_size = shift_size

        # token-wise 线性（1×1 Conv1d）
        self.w_q = nn.Conv1d(in_ch, self.h * self.qk_dim, 1, bias=False)
        self.w_k = nn.Conv1d(in_ch, self.h * self.qk_dim, 1, bias=False)
        self.w_v = nn.Conv1d(in_ch, self.h * self.v_dim, 1, bias=False)
        self.w_g = nn.Conv1d(in_ch, in_ch, 1, bias=True)
        self.w_o = nn.Conv1d(self.h * self.v_dim, in_ch, 1, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm_per_head = GroupNorm1dPerHead(num_heads, self.v_dim)

        # 衰减参数（每头）
        i = torch.arange(num_heads, dtype=torch.float32)
        gamma = 1.0 - torch.pow(2.0, -(5.0 + i))
        self.register_buffer("gamma", gamma.view(1, num_heads, 1, 1))
        self.theta = nn.Parameter(torch.linspace(0.0, 0.5, steps=num_heads).view(1, num_heads, 1, 1))

        self.pool = nn.AvgPool2d(downsample, downsample) if downsample > 1 else nn.Identity()
        self.upsample = nn.Upsample(scale_factor=downsample, mode="nearest") if downsample > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        xd = self.pool(x)                 # [B,C,h,w]
        h0, w0 = xd.shape[2], xd.shape[3]

        # 自动 pad
        ws = self.window_size
        pad_h = (ws - (h0 % ws)) % ws
        pad_w = (ws - (w0 % ws)) % ws
        if pad_h > 0 or pad_w > 0:
            xd = F.pad(xd, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        h, w = xd.shape[2], xd.shape[3]

        # shift
        x_shift = torch.roll(xd, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3)) \
                  if self.shift_size > 0 else xd

        attn_mask = build_swin_attn_mask(h, w, self.window_size, self.shift_size, device=xd.device)

        # 分窗
        x_win = window_partition(x_shift, self.window_size)      # [BnW,C,Wh,Ww]
        Wh = Ww = self.window_size
        Tw = Wh * Ww
        BnW = x_win.shape[0]
        xs = x_win.view(BnW, C, Tw)                              # [BnW,C,Tw]

        # QKV + gating
        q = self.w_q(xs); k = self.w_k(xs); v = self.w_v(xs)
        g = torch.sigmoid(self.w_g(xs)); y = xs * g

        def to_heads(t: torch.Tensor, d: int) -> torch.Tensor:   # -> [BnW,H,Tw,d]
            return t.view(BnW, self.h, d, Tw).permute(0, 1, 3, 2).contiguous()

        q = to_heads(q, self.qk_dim)
        k = to_heads(k, self.qk_dim)
        v = to_heads(v, self.v_dim)

        # 打分 + mask
        scale = 1.0 / float(np.sqrt(self.qk_dim))
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale    # [BnW,H,Tw,Tw]
        if attn_mask is not None:
            nW = (h // Wh) * (w // Ww)
            expanded = attn_mask.unsqueeze(0).repeat(BnW // nW, 1, 1, 1).unsqueeze(1)
            scores = scores + expanded

        # 因果 + 距离衰减
        idx = torch.arange(Tw, device=x.device)
        n = idx.view(1, 1, Tw, 1); m = idx.view(1, 1, 1, Tw)
        causal = (m <= n).float()
        dist = (n - m).clamp(min=0).float()
        D = torch.pow(self.gamma, dist) * causal                 # [1,H,Tw,Tw]
        D = D / torch.clamp(D.sum(dim=-1, keepdim=True).sqrt(), min=1.0)

        R = scores * D
        R = R / torch.clamp(R.abs().sum(dim=-1, keepdim=True), min=1.0)

        out = torch.matmul(R, v)                                 # [BnW,H,Tw,Vd]
        out = self.norm_per_head(out)
        out = out.permute(0, 1, 3, 2).contiguous().view(BnW, self.h * self.v_dim, Tw)

        out = self.w_o(out)
        out = self.dropout(out)
        out = out + y

        # 拼回 & 反向 shift & 裁剪
        out = out.view(BnW, C, Wh, Ww)
        x_merged = window_reverse(out, self.window_size, h, w)
        if self.shift_size > 0:
            x_merged = torch.roll(x_merged, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        if pad_h > 0 or pad_w > 0:
            x_merged = x_merged[:, :, :h0, :w0]

        return self.upsample(x_merged)


# =========================
# “RetNet 卷积”：用于替代 3×3 Conv
# =========================
class RetentionConv2d(nn.Module):
    """
    Drop-in 替代 3x3 Conv 的 RetNet 卷积：
    - 输入/输出: [B, C, H, W] -> [B, C, H, W]
    - 使用窗口 RetNet 做局部上下文，支持 shift 与自动 padding
    """
    def __init__(self, channels: int,
                 num_heads: int = 4, qk_dim: int = 64,
                 window_size: int = 8, shift_size: int = 0,
                 dropout: float = 0.0):
        super().__init__()
        self.ret = MultiScaleRetention2DWindowed(
            in_ch=channels, num_heads=num_heads, qk_dim=qk_dim,
            window_size=window_size, shift_size=shift_size,
            downsample=1, dropout=dropout
        )
        self.norm = nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ret(x)
        y = self.norm(y)
        y = self.act(y)
        return y


# =========================
# UNet Blocks
# =========================
class ConvINReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks: int = 3, stride: int = 1, pad: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, pad, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.relu(self.norm(self.conv(x))))


class UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvINReLU(in_ch, out_ch, dropout=dropout)
        self.conv2 = ConvINReLU(out_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class UNetBlockRet(nn.Module):
    """
    第一层: 3x3 Conv + IN + ReLU
    第二层: RetentionConv2d (替代第二个 3x3 Conv)
    """
    def __init__(self, in_ch: int, out_ch: int,
                 dropout: float = 0.0,
                 ret_heads: int = 4, ret_qkdim: int = 64,
                 window: int = 8, shift: int = 0):
        super().__init__()
        self.conv1 = ConvINReLU(in_ch, out_ch, dropout=dropout)
        self.conv2 = RetentionConv2d(
            out_ch, num_heads=ret_heads, qk_dim=ret_qkdim,
            window_size=window, shift_size=shift, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


# =========================
# U-Net 主体（支持两类 RetNet：并行插入 + 卷积替换）
# =========================
class UNetRetNet(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        num_classes: int = 3,
        base_ch: int = 32,
        retention_heads: int = 4,
        retention_dim: int = 64,
        retention_at: str = "bottleneck",   # ["bottleneck","enc3","enc4","all"]
        dropout: float = 0.0,
        retention_window: int = 8,
        retention_shift: int = 0,
        # 新增：选择在哪些 block 用 RetNet 替代第 2 个 3×3 卷积（不传则不替换）
        ret_replace_at: Optional[List[str]] = None,  # e.g. ['bottleneck','enc4','dec3']
    ) -> None:
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]

        # 小助手：按标签创建普通或 RetNet 版 block
        replace = set(ret_replace_at or [])

        def make_block(tag: str, in_c: int, out_c: int):
            if tag in replace:
                return UNetBlockRet(
                    in_c, out_c, dropout=dropout,
                    ret_heads=retention_heads, ret_qkdim=retention_dim,
                    window=retention_window, shift=retention_shift
                )
            return UNetBlock(in_c, out_c, dropout)

        # Encoder
        self.enc1 = make_block('enc1', in_ch, ch[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = make_block('enc2', ch[0], ch[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = make_block('enc3', ch[1], ch[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = make_block('enc4', ch[2], ch[3])
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = make_block('bottleneck', ch[3], ch[4])

        # 并行式 Retention（与你原代码一致）
        def need(p: str) -> bool:
            return retention_at in ["all", p]

        self.ret_enc3: Optional[MultiScaleRetention2DWindowed] = (
            MultiScaleRetention2DWindowed(
                ch[2], retention_heads, retention_dim,
                window_size=retention_window, shift_size=retention_shift,
                downsample=2, dropout=dropout
            ) if need("enc3") else None
        )
        self.ret_enc4: Optional[MultiScaleRetention2DWindowed] = (
            MultiScaleRetention2DWindowed(
                ch[3], retention_heads, retention_dim,
                window_size=retention_window, shift_size=retention_shift,
                downsample=2, dropout=dropout
            ) if need("enc4") else None
        )
        self.ret_bott: Optional[MultiScaleRetention2DWindowed] = (
            MultiScaleRetention2DWindowed(
                ch[4], retention_heads, retention_dim,
                window_size=retention_window, shift_size=retention_shift,
                downsample=1, dropout=dropout
            ) if need("bottleneck") else None
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(ch[4], ch[3], 2, 2)
        self.dec4 = make_block('dec4', ch[4], ch[3])
        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, 2)
        self.dec3 = make_block('dec3', ch[3], ch[2])
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, 2)
        self.dec2 = make_block('dec2', ch[2], ch[1])
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, 2)
        self.dec1 = make_block('dec1', ch[1], ch[0])

        # Head
        self.out_conv = nn.Conv2d(ch[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x);  p1 = self.pool1(e1)
        e2 = self.enc2(p1); p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        if self.ret_enc3 is not None:
            e3 = e3 + self.ret_enc3(e3)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        if self.ret_enc4 is not None:
            e4 = e4 + self.ret_enc4(e4)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)
        if self.ret_bott is not None:
            b = b + self.ret_bott(b)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)


# =========================
# 工厂 & 可选 profile
# =========================
def build_unet_retnet(
    in_ch: int = 1,
    num_classes: int = 3,
    base_ch: int = 32,
    retention_heads: int = 4,
    retention_dim: int = 64,
    retention_at: str = "bottleneck",
    dropout: float = 0.0,
    retention_window: int = 8,
    retention_shift: int = 0,
    ret_replace_at: Optional[List[str]] = None,
) -> UNetRetNet:
    return UNetRetNet(
        in_ch=in_ch,
        num_classes=num_classes,
        base_ch=base_ch,
        retention_heads=retention_heads,
        retention_dim=retention_dim,
        retention_at=retention_at,
        dropout=dropout,
        retention_window=retention_window,
        retention_shift=retention_shift,
        ret_replace_at=ret_replace_at,
    )


# 可选：模型复杂度统计（需要 fvcore）
def profile_model(model: nn.Module,
                  input_shape=(3, 160, 160),
                  device="cuda",
                  batch_size: int = 1):
    if parameter_count is None or FlopCountAnalysis is None:
        print("[profile] fvcore 未安装，跳过 FLOPs 统计。")
        return None, None

    model = model.to(device).eval()
    dummy = torch.randn(batch_size, *input_shape, device=device)
    params = parameter_count(model)[""]
    with torch.no_grad():
        _ = model(dummy)
    flops_ana = FlopCountAnalysis(model, dummy)
    flops_total = flops_ana.total()

    print("==== Model Profile ====")
    print(f"Input: (B={batch_size}, C={input_shape[0]}, H={input_shape[1]}, W={input_shape[2]})")
    print(f"Params: {params/1e6:.2f}M ({params:,})")
    print(f"FLOPs: {flops_total/1e9:.3f}G ({int(flops_total):,}) per forward")
    return params, flops_total


if __name__ == '__main__':
    # 示例：保持与原用法兼容；若要 Swin 平移窗口，设 retention_shift=retention_window//2
    model = UNetRetNet(
        in_ch=4, num_classes=3, base_ch=32,
        retention_heads=4, retention_dim=64,
        retention_at='bottleneck', dropout=0.0,
        retention_window=8, retention_shift=0,
        ret_replace_at=['bottleneck', 'enc4']   # 示例：把 bottleneck 和 enc4 的 block 第二层改为 RetNet
    ).to('cpu')

    profile_model(model, input_shape=(4, 160, 160), device='cpu', batch_size=1)
