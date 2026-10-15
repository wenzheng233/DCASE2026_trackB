import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchaudio
except Exception:  # pragma: no cover - fallback is for non-audio envs.
    torchaudio = None


def _group_count(channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class PositionalEncoding(nn.Module):
    """
    Lightweight sinusoidal positional encoding adapted from AV-SELD's CMAF block.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model, dtype=torch.float32)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x = x.transpose(0, 1)
        x = x + self.pe[:seq_len]
        x = self.dropout(x)
        return x.transpose(0, 1)


class CMAF_Layer(nn.Module):
    """
    Reusable cross-modal attention layer adapted from AV-SELD/models/CMAF.py.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        alpha = self.pos_encoder(alpha)
        beta = self.pos_encoder(beta)
        x_sa, _ = self.self_attn(alpha, alpha, alpha)
        x_ca, _ = self.cross_attn(alpha, beta, beta)
        x = self.norm1(alpha + x_sa + x_ca)
        x_ffn = self.linear2(self.dropout1(F.relu(self.linear1(x), inplace=False)))
        x = self.norm2(x + self.dropout2(x_ffn))
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        groups = _group_count(out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.relu(self.norm1(self.conv1(x)), inplace=True)
        x = self.norm2(self.conv2(x))
        x = F.relu(x + residual, inplace=True)
        return x


class AcousticFeaturePyramid(nn.Module):
    """
    Lightweight acoustic feature pyramid aligned to Mask R-CNN FPN levels 0..3.
    """

    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        stem_groups = _group_count(64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(stem_groups, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage0 = ResidualConvBlock(64, out_channels, stride=1)
        self.stage1 = ResidualConvBlock(out_channels, out_channels, stride=2)
        self.stage2 = ResidualConvBlock(out_channels, out_channels, stride=2)
        self.stage3 = ResidualConvBlock(out_channels, out_channels, stride=2)
        self.refine = nn.ModuleDict(
            {
                str(level): nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(_group_count(out_channels), out_channels),
                    nn.ReLU(inplace=True),
                )
                for level in range(4)
            }
        )

    def forward(self, x: torch.Tensor) -> OrderedDict:
        x = self.stem(x)
        p0 = self.refine["0"](self.stage0(x))
        p1 = self.refine["1"](self.stage1(p0))
        p2 = self.refine["2"](self.stage2(p1))
        p3 = self.refine["3"](self.stage3(p2))
        return OrderedDict(
            {
                "0": p0,
                "1": p1,
                "2": p2,
                "3": p3,
            }
        )


class CrossModalFusionBlock(nn.Module):
    """
    Uses pooled CMAF context to compute a safe audio-to-visual residual gate.
    The residual scale starts small so the detector initially behaves close to
    the pretrained RGB backbone.
    """

    def __init__(
        self,
        channels: int = 256,
        token_hw: tuple[int, int] = (3, 6),
        nhead: int = 8,
        dropout: float = 0.1,
        initial_scale_logit: float = -4.0,
    ):
        super().__init__()
        self.token_hw = token_hw
        self.audio_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
        )
        self.a2v = CMAF_Layer(d_model=channels, nhead=nhead, dropout=dropout)
        self.v2a = CMAF_Layer(d_model=channels, nhead=nhead, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.scale_logit = nn.Parameter(torch.tensor(float(initial_scale_logit)))

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, self.token_hw)
        return pooled.flatten(2).transpose(1, 2)

    def forward(self, visual_feat: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        audio_aligned = self.audio_proj(audio_feat)
        visual_tokens = self._to_tokens(visual_feat)
        audio_tokens = self._to_tokens(audio_aligned)

        visual_ctx = self.a2v(visual_tokens, audio_tokens)
        audio_ctx = self.v2a(audio_tokens, visual_tokens)
        gate_in = torch.cat([visual_ctx.mean(dim=1), audio_ctx.mean(dim=1)], dim=1)
        gate = self.gate(gate_in).unsqueeze(-1).unsqueeze(-1)

        scale = torch.sigmoid(self.scale_logit)
        return visual_feat + scale * gate * audio_aligned


class CrossModalFPNFusion(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        token_hw: tuple[int, int] = (3, 6),
        nhead: int = 8,
        dropout: float = 0.1,
        initial_scale_logit: float = -4.0,
    ):
        super().__init__()
        self.level_fusion = nn.ModuleDict(
            {
                str(level): CrossModalFusionBlock(
                    channels=channels,
                    token_hw=token_hw,
                    nhead=nhead,
                    dropout=dropout,
                    initial_scale_logit=initial_scale_logit,
                )
                for level in range(4)
            }
        )

    def forward(self, visual_fpn: OrderedDict, audio_fpn: OrderedDict) -> OrderedDict:
        fused = OrderedDict()
        for key, visual_feat in visual_fpn.items():
            if key in self.level_fusion and key in audio_fpn:
                fused[key] = self.level_fusion[key](visual_feat, audio_fpn[key])
            else:
                fused[key] = visual_feat
        return fused


class ConformerSequenceEncoder(nn.Module):
    """
    AV-SELD-style Conformer wrapper for batch-first feature tokens.

    The original AV-SELD ResNet-Conformer consumes frame-wise visual features
    with shape (B, T, C). Here we reuse the same idea on compact FPN tokens so
    the detector can keep its single-frame Mask R-CNN interface.
    """

    def __init__(
        self,
        channels: int = 256,
        nhead: int = 8,
        ffn_dim: int = 1024,
        num_layers: int = 2,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.uses_torchaudio = torchaudio is not None
        if self.uses_torchaudio:
            self.encoder = torchaudio.models.Conformer(
                input_dim=channels,
                num_heads=nhead,
                ffn_dim=ffn_dim,
                num_layers=num_layers,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                dropout=dropout,
            )
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=nhead,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.uses_torchaudio:
            lengths = torch.full(
                (x.shape[0],),
                x.shape[1],
                dtype=torch.int64,
                device=x.device,
            )
            x, _ = self.encoder(x, lengths)
            return x
        return self.encoder(x)


class FPNConformerResidualBlock(nn.Module):
    """
    Converts one FPN level to compact tokens, runs a Conformer, and writes the
    context back through a small gated residual. The residual scale starts low
    to preserve the pretrained ResNet/FPN behavior at the beginning of training.
    """

    def __init__(
        self,
        channels: int = 256,
        token_hw: tuple[int, int] = (3, 6),
        nhead: int = 8,
        ffn_dim: int = 1024,
        num_layers: int = 2,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
        initial_scale_logit: float = -4.0,
    ):
        super().__init__()
        self.token_hw = token_hw
        self.input_norm = nn.LayerNorm(channels)
        self.conformer = ConformerSequenceEncoder(
            channels=channels,
            nhead=nhead,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(channels)
        self.gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.scale_logit = nn.Parameter(torch.tensor(float(initial_scale_logit)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = x.shape
        pooled = F.adaptive_avg_pool2d(x, self.token_hw)
        tokens = pooled.flatten(2).transpose(1, 2)
        encoded = self.conformer(self.input_norm(tokens))

        residual_tokens = self.output_norm(encoded - tokens)
        residual_map = residual_tokens.transpose(1, 2).reshape(
            bsz,
            channels,
            self.token_hw[0],
            self.token_hw[1],
        )
        residual_map = F.interpolate(
            residual_map,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        gate = self.gate(encoded.mean(dim=1)).view(bsz, channels, 1, 1)
        return x + torch.sigmoid(self.scale_logit) * gate * residual_map


class ResNetConformerFPNAdapter(nn.Module):
    """
    ResNet-Conformer adapter for Mask R-CNN FPN features.

    This is the lowest-risk bridge from the AV-SELD ResNet-Conformer idea into
    this instance-segmentation baseline: keep ResNet/FPN and detection heads,
    add Conformer context as a small residual on FPN levels 0..3.
    """

    def __init__(
        self,
        channels: int = 256,
        token_hw: tuple[int, int] = (3, 6),
        nhead: int = 8,
        ffn_dim: int = 1024,
        num_layers: int = 2,
        depthwise_conv_kernel_size: int = 31,
        dropout: float = 0.1,
        initial_scale_logit: float = -4.0,
    ):
        super().__init__()
        self.level_blocks = nn.ModuleDict(
            {
                str(level): FPNConformerResidualBlock(
                    channels=channels,
                    token_hw=token_hw,
                    nhead=nhead,
                    ffn_dim=ffn_dim,
                    num_layers=num_layers,
                    depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                    dropout=dropout,
                    initial_scale_logit=initial_scale_logit,
                )
                for level in range(4)
            }
        )

    def forward(self, fpn: OrderedDict) -> OrderedDict:
        out = OrderedDict()
        for key, feat in fpn.items():
            if key in self.level_blocks:
                out[key] = self.level_blocks[key](feat)
            else:
                out[key] = feat
        return out


class RoICMAFFusionHead(nn.Module):
    """
    ROI-level audio-visual fusion for detection heads.

    The module keeps the pooled ROI tensor shape unchanged so we can reuse the
    standard torchvision TwoMLP box head and predictor after fusion.
    """

    def __init__(
        self,
        channels: int = 256,
        nhead: int = 8,
        dropout: float = 0.1,
        initial_scale_logit: float = -2.0,
    ):
        super().__init__()
        self.visual_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
        )
        self.audio_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
        )
        self.a2v = CMAF_Layer(d_model=channels, nhead=nhead, dropout=dropout)
        self.v2a = CMAF_Layer(d_model=channels, nhead=nhead, dropout=dropout)
        self.fuse_norm = nn.LayerNorm(channels)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.scale_logit = nn.Parameter(torch.tensor(float(initial_scale_logit)))

    def forward(self, visual_roi: torch.Tensor, audio_roi: torch.Tensor | None) -> torch.Tensor:
        if audio_roi is None or visual_roi.numel() == 0 or audio_roi.numel() == 0:
            return visual_roi

        bsz, channels, height, width = visual_roi.shape
        visual_base = visual_roi
        visual_roi = self.visual_proj(visual_roi)
        audio_roi = self.audio_proj(audio_roi)

        visual_tokens = visual_roi.flatten(2).transpose(1, 2)
        audio_tokens = audio_roi.flatten(2).transpose(1, 2)

        visual_ctx = self.a2v(visual_tokens, audio_tokens)
        audio_ctx = self.v2a(audio_tokens, visual_tokens)

        gate_in = torch.cat([visual_tokens.mean(dim=1), audio_ctx.mean(dim=1)], dim=1)
        gate = self.gate(gate_in).view(bsz, channels, 1, 1)
        scale = torch.sigmoid(self.scale_logit)

        # Keep the pretrained visual ROI as the identity path and let audio
        # contribute only a small, gated residual.
        residual_tokens = self.fuse_norm(visual_ctx - visual_tokens)
        residual_map = residual_tokens.transpose(1, 2).reshape(bsz, channels, height, width)
        return visual_base + scale * gate * residual_map
