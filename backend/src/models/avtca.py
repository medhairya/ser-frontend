"""
src/models/avtca.py
───────────────────
Full AVT-CA architecture.

Paper sections implemented here:
  III-A  Audio & Video Feature Extraction
  III-B  Transformer Blocks for Cross-Modal Learning
  III-C  Cross-Self-Attention Mechanism
  III-D  Final Emotion Prediction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# III-A-1  AUDIO ENCODER
# ═══════════════════════════════════════════════════════════

class AudioConvBlock(nn.Module):
    """
    One 1-D convolution block:
        Conv1d → BatchNorm → ReLU → MaxPool(2)

    Input  (B, C_in, T)
    Output (B, C_out, T//2)
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioEncoder(nn.Module):
    """
    Two-stage 1-D CNN that extracts spectral-temporal audio features.

    Input  (B, n_mels, T_a)
    Output (B, T_a//4, d_model)   ← ready for transformer
    """
    def __init__(self, n_mels: int = 64, d_model: int = 256):
        super().__init__()
        mid = max(n_mels * 2, 128)
        self.block1 = AudioConvBlock(n_mels, mid)          # → (B, mid, T//2)
        self.block2 = AudioConvBlock(mid, d_model)          # → (B, d_model, T//4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, T_a)
        x = self.block1(x)         # (B, mid, T//2)
        x = self.block2(x)         # (B, d_model, T//4)
        return x.transpose(1, 2)   # (B, T//4, d_model)


# ═══════════════════════════════════════════════════════════
# III-A-2  VIDEO ENCODER  (channel attn + spatial attn +
#           local feature extractor + inverted residual)
# ═══════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """
    SE-style channel recalibration.
    Ṽ_c = σ(W_fc2 · σ(W_fc1 · AvgPool(V̄) + b_fc1) + b_fc2)

    Input / Output  (B, C, H, W)
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(channels, mid)
        self.fc2  = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        w = self.pool(x).view(B, C)           # (B, C)
        w = F.relu(self.fc1(w))               # (B, mid)
        w = torch.sigmoid(self.fc2(w))        # (B, C)
        return x * w.view(B, C, 1, 1)         # broadcast multiply


class SpatialAttention(nn.Module):
    """
    Spatial attention via 1×1 conv + softmax over H×W.
    Ṽ_s = softmax(Conv2D(V̄, W_s, b_s))

    Input / Output  (B, C, H, W)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        z = self.conv(x)                                   # (B, 1, H, W)
        attn = F.softmax(z.view(B, 1, -1), dim=-1)        # over HW
        attn = attn.view(B, 1, H, W)
        return x * attn


class LocalFeatureExtractor(nn.Module):
    """
    Divides the feature map into an n_grid × n_grid patch grid,
    applies a conv to each patch, stitches back, and adds as
    a residual to the input.

    Input / Output  (B, C, H, W)
    """
    def __init__(self, channels: int, n_grid: int = 2):
        super().__init__()
        self.n_grid = n_grid
        self.patch_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        g = self.n_grid
        ph = H // g
        pw = W // g

        rows = []
        for i in range(g):
            cols = []
            for j in range(g):
                patch = x[:, :, i * ph:(i+1) * ph, j * pw:(j+1) * pw]
                cols.append(self.patch_conv(patch))
            rows.append(torch.cat(cols, dim=3))           # join along W
        local_out = torch.cat(rows, dim=2)                # join along H

        return x + local_out                              # residual


class InvertedResidualBlock(nn.Module):
    """
    MobileNetV2-style inverted residual with depthwise-separable conv.
    Reduces parameters from C×C×k² → C×k² + C×C  (depthwise + pointwise).

    Input / Output  (B, C, H, W)
    """
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        mid = channels * expansion
        self.block = nn.Sequential(
            # Pointwise expansion
            nn.Conv2d(channels, mid, 1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            # Depthwise 3×3
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            # Pointwise projection
            nn.Conv2d(mid, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class VideoFrameEncoder(nn.Module):
    """
    Encodes a single video frame to a d_model-dimensional vector.

    Input  (B, 3, H, W)
    Output (B, d_model)

    Pipeline:
        Conv2d + MaxPool
        → Channel Attention (Ṽ_c)
        → Spatial Attention (Ṽ_s)
        → Combined (Ṽ = Ṽ_c ⊙ Ṽ_s)
        → Local Feature Extractor  (V̂ = Ṽ + ψ(o_local))
        → 2× Inverted Residual Block
        → 2nd Conv block
        → Global Avg Pool → Linear projection
    """
    def __init__(self, d_model: int = 256, cnn_ch: int = 64):
        super().__init__()
        # Initial conv + pool
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, cnn_ch, 3, padding=1),
            nn.BatchNorm2d(cnn_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Attention modules
        self.ch_attn   = ChannelAttention(cnn_ch)
        self.sp_attn   = SpatialAttention(cnn_ch)
        # Local feature extractor
        self.local_ext = LocalFeatureExtractor(cnn_ch, n_grid=2)
        # Inverted residual blocks (2×)
        self.inv_res1  = InvertedResidualBlock(cnn_ch)
        self.inv_res2  = InvertedResidualBlock(cnn_ch)
        # Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(cnn_ch, cnn_ch * 2, 3, padding=1),
            nn.BatchNorm2d(cnn_ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Projection head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(cnn_ch * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        f = self.init_conv(x)                     # (B, cnn_ch, H//2, W//2)

        # ── Combined attention (Ṽ = Ṽ_c ⊙ Ṽ_s) ──
        f_c = self.ch_attn(f)                     # channel-refined
        f_s = self.sp_attn(f)                     # spatially-refined
        f   = f_c * f_s                           # element-wise combine

        # ── Local feature + residual ───────────────
        f = self.local_ext(f)

        # ── Inverted residual blocks ───────────────
        f = self.inv_res1(f)
        f = self.inv_res2(f)

        # ── Deepen with second conv block ──────────
        f = self.conv2(f)                         # (B, cnn_ch*2, H//4, W//4)

        # ── Global pool + projection ───────────────
        f = self.pool(f).flatten(1)               # (B, cnn_ch*2)
        return self.proj(f)                       # (B, d_model)


class VideoEncoder(nn.Module):
    """
    Applies VideoFrameEncoder to every frame in a clip.

    Input  (B, T_v, 3, H, W)
    Output (B, T_v, d_model)
    """
    def __init__(self, d_model: int = 256, cnn_ch: int = 64):
        super().__init__()
        self.frame_enc = VideoFrameEncoder(d_model, cnn_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)              # flatten batch+time
        f = self.frame_enc(x)                    # (B*T, d_model)
        return f.view(B, T, -1)                  # (B, T, d_model)


# ═══════════════════════════════════════════════════════════
# III-B  TRANSFORMER BLOCKS FOR CROSS-MODAL LEARNING
# ═══════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Standard transformer encoder block (pre-LN variant).

    Self-attention:
        o_attn  = softmax(Q K^T / √d_k) V
        o_trans = o_attn + x
        o_norm  = LayerNorm(o_trans)

    Feed-forward (GELU activation, Dropout):
        o_ffn  = Dropout(GELU(o_norm W1 + b1)) W2 + b2
        output = LayerNorm(o_norm + o_ffn)

    Input / Output  (B, T, d_model)
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Pre-LN self-attention
        x2, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=key_padding_mask,
        )
        x = x + self.drop(x2)
        # Pre-LN FFN
        x = x + self.ffn(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════
# III-C  CROSS-SELF-ATTENTION MECHANISM
# ═══════════════════════════════════════════════════════════

class CrossAttentionBlock(nn.Module):
    """
    Bidirectional cross-attention (A→V and V→A simultaneously).

        o_AV = softmax(Q_a K_v^T / √d_k) V_v
        o_VA = softmax(Q_v K_a^T / √d_k) V_a

        ỡ_audio = LayerNorm(audio + o_AV)
        ỡ_video = LayerNorm(video + o_VA)

    Args:
        d_model   : feature dimension
        num_heads : number of attention heads
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn_av = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_va = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ):
        # Audio queries attend to Video keys/values  →  o_AV
        av, _ = self.attn_av(audio, video, video)
        # Video queries attend to Audio keys/values  →  o_VA
        va, _ = self.attn_va(video, audio, audio)

        audio_out = self.norm_a(audio + self.drop(av))
        video_out = self.norm_v(video + self.drop(va))
        return audio_out, video_out


# ═══════════════════════════════════════════════════════════
# FULL MODEL:  AVT-CA
# ═══════════════════════════════════════════════════════════

class AVTCA(nn.Module):
    """
    Audio-Video Transformer with Cross Attention (AVT-CA).

    End-to-end pipeline (Algorithm 1 in paper):
        1. Audio feature extraction  (AudioEncoder)
        2. Video feature extraction  (VideoEncoder)
        3. Intermediate transformer fusion  (IT-4 in ablation)
        4. Cross-self-attention  (CT-4 in ablation)
        5. Final cross-attention
        6. Max-pool + element-wise add + FC + softmax

    Args:
        num_classes            : number of emotion classes
        n_mels                 : mel spectrogram bins
        d_model                : transformer dimension
        num_heads              : attention heads (IT-4 / CT-4 → 4)
        num_transformer_layers : transformer depth per branch
        ffn_dim                : FFN hidden size (4 × d_model)
        dropout                : dropout rate
        cnn_ch                 : base CNN channels for VideoEncoder
    """

    def __init__(
        self,
        num_classes: int = 8,
        n_mels: int = 64,
        d_model: int = 256,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        cnn_ch: int = 64,
    ):
        super().__init__()
        self.d_model = d_model

        # ── Feature encoders ─────────────────────────────────
        self.audio_enc = AudioEncoder(n_mels=n_mels, d_model=d_model)
        self.video_enc = VideoEncoder(d_model=d_model, cnn_ch=cnn_ch)

        # ── Intermediate transformer fusion (IT-4) ────────────
        self.audio_transformer = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_transformer_layers)
        ])
        self.video_transformer = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_transformer_layers)
        ])

        # ── Cross-self attention (CT-4) ───────────────────────
        self.cross_self_attn = CrossAttentionBlock(d_model, num_heads, dropout)

        # ── Final cross attention ─────────────────────────────
        self.final_cross_attn = CrossAttentionBlock(d_model, num_heads, dropout)

        # ── Classification head ───────────────────────────────
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    # ── Weight initialisation ────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward pass ─────────────────────────────────────────

    def forward(
        self,
        audio: torch.Tensor,   # (B, n_mels, T_a)
        video: torch.Tensor,   # (B, T_v, 3, H, W)
    ) -> torch.Tensor:         # (B, num_classes)
        """
        Returns raw logits (no softmax). Use CrossEntropyLoss during training.
        """
        # ── Step 1 & 2: Feature extraction ───────────────────
        a = self.audio_enc(audio)   # (B, T_a', d_model)
        v = self.video_enc(video)   # (B, T_v,  d_model)

        # ── Step 3: Intermediate transformer fusion ───────────
        for layer in self.audio_transformer:
            a = layer(a)
        for layer in self.video_transformer:
            v = layer(v)

        # ── Step 4: Cross-self attention ──────────────────────
        a, v = self.cross_self_attn(a, v)

        # ── Step 5: Final cross attention ─────────────────────
        a, v = self.final_cross_attn(a, v)

        # ── Step 6: Max-pool + fuse + classify ───────────────
        a_pool = a.max(dim=1).values   # (B, d_model)
        v_pool = v.max(dim=1).values   # (B, d_model)
        fused  = a_pool + v_pool       # element-wise add (paper eq.)

        return self.classifier(fused)  # (B, num_classes)

    # ── Utility ──────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
