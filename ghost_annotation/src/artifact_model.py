"""
artifact_model.py

Lightweight fully-convolutional network for per-pixel lidar edge-blur
artifact detection on a (14 x 360) range image.

Design constraints that shape the architecture:

  Circular azimuth (W=360).
    Column 0 and column 359 are physically adjacent. Zero-padding would
    create a spurious seam at the wrap point, so all convolutions use
    circular padding on the W axis.

  No aggressive downsampling on elevation (H=14).
    14 rows collapses to nothing in two standard pooling steps. Spatial
    pooling is avoided entirely; width (azimuth) is preserved to full
    resolution throughout. Increasing dilation on the azimuth axis expands
    the receptive field without reducing spatial resolution.

  Replicate padding on elevation.
    The top and bottom rows of the range image are hard FOV boundaries with
    no data beyond them. Replicate padding (copy the edge row) is more
    honest than reflecting or zero-padding at these boundaries.

Input: (B, 4, 14, 360)
    Ch 0: range / MAX_RANGE_M, NaN -> 0
    Ch 1: valid mask (1 where a point exists, 0 for NaN / telemetry gap)
    Ch 2: azimuthal gradient (range_right - range_self) / MAX_RANGE_M,
          0 where either neighbour is NaN
    Ch 3: cross-ring gradient (range_below - range_self) / MAX_RANGE_M,
          0 where either neighbour is NaN or at elevation boundary

    Channels 2-3 give the model a head start on the two artifact signatures:
      Ghost/blur:   |az_grad| is large AND point sits between neighbours
                    (az_grad left and right have opposite signs).
      Pop-out:      point is closer than neighbours (both gradients negative).
    With only ~2000 positive examples, these inductive biases meaningfully
    reduce the amount of data needed to learn useful features.

Output: (B, 1, 14, 360) -- per-pixel artifact logit (apply sigmoid for prob).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Padding helpers
# ─────────────────────────────────────────────────────────────────────────────

def circ_replicate_pad(x: torch.Tensor, pad_w: int, pad_h: int) -> torch.Tensor:
    """
    Pads the W (azimuth) dimension circularly and the H (elevation) dimension
    with replicate padding.

    torch.nn.functional.pad does not support mixed modes in one call, so we
    do it in two steps. Order matters: circular first (on original H), then
    replicate on H after the W pad is already in place.
    """
    if pad_w > 0:
        x = torch.cat([x[..., -pad_w:], x, x[..., :pad_w]], dim=-1)
    if pad_h > 0:
        x = F.pad(x, (0, 0, pad_h, pad_h), mode="replicate")
    return x


class CircReplicateConv(nn.Module):
    """
    Conv2d with circular azimuth + replicate elevation padding, BatchNorm,
    and ReLU. Kernel and dilation are specified per-axis (H, W) so the
    azimuth receptive field can grow independently of the elevation RF.
    """
    def __init__(
        self,
        in_ch:  int,
        out_ch: int,
        kH:     int,
        kW:     int,
        dH:     int = 1,
        dW:     int = 1,
    ):
        super().__init__()
        self.pad_h = dH * (kH - 1) // 2
        self.pad_w = dW * (kW - 1) // 2
        # padding=0: we handle it ourselves above to use the mixed mode
        self.conv  = nn.Conv2d(in_ch, out_ch,
                               kernel_size=(kH, kW),
                               dilation=(dH, dW),
                               bias=False)
        self.bn    = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = circ_replicate_pad(x, self.pad_w, self.pad_h)
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ─────────────────────────────────────────────────────────────────────────────
# Network
# ─────────────────────────────────────────────────────────────────────────────

class ArtifactNet(nn.Module):
    """
    Azimuth receptive field per layer (at dW=1,2,4,8, kW=5):
      Layer 1: 5   pixels   (≈ 0.5°)
      Layer 2: 13  pixels   (≈ 1.3°)
      Layer 3: 29  pixels   (≈ 2.9°)
      Layer 4: 61  pixels   (≈ 6.1°)

    The ghost/blur signature spans at most 2-3 points at your angular
    resolution, so layers 1-2 already cover the local detection case;
    layers 3-4 provide cross-edge context that helps suppress false
    positives at clean geometry transitions.

    Total parameters with base_ch=16: ~17k. Small enough that
    overfitting on 350 scans is unlikely, large enough to be expressive.
    Increase base_ch to 32 once you have more data.
    """
    def __init__(self, in_channels: int = 4, base_ch: int = 16):
        super().__init__()
        C = base_ch
        self.enc = nn.Sequential(
            CircReplicateConv(in_channels, C,     kH=3, kW=5, dW=1),
            CircReplicateConv(C,           C * 2, kH=3, kW=5, dW=2),
            nn.Dropout2d(p=0.15),
            CircReplicateConv(C * 2,       C * 2, kH=3, kW=5, dW=4),
            nn.Dropout2d(p=0.15),
            CircReplicateConv(C * 2,       C,     kH=3, kW=5, dW=8),
        )
        # 1x1 head: no padding needed, no spatial mixing
        self.head = nn.Conv2d(C, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.enc(x))


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Sigmoid focal loss (Lin et al., 2017) for binary per-pixel classification.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where alpha_t = alpha for positives, (1-alpha) for negatives, and
    p_t = sigmoid(logit) for positives, 1 - sigmoid(logit) for negatives.

    With ~0.25% positive pixels (rough estimate from your annotation density),
    alpha=0.99 applies a ~400x weight boost to the positive class relative to
    an alpha=0.5 baseline. Start there; if val recall is too low (missing real
    artifacts), increase alpha toward 0.999. If val precision is too low (too
    many false positives), decrease it toward 0.95.

    gamma=2.0 is the standard value. It down-weights easy negatives (the vast
    majority of clean pixels the model confidently gets right) and focuses
    gradient updates on the hard examples near the decision boundary.

    The `mask` argument lets you exclude unlabeled cells (label == -1) from
    the loss entirely. This is critical: do not train on cells with no point;
    you don't know what their label would have been.
    """

    def __init__(self, alpha: float = 0.99, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits:  torch.Tensor,   # (B, 1, H, W) -- raw network output
        targets: torch.Tensor,   # (B, 1, H, W) -- float32, values in {0.0, 1.0}
        mask:    torch.Tensor,   # (B, 1, H, W) -- float32, 1=include 0=exclude
    ) -> torch.Tensor:
        p   = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Per-pixel p_t and alpha_t
        p_t     = torch.where(targets == 1, p, 1.0 - p)
        alpha_t = torch.where(
            targets == 1,
            logits.new_full(logits.shape, self.alpha),
            logits.new_full(logits.shape, 1.0 - self.alpha),
        )
        loss    = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        # Zero out masked cells and normalise by the number of valid pixels
        loss    = loss * mask
        n_valid = mask.sum().clamp(min=1.0)
        return loss.sum() / n_valid
