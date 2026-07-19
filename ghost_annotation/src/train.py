#!/usr/bin/env python3
"""
train.py

Training script for the lidar artifact detector.

Usage:
    python3 train.py \
        --dataset dataset.h5 \
        [--epochs 60] \
        [--batch-size 8] \
        [--lr 1e-3] \
        [--focal-alpha 0.99] \
        [--focal-gamma 2.0] \
        [--base-ch 16] \
        [--output-dir checkpoints]

Primary metric: AUPRC (area under the precision-recall curve). This is the
right metric for rare-event detection because it captures the full precision-
recall trade-off without being diluted by the large number of true negatives
(clean pixels) the way AUROC is. A random classifier scores AUPRC ≈ positive
rate ≈ 0.003; a useful model should be well above 0.3 after ~50 epochs.

Checkpoints:
    best.pt   the model with the highest val AUPRC seen so far
    last.pt   the model at the end of the last completed epoch
Both include model state, optimizer state, and the epoch's metrics for
easy resumption or threshold tuning later.

Input channels fed to the model (see artifact_model.py for rationale):
    Ch 0: range (m) / MAX_RANGE_M           NaN -> 0
    Ch 1: valid mask                         1 where a point exists
    Ch 2: azimuthal gradient / MAX_RANGE_M   0 at invalid neighbours
    Ch 3: cross-ring gradient / MAX_RANGE_M  0 at FOV boundary or invalid
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not found. AUPRC will not be computed. "
          "Install with: pip install scikit-learn")

from artifact_model import ArtifactNet, FocalLoss

MAX_RANGE_M = 30.0   # Ranges beyond this are clipped before normalisation.
                      # Should cover the range of your artifacts (you said
                      # close-in, ≤3m) with plenty of headroom for context.


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RangeDataset(Dataset):
    """
    Loads annotated range images from the HDF5 cache and builds the
    4-channel input tensor expected by ArtifactNet.

    Augmentation (train only):
      - Random circular azimuth roll: shifts the whole scan left or right
        by a random amount. Physically valid because azimuth wraps around.
        Critically, the same roll is applied to both the range image and
        the label map, keeping annotations aligned.
      - Light Gaussian range noise (σ=0.02m, ≈ typical lidar precision):
        adds robustness to small measurement jitter without changing the
        gradient structure the model relies on.

    No vertical flip: would invert the ring ordering, which has physical
    meaning (higher rings = higher elevation). Not a valid augmentation.
    """

    def __init__(self, h5_path: str, split: str = "train", augment: bool = True):
        self.augment = augment
        with h5py.File(h5_path, "r") as f:
            if split not in f:
                raise ValueError(
                    f"Split '{split}' not found in {h5_path}. "
                    f"Available: {list(f.keys())}"
                )
            grp = f[split]
            # Load fully into RAM -- at 14x360xfloat32, 350 scans is ~7 MB total
            self.range_imgs = grp["range_imgs"][:]   # (N, H, W) float32
            self.label_maps = grp["label_maps"][:]   # (N, H, W) int8

        if len(self.range_imgs) == 0:
            raise ValueError(f"Split '{split}' in {h5_path} contains no scans.")

    def __len__(self):
        return len(self.range_imgs)

    def __getitem__(self, idx):
        rng = self.range_imgs[idx].copy().astype(np.float32)   # (H, W)
        lbl = self.label_maps[idx].copy()                       # (H, W) int8

        # ── Augmentation ────────────────────────────────────────────────────
        if self.augment:
            # Circular azimuth roll (same shift for range and label)
            if np.random.random() < 0.5:
                shift = np.random.randint(0, rng.shape[1])
                rng   = np.roll(rng, shift, axis=1)
                lbl   = np.roll(lbl, shift, axis=1)

            # Light range noise on valid cells only
            valid_mask = np.isfinite(rng)
            rng[valid_mask] += np.random.normal(0, 0.02, size=valid_mask.sum()).astype(np.float32)

        # ── Feature engineering ─────────────────────────────────────────────
        valid = np.isfinite(rng)

        # Channel 0: normalised range
        ch_range = np.where(valid, np.clip(rng / MAX_RANGE_M, 0.0, 1.0), 0.0)

        # Channel 1: valid mask
        ch_valid = valid.astype(np.float32)

        # Channel 2: azimuthal gradient (circular wrap, so roll is safe)
        rng_right        = np.roll(rng, -1, axis=1)
        valid_az         = valid & np.isfinite(rng_right)
        ch_az_grad       = np.where(valid_az,
                                    (rng_right - rng) / MAX_RANGE_M,
                                    0.0)

        # Channel 3: cross-ring gradient (elevation is NOT circular)
        #   rng_below[i, :] = rng[i+1, :]  for i = 0..H-2
        #   rng_below[H-1, :] = NaN         (no row below the bottom ring)
        rng_below          = np.full_like(rng, np.nan)
        rng_below[:-1, :] = rng[1:, :]
        valid_el           = valid & np.isfinite(rng_below)
        ch_el_grad         = np.where(valid_el,
                                      (rng_below - rng) / MAX_RANGE_M,
                                      0.0)

        inp = np.stack([
            ch_range.astype(np.float32),
            ch_valid,
            ch_az_grad.astype(np.float32),
            ch_el_grad.astype(np.float32),
        ])   # (4, H, W)

        # ── Targets and mask ─────────────────────────────────────────────────
        target = (lbl == 1).astype(np.float32)   # (H, W)  binary
        mask   = (lbl >= 0).astype(np.float32)   # (H, W)  1=valid  0=no-point

        return (
            torch.from_numpy(inp),
            torch.from_numpy(target).unsqueeze(0),   # (1, H, W)
            torch.from_numpy(mask).unsqueeze(0),     # (1, H, W)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss  = 0.0
    all_probs   = []
    all_targets = []
    all_masks   = []

    for inp, target, mask in loader:
        inp, target, mask = inp.to(device), target.to(device), mask.to(device)
        logits      = model(inp)
        loss        = criterion(logits, target, mask)
        total_loss += loss.item() * inp.size(0)

        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_masks.append(mask.cpu().numpy())

    avg_loss = total_loss / max(len(loader.dataset), 1)

    # Flatten and restrict to valid (non-masked) pixels for PR metrics.
    probs_f   = np.concatenate([p.ravel() for p in all_probs])
    targets_f = np.concatenate([t.ravel() for t in all_targets])
    masks_f   = np.concatenate([m.ravel() for m in all_masks]).astype(bool)

    probs_v   = probs_f[masks_f]
    targets_v = targets_f[masks_f]

    n_pos = int(targets_v.sum())
    n_tot = len(targets_v)

    # AUPRC: most informative metric for rare events.
    if HAS_SKLEARN and n_pos > 0:
        auprc = float(average_precision_score(targets_v, probs_v))
    else:
        auprc = float("nan")

    # F1 at threshold 0.5 (useful as a quick sanity check during training).
    pred  = (probs_v > 0.5).astype(np.float32)
    tp    = float((pred * targets_v).sum())
    fp    = float((pred * (1 - targets_v)).sum())
    fn    = float(((1 - pred) * targets_v).sum())
    prec  = tp / (tp + fp + 1e-8)
    rec   = tp / (tp + fn + 1e-8)
    f1    = 2 * prec * rec / (prec + rec + 1e-8)

    return {
        "loss":     avg_loss,
        "auprc":    auprc,
        "prec":     prec,
        "rec":      rec,
        "f1":       f1,
        "n_pos":    n_pos,
        "n_valid":  n_tot,
        "pos_rate": n_pos / max(n_tot, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, epoch, metrics, args):
    torch.save({
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics":        metrics,
        "args":           vars(args),
    }, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",      required=True,
                    help="HDF5 file produced by build_dataset.py")
    ap.add_argument("--output-dir",   default="checkpoints")
    ap.add_argument("--epochs",       type=int,   default=60)
    ap.add_argument("--batch-size",   type=int,   default=8)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--focal-alpha",  type=float, default=0.99,
                    help="Positive class weight. Increase toward 0.999 if "
                         "val recall is too low; decrease toward 0.95 if "
                         "too many false positives.")
    ap.add_argument("--focal-gamma",  type=float, default=2.0)
    ap.add_argument("--base-ch",      type=int,   default=16,
                    help="Feature width multiplier. Increase to 32 once you "
                         "have significantly more annotated data.")
    ap.add_argument("--workers",      type=int,   default=4)
    ap.add_argument("--val-every",    type=int,   default=5,
                    help="Run validation every N epochs")
    ap.add_argument("--resume",       default=None,
                    help="Path to a checkpoint to resume from")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = RangeDataset(args.dataset, split="train", augment=True)
    val_ds   = RangeDataset(args.dataset, split="val",   augment=False)
    print(f"Train: {len(train_ds)} scans   Val: {len(val_ds)} scans")

    # Positive rates for sanity check -- should be very small (< 1%)
    with h5py.File(args.dataset, "r") as f:
        for split in ("train", "val"):
            if split in f:
                g = f[split]
                pa = g.attrs.get("n_artifact", "?")
                nc = g.attrs.get("n_clean", "?")
                if isinstance(pa, (int, np.integer)) and isinstance(nc, (int, np.integer)):
                    rate = pa / max(pa + nc, 1) * 100
                    print(f"  {split}: {pa:,} artifact pixels / {pa+nc:,} valid "
                          f"({rate:.3f}% positive rate)")

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = ArtifactNet(in_channels=4, base_ch=args.base_ch).to(device)
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ArtifactNet (base_ch={args.base_ch}, params={n_params:,})")

    start_epoch = 1
    best_auprc  = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_auprc  = ckpt.get("metrics", {}).get("auprc", 0.0)
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']}, "
              f"best AUPRC={best_auprc:.4f})")

    # ── Training ──────────────────────────────────────────────────────────────
    log_path = out_dir / "training_log.jsonl"
    print(f"\n{'Epoch':>6}  {'train_loss':>10}  {'val_loss':>9}  "
          f"{'AUPRC':>7}  {'P':>6}  {'R':>6}  {'F1':>6}  {'LR':>8}")
    print("─" * 72)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for inp, target, mask in train_dl:
            inp, target, mask = inp.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inp), target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inp.size(0)

        train_loss /= len(train_ds)
        current_lr  = scheduler.get_last_lr()[0]
        scheduler.step()

        log_row = {"epoch": epoch, "train_loss": train_loss, "lr": current_lr}

        if epoch % args.val_every == 0 or epoch == args.epochs or epoch == 1:
            val_m = evaluate(model, val_dl, criterion, device)
            log_row.update({f"val_{k}": v for k, v in val_m.items()})

            print(
                f"{epoch:>6}  {train_loss:>10.5f}  {val_m['loss']:>9.5f}  "
                f"{val_m['auprc']:>7.4f}  {val_m['prec']:>6.3f}  "
                f"{val_m['rec']:>6.3f}  {val_m['f1']:>6.3f}  "
                f"{current_lr:>8.2e}"
            )

            if val_m["auprc"] > best_auprc:
                best_auprc = val_m["auprc"]
                save_checkpoint(out_dir / "best.pt", model, optimizer,
                                epoch, val_m, args)
                print(f"         -> best.pt  (AUPRC={best_auprc:.4f})")
        else:
            print(f"{epoch:>6}  {train_loss:>10.5f}  "
                  f"{'─':>9}  {'─':>7}  {'─':>6}  {'─':>6}  {'─':>6}  "
                  f"{current_lr:>8.2e}")

        # Always save the last checkpoint for safe resumption
        save_checkpoint(out_dir / "last.pt", model, optimizer,
                        epoch, log_row, args)

        with open(log_path, "a") as f:
            f.write(json.dumps(log_row) + "\n")

    print(f"\nTraining complete. Best val AUPRC: {best_auprc:.4f}")
    print(f"Checkpoints and log saved in: {out_dir}")
    print(
        "\nNext steps:\n"
        "  1. Inspect training_log.jsonl -- if AUPRC is not improving after\n"
        "     ~20 epochs, try increasing --focal-alpha toward 0.995.\n"
        "  2. If recall is low (model misses most artifacts) but precision is\n"
        "     reasonable, increase alpha. If precision is low (too many false\n"
        "     positives), decrease alpha or raise the inference threshold.\n"
        "  3. Once the model is usable, run it on new bags to generate\n"
        "     candidate highlights for your annotation tool, then re-train on\n"
        "     the expanded dataset (active learning loop).\n"
        "  4. When you have more annotated data, increase --base-ch to 32.\n"
    )


if __name__ == "__main__":
    main()
