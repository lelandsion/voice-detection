#!/usr/bin/env python3
"""Fine-tune the speaker encoder on noisy data."""

import argparse
import sys
from pathlib import Path

import torch

try:
    from .train import TrainConfig, build_speaker_map, train, load_model
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from train import TrainConfig, build_speaker_map, train, load_model


def parse_args():
    p = argparse.ArgumentParser(description="fine-tune speaker encoder on noisy audio")
    p.add_argument("--noisy-csv", required=True, help="CSV with noisy audio paths + client_id")
    p.add_argument("--noisy-dir", required=True, help="root directory for noisy audio files")
    p.add_argument("--clean-csv", default=None, help="optionally mix in clean training data")
    p.add_argument("--clean-dir", default=None)
    p.add_argument("--baseline", default=None, help="checkpoint to fine-tune from (skips random init)")
    p.add_argument("--out", default="checkpoints/model_noisy.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=100, help="training steps per epoch")
    return p.parse_args()


def merge_speaker_maps(a, b):
    """Combine two speaker maps. Utterances from the same speaker are pooled."""
    merged = dict(a)
    for spk, paths in b.items():
        if spk in merged:
            merged[spk] = merged[spk] + paths
        else:
            merged[spk] = paths
    return merged


def main():
    args = parse_args()

    print("building speaker map from noisy data...")
    speaker_map = build_speaker_map(args.noisy_csv, args.noisy_dir)
    print(f"  {len(speaker_map)} speakers")

    if args.clean_csv and args.clean_dir:
        print("merging clean data...")
        clean_map = build_speaker_map(args.clean_csv, args.clean_dir)
        speaker_map = merge_speaker_maps(speaker_map, clean_map)
        print(f"  {len(speaker_map)} speakers after merge")

    cfg = TrainConfig(
        n_epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.lr,
    )

    # load baseline weights to fine-tune from if provided
    baseline = None
    if args.baseline:
        ckpt = Path(args.baseline)
        if not ckpt.exists():
            print(f"warning: baseline checkpoint not found at {ckpt}, starting from scratch")
        else:
            print(f"loading baseline from {ckpt}")
            baseline = load_model(ckpt, cfg)

    print(f"training for up to {cfg.n_epochs} epochs (patience={cfg.patience})...")
    model, history = train(
        speaker_map,
        cfg,
        steps_per_epoch=args.steps,
        verbose=True,
        pretrained=baseline,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, str(out))
    print(f"\nsaved to {out}")

    if history.val_loss:
        best_epoch = history.val_loss.index(min(history.val_loss)) + 1
        print(f"best val loss: {min(history.val_loss):.4f} at epoch {best_epoch}")
        print(f"trained for {len(history.loss)} epochs total")


if __name__ == "__main__":
    main()
