#!/usr/bin/env python3
"""Train or fine-tune the speaker encoder on noisy data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    from .train import TrainConfig, build_speaker_map, load_model, resolve_device, save_model, train
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from train import TrainConfig, build_speaker_map, load_model, resolve_device, save_model, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train speaker encoder on noisy audio")
    parser.add_argument(
        "--noisy-csv",
        type=Path,
        default=Path("data/processed/commonvoice_noisy.csv"),
        help="CSV with noisy audio paths and client_id.",
    )
    parser.add_argument(
        "--noisy-dir",
        type=Path,
        default=Path("data/processed/noisy/audio_files"),
        help="Root directory for noisy audio files listed in --noisy-csv.",
    )
    parser.add_argument("--clean-csv", type=Path, default=None)
    parser.add_argument("--clean-dir", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None, help="Optional checkpoint to fine-tune from.")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/model_noisy.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=50, help="Training steps per epoch")
    parser.add_argument("--n-speakers", type=int, default=8)
    parser.add_argument("--n-utterances", type=int, default=5)
    parser.add_argument("--min-utterances", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. 'auto' prefers CUDA when available.",
    )
    return parser.parse_args()


def merge_speaker_maps(
    first: dict[str, list[Path]],
    second: dict[str, list[Path]],
) -> dict[str, list[Path]]:
    merged = dict(first)
    for speaker, paths in second.items():
        if speaker in merged:
            merged[speaker] = merged[speaker] + paths
        else:
            merged[speaker] = paths
    return merged


def main() -> None:
    args = parse_args()

    if not args.noisy_csv.exists():
        raise FileNotFoundError(f"Noisy CSV not found: {args.noisy_csv}")
    if not args.noisy_dir.exists():
        raise FileNotFoundError(f"Noisy audio dir not found: {args.noisy_dir}")

    print("Building speaker map from noisy data...")
    speaker_map = build_speaker_map(
        csv_path=args.noisy_csv,
        audio_dir=args.noisy_dir,
        min_utterances=args.min_utterances,
    )
    print(f"Noisy speakers: {len(speaker_map)}")

    if args.clean_csv and args.clean_dir:
        print("Merging optional clean training data...")
        clean_map = build_speaker_map(
            csv_path=args.clean_csv,
            audio_dir=args.clean_dir,
            min_utterances=args.min_utterances,
        )
        speaker_map = merge_speaker_maps(speaker_map, clean_map)
        print(f"Speakers after merge: {len(speaker_map)}")

    if len(speaker_map) < args.n_speakers:
        raise ValueError(
            f"Need at least {args.n_speakers} speakers after filtering, found {len(speaker_map)}."
        )

    cfg = TrainConfig(
        n_epochs=args.epochs,
        learning_rate=args.lr,
        n_speakers=args.n_speakers,
        n_utterances=args.n_utterances,
        min_utterances=args.min_utterances,
        random_state=args.seed,
    )

    if args.baseline is not None:
        if args.baseline.exists():
            print(
                "Warning: baseline loading is currently disabled because src.train.train() "
                "does not support a pretrained parameter in this branch. Training from scratch."
            )
        else:
            print(f"Warning: baseline checkpoint missing at {args.baseline}; training from scratch")

    print(f"Training noisy model for up to {cfg.n_epochs} epochs...")
    model, history = train(
        speaker_map=speaker_map,
        config=cfg,
        steps_per_epoch=args.steps,
        device=args.device,
        verbose=True,
    )

    resolved_device = resolve_device(args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.out)

    summary = {
        "status": "trained",
        "output_path": str(args.out),
        "noisy_csv": str(args.noisy_csv),
        "noisy_dir": str(args.noisy_dir),
        "epochs_requested": args.epochs,
        "epochs_ran": len(history.loss),
        "steps_per_epoch": args.steps,
        "final_train_loss": history.loss[-1] if history.loss else None,
        "device": str(resolved_device),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
