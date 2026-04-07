from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .train import TrainConfig, build_speaker_map, save_model, train
except ImportError:
    from train import TrainConfig, build_speaker_map, save_model, train


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a GE2E speaker embedding checkpoint for prompt verification."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/commonvoice-v24_en-AU 3/commonvoice-v24_en-AU.csv"),
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/commonvoice-v24_en-AU 3/audio_files"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("checkpoints/ge2e_prompt_verifier.pt"),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=25)
    parser.add_argument("--n-speakers", type=int, default=8)
    parser.add_argument("--n-utterances", type=int, default=5)
    parser.add_argument("--min-utterances", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. 'auto' prefers CUDA when available.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    speaker_map = build_speaker_map(
        csv_path=args.csv_path,
        audio_dir=args.audio_dir,
        min_utterances=args.min_utterances,
    )
    if len(speaker_map) < args.n_speakers:
        raise ValueError(
            f"Need at least {args.n_speakers} speakers after filtering, found {len(speaker_map)}."
        )

    config = TrainConfig(
        n_epochs=args.epochs,
        n_speakers=args.n_speakers,
        n_utterances=args.n_utterances,
        min_utterances=args.min_utterances,
        learning_rate=args.learning_rate,
        random_state=args.random_state,
    )

    model, history = train(
        speaker_map=speaker_map,
        config=config,
        steps_per_epoch=args.steps_per_epoch,
        device=args.device,
        verbose=True,
    )
    save_model(model, args.output_path)

    print(
        json.dumps(
            {
                "status": "trained",
                "output_path": str(args.output_path),
                "num_speakers": len(speaker_map),
                "num_utterances": sum(len(v) for v in speaker_map.values()),
                "epochs": args.epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "final_loss": history.loss[-1] if history.loss else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
