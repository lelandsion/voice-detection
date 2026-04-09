"""
Real-world evaluation: collect genuine/imposter scores from live mic.
Each person records N utterances, scores are saved to CSV for analysis.

Usage:
    source .venv/bin/activate && python src/eval_realworld.py

After collecting data from multiple people, run analysis:
    python src/eval_realworld.py --analyze
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from audio import MicrophoneRecorder
from train import TrainConfig, LSTMEmbedder, embed, resolve_device
from prompt_verification import load_embedding_model, EnrollmentDB, _cosine_similarity, _unit_vector

RESULTS_CSV = Path("data/eval_results.csv")
DB_PATH = Path("data/processed/enrollment_db")
N_UTTERANCES = 10


def collect(args: argparse.Namespace) -> None:
    model_path = args.model
    model, cfg = load_embedding_model(model_path)
    device = resolve_device("auto")
    model = model.to(device)

    db = EnrollmentDB(DB_PATH)
    enrolled = db.speakers()
    if not enrolled:
        print("No enrolled speakers found. Enroll first via the UI.")
        return

    print(f"Model: {model_path}")
    print(f"Enrolled speakers: {', '.join(enrolled)}")
    print(f"Utterances per test: {args.n}\n")

    tester = input("Who is speaking? (your name): ").strip().lower()
    if not tester:
        return
    print(f"\nSpeak anything for 3+ seconds each time.\n")

    recorder = MicrophoneRecorder(
        output_dir=Path("data/raw/eval_recordings"),
        metadata_csv=Path("data/raw/eval_recordings/metadata.csv"),
        preprocess=True,
    )

    rows = []
    for i in range(1, args.n + 1):
        input(f"[{i}/{args.n}] Press Enter to start (wait for 'Speak now' before speaking)...")
        def _on_state(s: str) -> None:
            if s == "waiting":
                print("        Speak now", flush=True)
        try:
            result = recorder.record_vad(
                max_seconds=5.0,
                min_speech_ms=500,
                save=True,
                file_prefix=f"eval_{tester}_{i:03d}",
                on_state_change=_on_state,
            )
            if result.duration < 1.5:
                print(f"    too short ({result.duration:.1f}s), skipping")
                continue
        except RuntimeError as e:
            print(f"    failed: {e}")
            continue

        query = _unit_vector(embed(model, [result.saved_path], cfg, device))

        print(f"        duration={result.duration:.1f}s  scores:", end="")
        for speaker_id in enrolled:
            prompts = db.prompts_for_speaker(speaker_id)
            if not prompts:
                continue
            vecs = [db.load_prompt_vector(speaker_id, k) for k in prompts]
            ref = _unit_vector(np.mean(np.stack(vecs, axis=0), axis=0))
            score = float(_cosine_similarity(query, ref))

            is_genuine = (tester == speaker_id)
            rows.append({
                "tester": tester,
                "enrolled_speaker": speaker_id,
                "score": round(score, 4),
                "is_genuine": is_genuine,
                "duration": round(result.duration, 2),
                "file": str(result.saved_path),
            })
            marker = " <--" if is_genuine else ""
            print(f"  {speaker_id}={score:.4f}{marker}", end="")
        print()

    if rows:
        df = pd.DataFrame(rows)
        if RESULTS_CSV.exists():
            old = pd.read_csv(RESULTS_CSV)
            df = pd.concat([old, df], ignore_index=True)
        RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULTS_CSV, index=False)
        print(f"\nSaved {len(rows)} scores to {RESULTS_CSV}")
        print(f"Total rows in CSV: {len(df)}")
    else:
        print("No valid recordings captured.")


def analyze(args: argparse.Namespace) -> None:
    if not RESULTS_CSV.exists():
        print(f"No results found at {RESULTS_CSV}. Collect data first.")
        return

    df = pd.read_csv(RESULTS_CSV)
    genuine = df[df["is_genuine"] == True]["score"].values
    imposter = df[df["is_genuine"] == False]["score"].values

    print(f"=== Evaluation Results ===")
    print(f"Total scores: {len(df)} ({len(genuine)} genuine, {len(imposter)} imposter)")
    print(f"Testers: {', '.join(df['tester'].unique())}")
    print(f"Enrolled: {', '.join(df['enrolled_speaker'].unique())}")
    print()
    if len(genuine) > 0:
        print(f"Genuine  scores: mean={genuine.mean():.4f} std={genuine.std():.4f} min={genuine.min():.4f} max={genuine.max():.4f}")
    else:
        print("Genuine  scores: no data (enrolled speaker hasn't tested yet)")
    if len(imposter) > 0:
        print(f"Imposter scores: mean={imposter.mean():.4f} std={imposter.std():.4f} min={imposter.min():.4f} max={imposter.max():.4f}")
    else:
        print("Imposter scores: no data (need other people to test)")
    print()

    # Per-tester breakdown
    print("--- Per tester ---")
    for tester in sorted(df["tester"].unique()):
        t_df = df[df["tester"] == tester]
        g = t_df[t_df["is_genuine"] == True]["score"]
        i = t_df[t_df["is_genuine"] == False]["score"]
        g_str = f"genuine: mean={g.mean():.4f} ({len(g)} samples)" if len(g) > 0 else "genuine: N/A (not enrolled)"
        i_str = f"imposter: mean={i.mean():.4f} ({len(i)} samples)" if len(i) > 0 else ""
        print(f"  {tester}: {g_str}  {i_str}")
    print()

    # Threshold sweep
    print("--- Threshold sweep ---")
    print(f"{'threshold':>10} {'FAR':>8} {'FRR':>8} {'accuracy':>10}")
    best_acc = 0
    best_t = 0
    for t in np.arange(0.05, 0.95, 0.05):
        far = (imposter >= t).mean() if len(imposter) > 0 else 0
        frr = (genuine < t).mean() if len(genuine) > 0 else 0
        total = len(genuine) + len(imposter)
        correct = ((genuine >= t).sum() + (imposter < t).sum())
        acc = correct / total if total > 0 else 0
        print(f"  {t:.2f}     {far:.4f}   {frr:.4f}   {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_t = t

    # EER
    thresholds = np.linspace(0, 1, 1000)
    fars = np.array([(imposter >= t).mean() for t in thresholds]) if len(imposter) > 0 else np.zeros_like(thresholds)
    frrs = np.array([(genuine < t).mean() for t in thresholds]) if len(genuine) > 0 else np.zeros_like(thresholds)
    eer_idx = np.argmin(np.abs(fars - frrs))
    eer = fars[eer_idx]
    eer_t = thresholds[eer_idx]

    print()
    print(f"EER: {eer:.4f} at threshold={eer_t:.4f}")
    print(f"Best accuracy: {best_acc:.4f} at threshold={best_t:.2f}")
    print(f"Recommended threshold: {eer_t:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-world speaker verification evaluation")
    parser.add_argument("--analyze", action="store_true", help="Analyze collected results")
    parser.add_argument("--model", default="checkpoints/model_mfcc40_v1.pt", help="Model checkpoint")
    parser.add_argument("--n", type=int, default=N_UTTERANCES, help="Utterances per test")
    args = parser.parse_args()

    if args.analyze:
        analyze(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
