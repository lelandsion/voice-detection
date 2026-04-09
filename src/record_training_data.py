"""Record real microphone data for training. Run from project root."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audio import MicrophoneRecorder

OUTPUT_DIR = Path("data/real_mic")
RECORDINGS_PER_PERSON = 20


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Record Training Data ===")
    print(f"Each person records {RECORDINGS_PER_PERSON} utterances.\n")

    while True:
        name = input("Speaker name (or 'done' to finish): ").strip().lower()
        if name == "done":
            break
        if not name:
            continue

        speaker_dir = OUTPUT_DIR / name
        speaker_dir.mkdir(exist_ok=True)
        existing = list(speaker_dir.glob("*.wav"))
        print(f"\n  Speaker: {name} ({len(existing)} existing recordings)")

        recorder = MicrophoneRecorder(
            output_dir=speaker_dir,
            metadata_csv=OUTPUT_DIR / "metadata.csv",
            preprocess=True,
        )

        for i in range(1, RECORDINGS_PER_PERSON + 1):
            input(f"  [{i}/{RECORDINGS_PER_PERSON}] Press Enter then speak for 3 seconds...")
            try:
                result = recorder.record_vad(
                    max_seconds=5.0,
                    min_speech_ms=500,
                    save=True,
                    file_prefix=f"{name}_{i:03d}",
                )
                print(f"    saved: {result.saved_path} ({result.duration:.1f}s, {result.quality})")
            except RuntimeError as e:
                print(f"    failed: {e} — try again")
                i -= 1
                continue

        total = len(list(speaker_dir.glob("*.wav")))
        print(f"  Done! {name} has {total} recordings.\n")

    # Generate training CSV
    print("\nGenerating training CSV...")
    import pandas as pd
    rows = []
    for speaker_dir in sorted(OUTPUT_DIR.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker = speaker_dir.name
        for wav in sorted(speaker_dir.glob("*.wav")):
            rows.append({"client_id": speaker, "path": wav.name})

    if rows:
        csv_path = OUTPUT_DIR / "real_mic_training.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved {len(rows)} rows to {csv_path}")
        print(f"Speakers: {len(set(r['client_id'] for r in rows))}")
    else:
        print("No recordings found.")


if __name__ == "__main__":
    main()
