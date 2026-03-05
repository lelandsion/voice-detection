
# #15 ("Preprocess data to begin with a start frequency").

# 1. Place `.wav` files in `data/raw/recordings/`.
# 2. Run: python preprocess_start_frequency.py
# 3. Output file will be created at:
# `data/processed/start_frequency_features.csv`
#  generated table includes:
# - `start_time_s` and `start_freq_hz` (detected onset location and dominant start frequency)
# - normalized spectral bins (`fft_bin_000` to `fft_bin_128`) for model input
# - additional normalized signal stats (`rms`, `zero_crossing_rate`, `duration_s`)

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile


TARGET_SAMPLE_RATE = 16000
RAW_RECORDINGS_DIR = Path("data/raw/recordings")
OUTPUT_FEATURES_CSV = Path("data/processed/start_frequency_features.csv")


@dataclass
class ClipFeatures:
    file: str
    sample_rate: int
    duration_s: float
    start_time_s: float
    start_freq_hz: float
    rms: float
    zero_crossing_rate: float
    fft_0_128: np.ndarray


def _audio_to_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype.kind in {"i", "u"}:
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.clip(audio, -1.0, 1.0)


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    if len(audio) < 2:
        return audio

    src_x = np.arange(len(audio), dtype=np.float32)
    target_len = int(round(len(audio) * (dst_sr / src_sr)))
    dst_x = np.linspace(0, len(audio) - 1, target_len, dtype=np.float32)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def _noise_reduce_and_normalize(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    noise_len = min(int(sample_rate * 0.5), len(audio))
    if noise_len == 0:
        return audio.astype(np.float32)

    noise_clip = audio[:noise_len]
    threshold = np.std(noise_clip) * 1.5
    reduced = np.where(np.abs(audio) < threshold, 0.0, audio)

    peak = np.max(np.abs(reduced))
    if peak < 1e-8:
        return reduced.astype(np.float32)
    return (reduced / peak).astype(np.float32)


def _find_start_index(audio: np.ndarray, sample_rate: int, frame_ms: int = 20) -> int:
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    if len(audio) <= frame_len:
        return 0

    energy = np.convolve(audio ** 2, np.ones(frame_len, dtype=np.float32), mode="same") / frame_len
    baseline = np.median(energy[: min(len(energy), sample_rate // 2 or 1)])
    threshold = max(baseline * 4.0, 1e-5)

    above = np.where(energy > threshold)[0]
    return int(above[0]) if above.size else 0


def _start_frequency(audio: np.ndarray, sample_rate: int, start_idx: int, window_ms: int = 80) -> float:
    win_len = max(64, int(sample_rate * window_ms / 1000))
    end_idx = min(len(audio), start_idx + win_len)
    segment = audio[start_idx:end_idx]
    if len(segment) < 16:
        return 0.0

    window = np.hanning(len(segment)).astype(np.float32)
    spectrum = np.fft.rfft(segment * window)
    mags = np.abs(spectrum)
    if np.all(mags <= 1e-12):
        return 0.0

    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sample_rate)
    peak_idx = int(np.argmax(mags))
    return float(freqs[peak_idx])


def _fft_features(audio: np.ndarray, start_idx: int, n_fft: int = 256) -> np.ndarray:
    left = max(0, min(start_idx, len(audio) - 1)) if len(audio) else 0
    segment = audio[left : left + n_fft]

    if len(segment) < n_fft:
        padded = np.pad(segment, (0, n_fft - len(segment)))
    else:
        padded = segment

    mags = np.abs(np.fft.rfft(padded * np.hanning(len(padded))))
    mags = mags[:129]
    mags = mags / (np.max(mags) + 1e-8)
    return mags.astype(np.float32)


def extract_features_for_file(path: Path, target_sample_rate: int) -> ClipFeatures:
    sample_rate, audio = wavfile.read(path)
    audio_f32 = _audio_to_float32(audio)
    audio_resampled = _resample_linear(audio_f32, sample_rate, target_sample_rate)
    audio_processed = _noise_reduce_and_normalize(audio_resampled, target_sample_rate)

    start_idx = _find_start_index(audio_processed, target_sample_rate)
    start_freq = _start_frequency(audio_processed, target_sample_rate, start_idx)

    rms = float(np.sqrt(np.mean(audio_processed ** 2))) if audio_processed.size else 0.0
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio_processed))))) if len(audio_processed) > 1 else 0.0

    return ClipFeatures(
        file=str(path),
        sample_rate=target_sample_rate,
        duration_s=float(len(audio_processed) / target_sample_rate),
        start_time_s=float(start_idx / target_sample_rate),
        start_freq_hz=start_freq,
        rms=rms,
        zero_crossing_rate=zcr,
        fft_0_128=_fft_features(audio_processed, start_idx=start_idx),
    )


def build_feature_table(input_dir: Path, output_csv: Path, target_sample_rate: int) -> pd.DataFrame:
    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    rows: list[dict[str, float | int | str]] = []
    for wav_path in wav_files:
        clip = extract_features_for_file(wav_path, target_sample_rate)
        row: dict[str, float | int | str] = {
            "file": clip.file,
            "sample_rate": clip.sample_rate,
            "duration_s": clip.duration_s,
            "start_time_s": clip.start_time_s,
            "start_freq_hz": clip.start_freq_hz,
            "rms": clip.rms,
            "zero_crossing_rate": clip.zero_crossing_rate,
        }
        for i, value in enumerate(clip.fft_0_128):
            row[f"fft_bin_{i:03d}"] = float(value)
        rows.append(row)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess wav files with start-frequency extraction and normalized FFT features."
    )
    parser.add_argument("--input-dir", type=Path, default=RAW_RECORDINGS_DIR)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_FEATURES_CSV)
    parser.add_argument("--sample-rate", type=int, default=TARGET_SAMPLE_RATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_feature_table(args.input_dir, args.output_csv, args.sample_rate)
    print(f"Processed {len(df)} clips")
    print(f"Wrote features to {args.output_csv}")


if __name__ == "__main__":
    main()