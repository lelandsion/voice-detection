
# #17:Process Input Sound Using DFT/FFT
# Use this script to convert time-domain waveforms into frequency-domain representations and measure frequency strength.
# 1. Place `.wav` files in `data/raw/recordings/`.
# 2. Run: python process_input_fft.py
# 3. Outputs:
# - `data/processed/fft_summary.csv` (dominant frequency + spectral metrics per clip)
# - `data/processed/fft_spectra/*.csv` (full per-file FFT spectrum: `freq_hz`, `magnitude`, `normalized_magnitude`)
# `fft_summary.csv` includes:
# - `dominant_freq_hz`: strongest detected frequency component
# - `dominant_strength`: FFT magnitude at that frequency
# - `spectral_centroid_hz`: energy-weighted average frequency
# - `spectral_bandwidth_hz`: spectral spread around the centroid
# - `rolloff_85_hz`: frequency below which 85% of spectral energy lies

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile


TARGET_SAMPLE_RATE = 16000
DEFAULT_INPUT_DIR = Path("data/raw/recordings")
DEFAULT_SUMMARY_CSV = Path("data/processed/fft_summary.csv")
DEFAULT_SPECTRUM_DIR = Path("data/processed/fft_spectra")


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


def _normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak < 1e-8:
        return audio.astype(np.float32)
    return (audio / peak).astype(np.float32)


def _fft_spectrum(audio: np.ndarray, sample_rate: int, n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    if len(audio) < n_fft:
        padded = np.pad(audio, (0, n_fft - len(audio)))
    else:
        padded = audio[:n_fft]

    windowed = padded * np.hanning(len(padded))
    fft_complex = np.fft.rfft(windowed)
    magnitude = np.abs(fft_complex).astype(np.float32)
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate).astype(np.float32)
    return freqs, magnitude


def _spectral_metrics(freqs: np.ndarray, magnitude: np.ndarray) -> dict[str, float]:
    power = magnitude ** 2
    power_sum = float(np.sum(power))

    if power_sum <= 1e-12:
        return {
            "dominant_freq_hz": 0.0,
            "dominant_strength": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_bandwidth_hz": 0.0,
            "rolloff_85_hz": 0.0,
        }

    dominant_idx = int(np.argmax(magnitude))
    dominant_freq_hz = float(freqs[dominant_idx])
    dominant_strength = float(magnitude[dominant_idx])

    centroid = float(np.sum(freqs * power) / power_sum)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / power_sum))

    cumulative_power = np.cumsum(power)
    rolloff_idx = int(np.searchsorted(cumulative_power, 0.85 * power_sum))
    rolloff_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    return {
        "dominant_freq_hz": dominant_freq_hz,
        "dominant_strength": dominant_strength,
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "rolloff_85_hz": rolloff_hz,
    }


def process_wav_file(
    wav_path: Path,
    target_sample_rate: int,
    n_fft: int,
    spectrum_dir: Path,
) -> dict[str, str | int | float]:
    sample_rate, audio = wavfile.read(wav_path)
    audio = _audio_to_float32(audio)
    audio = _resample_linear(audio, sample_rate, target_sample_rate)
    audio = _normalize(audio)

    freqs, magnitude = _fft_spectrum(audio, target_sample_rate, n_fft)
    metrics = _spectral_metrics(freqs, magnitude)

    spectrum_df = pd.DataFrame(
        {
            "freq_hz": freqs,
            "magnitude": magnitude,
            "normalized_magnitude": magnitude / (np.max(magnitude) + 1e-8),
        }
    )
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    spectrum_path = spectrum_dir / f"{wav_path.stem}_fft.csv"
    spectrum_df.to_csv(spectrum_path, index=False)

    return {
        "file": str(wav_path),
        "sample_rate": target_sample_rate,
        "duration_s": float(len(audio) / target_sample_rate),
        "n_fft": n_fft,
        "spectrum_csv": str(spectrum_path),
        **metrics,
    }


def build_fft_outputs(
    input_dir: Path,
    summary_csv: Path,
    spectrum_dir: Path,
    target_sample_rate: int,
    n_fft: int,
) -> pd.DataFrame:
    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {input_dir}")

    rows: list[dict[str, str | int | float]] = []
    for wav_path in wav_files:
        rows.append(process_wav_file(wav_path, target_sample_rate, n_fft, spectrum_dir))

    df = pd.DataFrame(rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process input audio with FFT/DFT to analyze present frequencies and strengths."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--spectrum-dir", type=Path, default=DEFAULT_SPECTRUM_DIR)
    parser.add_argument("--sample-rate", type=int, default=TARGET_SAMPLE_RATE)
    parser.add_argument("--n-fft", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_fft_outputs(
        input_dir=args.input_dir,
        summary_csv=args.summary_csv,
        spectrum_dir=args.spectrum_dir,
        target_sample_rate=args.sample_rate,
        n_fft=args.n_fft,
    )
    print(f"Processed {len(df)} clips with FFT")
    print(f"Summary written to {args.summary_csv}")
    print(f"Per-file spectra written to {args.spectrum_dir}")


if __name__ == "__main__":
    main()