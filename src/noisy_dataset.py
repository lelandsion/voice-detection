from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.io.wavfile import write as wav_write


SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass
class NoiseDatasetConfig:
    input_csv: Path
    clean_audio_dir: Path
    music_noise_dir: Path
    env_noise_dir: Path
    output_audio_dir: Path
    output_csv: Path
    output_manifest: Path
    sample_rate: int = 16000
    num_augmentations: int = 1
    seed: int = 42
    music_snr_db: tuple[float, ...] = (20.0, 15.0, 10.0, 5.0, 0.0)
    env_snr_db: tuple[float, ...] = (20.0, 15.0, 10.0, 5.0, 0.0)


@dataclass
class NoiseDatasetStats:
    source_rows: int
    generated_rows: int
    skipped_missing_clean_audio: int
    output_csv: str
    output_audio_dir: str
    output_manifest: str


def _list_audio_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            files.append(p)
    return files


def _candidate_child_dirs(root: Path) -> list[str]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _validate_noise_dir(root: Path, label: str) -> list[Path]:
    if not root.exists() or not root.is_dir():
        parent = root.parent
        sibling_dirs = _candidate_child_dirs(parent)
        sibling_msg = ", ".join(sibling_dirs[:8]) if sibling_dirs else "none"
        raise FileNotFoundError(
            f"{label} directory not found: {root}. "
            f"Available directories under {parent}: {sibling_msg}."
        )

    files = _list_audio_files(root)
    if not files:
        child_dirs = _candidate_child_dirs(root)
        child_msg = ", ".join(child_dirs[:8]) if child_dirs else "none"
        exts = ", ".join(sorted(SUPPORTED_AUDIO_EXTS))
        raise FileNotFoundError(
            f"No supported audio files found in {label}: {root}. "
            f"Supported extensions: {exts}. "
            f"Subdirectories found: {child_msg}."
        )

    return files


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def _fit_to_length(noise: np.ndarray, target_len: int, rng: np.random.Generator) -> np.ndarray:
    if target_len <= 0:
        return np.zeros((0,), dtype=np.float32)
    if noise.size == 0:
        return np.zeros((target_len,), dtype=np.float32)

    if noise.size < target_len:
        reps = int(np.ceil(target_len / max(noise.size, 1)))
        tiled = np.tile(noise, reps)
        return tiled[:target_len].astype(np.float32)

    if noise.size == target_len:
        return noise.astype(np.float32)

    start = int(rng.integers(0, noise.size - target_len + 1))
    return noise[start : start + target_len].astype(np.float32)


def _scale_noise_for_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_rms = _rms(clean)
    noise_rms = _rms(noise)

    if clean_rms < 1e-8 or noise_rms < 1e-8:
        return np.zeros_like(noise, dtype=np.float32)

    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scale = target_noise_rms / (noise_rms + 1e-8)
    return (noise * scale).astype(np.float32)


def _mix_clean_with_noise(
    clean: np.ndarray,
    music_noise: np.ndarray,
    env_noise: np.ndarray,
    music_snr_db: float,
    env_snr_db: float,
) -> np.ndarray:
    music_scaled = _scale_noise_for_snr(clean=clean, noise=music_noise, snr_db=music_snr_db)
    env_scaled = _scale_noise_for_snr(clean=clean, noise=env_noise, snr_db=env_snr_db)

    mixed = clean + music_scaled + env_scaled
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.999:
        mixed = mixed / peak * 0.999
    return mixed.astype(np.float32)


def _write_wav(path: Path, audio_float32: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_clipped = np.clip(audio_float32, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767.0).astype(np.int16)
    wav_write(str(path), sample_rate, audio_int16)


def _stable_seed(base_seed: int, rel_path: str, aug_idx: int) -> int:
    payload = f"{base_seed}|{rel_path}|{aug_idx}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed64 = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return seed64 % (2**32)


def build_noisy_dataset(config: NoiseDatasetConfig) -> NoiseDatasetStats:
    if config.num_augmentations < 1:
        raise ValueError("num_augmentations must be at least 1")

    df = pd.read_csv(config.input_csv)
    if "path" not in df.columns:
        raise ValueError("Input CSV must contain a 'path' column.")
    if "client_id" not in df.columns:
        raise ValueError("Input CSV must contain a 'client_id' column.")

    music_files = _validate_noise_dir(config.music_noise_dir, label="music_noise_dir")
    env_files = _validate_noise_dir(config.env_noise_dir, label="env_noise_dir")

    rows_out: list[dict[str, object]] = []
    skipped_missing_clean_audio = 0
    noise_cache: dict[Path, np.ndarray] = {}

    def load_noise_cached(path: Path) -> np.ndarray:
        cached = noise_cache.get(path)
        if cached is not None:
            return cached
        audio = _load_audio(path, sample_rate=config.sample_rate)
        noise_cache[path] = audio
        return audio

    for _, row in df.iterrows():
        clean_rel = Path(str(row["path"]).replace("\\", "/"))
        clean_abs = config.clean_audio_dir / clean_rel

        if not clean_abs.exists():
            skipped_missing_clean_audio += 1
            continue

        clean_audio = _load_audio(clean_abs, sample_rate=config.sample_rate)

        for aug_idx in range(config.num_augmentations):
            sample_seed = _stable_seed(config.seed, clean_rel.as_posix(), aug_idx)
            rng = np.random.default_rng(sample_seed)

            music_path = music_files[int(rng.integers(0, len(music_files)))]
            env_path = env_files[int(rng.integers(0, len(env_files)))]
            music_snr = float(rng.choice(np.asarray(config.music_snr_db, dtype=np.float32)))
            env_snr = float(rng.choice(np.asarray(config.env_snr_db, dtype=np.float32)))

            music_audio = load_noise_cached(music_path)
            env_audio = load_noise_cached(env_path)

            music_seg = _fit_to_length(music_audio, target_len=clean_audio.size, rng=rng)
            env_seg = _fit_to_length(env_audio, target_len=clean_audio.size, rng=rng)

            mixed = _mix_clean_with_noise(
                clean=clean_audio,
                music_noise=music_seg,
                env_noise=env_seg,
                music_snr_db=music_snr,
                env_snr_db=env_snr,
            )

            out_rel = Path(clean_rel.with_suffix(""))
            out_rel = Path(f"{out_rel.as_posix()}__noisy_{aug_idx:02d}.wav")
            out_abs = config.output_audio_dir / out_rel
            _write_wav(path=out_abs, audio_float32=mixed, sample_rate=config.sample_rate)

            new_row = row.to_dict()
            new_row["path"] = out_rel.as_posix()
            new_row["clean_path"] = clean_rel.as_posix()
            new_row["noise_music_path"] = str(music_path)
            new_row["noise_env_path"] = str(env_path)
            new_row["noise_music_snr_db"] = music_snr
            new_row["noise_env_snr_db"] = env_snr
            new_row["noise_seed"] = int(sample_seed)
            new_row["augmentation_index"] = int(aug_idx)
            rows_out.append(new_row)

    out_df = pd.DataFrame(rows_out)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(config.output_csv, index=False)

    manifest = {
        "config": {
            **asdict(config),
            "input_csv": str(config.input_csv),
            "clean_audio_dir": str(config.clean_audio_dir),
            "music_noise_dir": str(config.music_noise_dir),
            "env_noise_dir": str(config.env_noise_dir),
            "output_audio_dir": str(config.output_audio_dir),
            "output_csv": str(config.output_csv),
            "output_manifest": str(config.output_manifest),
            "music_snr_db": list(config.music_snr_db),
            "env_snr_db": list(config.env_snr_db),
        },
        "stats": {
            "source_rows": int(len(df)),
            "generated_rows": int(len(rows_out)),
            "skipped_missing_clean_audio": int(skipped_missing_clean_audio),
            "music_noise_files": int(len(music_files)),
            "env_noise_files": int(len(env_files)),
        },
    }

    config.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    config.output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return NoiseDatasetStats(
        source_rows=int(len(df)),
        generated_rows=int(len(rows_out)),
        skipped_missing_clean_audio=int(skipped_missing_clean_audio),
        output_csv=str(config.output_csv),
        output_audio_dir=str(config.output_audio_dir),
        output_manifest=str(config.output_manifest),
    )


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return tuple(values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible noisy training dataset by mixing each clean clip "
            "with one music noise source and one environmental noise source."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/commonvoice-v24_en-AU 3/commonvoice-v24_en-AU.csv"),
        help="CSV with at least 'path' and 'client_id' columns.",
    )
    parser.add_argument(
        "--clean-audio-dir",
        type=Path,
        default=Path("data/commonvoice-v24_en-AU 3/audio_files"),
        help="Root directory used by the input CSV paths.",
    )
    parser.add_argument("--music-noise-dir", type=Path, required=True)
    parser.add_argument("--env-noise-dir", type=Path, required=True)
    parser.add_argument(
        "--output-audio-dir",
        type=Path,
        default=Path("data/processed/noisy/audio_files"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/commonvoice_noisy.csv"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("data/processed/commonvoice_noisy_manifest.json"),
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-augmentations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--music-snr-db",
        type=str,
        default="20,15,10,5,0",
        help="Comma-separated list of music SNR dB values sampled per clip.",
    )
    parser.add_argument(
        "--env-snr-db",
        type=str,
        default="20,15,10,5,0",
        help="Comma-separated list of environmental SNR dB values sampled per clip.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config = NoiseDatasetConfig(
        input_csv=args.input_csv,
        clean_audio_dir=args.clean_audio_dir,
        music_noise_dir=args.music_noise_dir,
        env_noise_dir=args.env_noise_dir,
        output_audio_dir=args.output_audio_dir,
        output_csv=args.output_csv,
        output_manifest=args.output_manifest,
        sample_rate=args.sample_rate,
        num_augmentations=args.num_augmentations,
        seed=args.seed,
        music_snr_db=_parse_float_list(args.music_snr_db),
        env_snr_db=_parse_float_list(args.env_snr_db),
    )

    stats = build_noisy_dataset(config)
    print("Noisy dataset generation complete.")
    print(f"  Source rows: {stats.source_rows}")
    print(f"  Generated rows: {stats.generated_rows}")
    print(f"  Skipped missing clean audio: {stats.skipped_missing_clean_audio}")
    print(f"  Output CSV: {stats.output_csv}")
    print(f"  Output audio root: {stats.output_audio_dir}")
    print(f"  Manifest: {stats.output_manifest}")


if __name__ == "__main__":
    main()