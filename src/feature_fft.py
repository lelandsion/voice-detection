from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd


N_MFCC = 13
SR_TARGET = 22050


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
	"""Convert audio array to mono float32 in [-1, 1]."""
	x = np.asarray(audio)

	if x.dtype.kind in {"i", "u"}:
		max_val = np.iinfo(x.dtype).max
		x = x.astype(np.float32) / float(max_val)
	else:
		x = x.astype(np.float32)

	if x.ndim > 1:
		x = np.mean(x, axis=1)

	return np.clip(x, -1.0, 1.0)


def extract_features_from_audio(
	audio: np.ndarray,
	sample_rate: int,
	sr_target: int = SR_TARGET,
	n_mfcc: int = N_MFCC,
	min_duration_s: float = 0.3,
) -> dict[str, float] | None:
	"""Extract the project feature vector from in-memory audio."""
	try:
		y = _to_mono_float32(audio)
		if sample_rate != sr_target:
			y = librosa.resample(y=y, orig_sr=sample_rate, target_sr=sr_target)
		sr = sr_target

		if len(y) < int(sr * min_duration_s):
			return None

		feats: dict[str, float] = {}

		mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
		for i in range(n_mfcc):
			feats[f"mfcc{i + 1}_mean"] = float(np.mean(mfcc[i]))
			feats[f"mfcc{i + 1}_std"] = float(np.std(mfcc[i]))

		f0 = librosa.yin(y=y, fmin=50, fmax=400, sr=sr)
		voiced = f0[f0 > 0]
		feats["pitch_mean"] = float(np.mean(voiced)) if len(voiced) > 0 else 0.0
		feats["pitch_std"] = float(np.std(voiced)) if len(voiced) > 0 else 0.0
		feats["voiced_ratio"] = float(len(voiced) / max(len(f0), 1))

		centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
		bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
		rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
		flatness = librosa.feature.spectral_flatness(y=y)[0]
		feats["spec_centroid_mean"] = float(np.mean(centroid))
		feats["spec_centroid_std"] = float(np.std(centroid))
		feats["spec_bandwidth_mean"] = float(np.mean(bandwidth))
		feats["spec_rolloff_mean"] = float(np.mean(rolloff))
		feats["spec_flatness_mean"] = float(np.mean(flatness))

		zcr = librosa.feature.zero_crossing_rate(y=y)[0]
		feats["zcr_mean"] = float(np.mean(zcr))
		feats["zcr_std"] = float(np.std(zcr))

		rms = librosa.feature.rms(y=y)[0]
		feats["rms_mean"] = float(np.mean(rms))
		feats["rms_std"] = float(np.std(rms))

		chroma = librosa.feature.chroma_stft(y=y, sr=sr)
		feats["chroma_mean"] = float(np.mean(chroma))
		feats["chroma_std"] = float(np.std(chroma))

		mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
		mel_db = librosa.power_to_db(mel)
		for band_idx, (lo, hi) in enumerate(((0, 16), (16, 32), (32, 48), (48, 64)), start=1):
			feats[f"mel_band{band_idx}_mean"] = float(np.mean(mel_db[lo:hi]))

		return feats
	except Exception:
		return None


def extract_features_from_file(
	path: str | Path,
	sr_target: int = SR_TARGET,
	n_mfcc: int = N_MFCC,
	min_duration_s: float = 0.3,
) -> dict[str, float] | None:
	"""Extract the project feature vector from an audio file path."""
	try:
		y, sr = librosa.load(str(path), sr=sr_target, mono=True)
		return extract_features_from_audio(
			audio=y,
			sample_rate=int(sr),
			sr_target=sr_target,
			n_mfcc=n_mfcc,
			min_duration_s=min_duration_s,
		)
	except Exception:
		return None


def extract_feature_table(
	df: pd.DataFrame,
	audio_path_col: str = "audio_path",
	sr_target: int = SR_TARGET,
	n_mfcc: int = N_MFCC,
	min_duration_s: float = 0.3,
	meta_cols: list[str] | None = None,
) -> pd.DataFrame:
	"""Batch-extract features for a dataframe with one audio path per row."""
	if audio_path_col not in df.columns:
		raise ValueError(f"Missing required column: {audio_path_col}")

	if meta_cols is None:
		default_meta = ["path", "client_id", "spk_idx", "gender", "age", "duration_ms"]
		meta_cols = [c for c in default_meta if c in df.columns]

	rows: list[dict[str, object]] = []
	for _, row in df.iterrows():
		feats = extract_features_from_file(
			path=row[audio_path_col],
			sr_target=sr_target,
			n_mfcc=n_mfcc,
			min_duration_s=min_duration_s,
		)
		if feats is None:
			continue

		out: dict[str, object] = dict(feats)
		for col in meta_cols:
			out[col] = row[col]
		rows.append(out)

	return pd.DataFrame(rows)


def get_feature_columns(df: pd.DataFrame, meta_cols: list[str] | None = None) -> list[str]:
	"""Return numeric feature columns excluding metadata columns."""
	if meta_cols is None:
		meta_cols = ["path", "client_id", "spk_idx", "gender", "age", "duration", "duration_ms"]
	return [
		c
		for c in df.columns
		if c not in set(meta_cols) and pd.api.types.is_numeric_dtype(df[c])
	]

