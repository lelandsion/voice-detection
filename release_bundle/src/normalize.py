from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile


TARGET_SR = 16000
TARGET_DB = -20.0
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512


def _to_float_mono(audio: np.ndarray) -> np.ndarray:
	x = np.asarray(audio)
	if x.dtype.kind in {"i", "u"}:
		max_val = np.iinfo(x.dtype).max
		x = x.astype(np.float32) / float(max_val)
	else:
		x = x.astype(np.float32)
	if x.ndim > 1:
		x = np.mean(x, axis=1)
	return x.astype(np.float32)


def standardize_sample_rate(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
	"""Resample audio to target_sr using polyphase resampling."""
	if orig_sr == target_sr:
		return audio.astype(np.float32)
	gcd = np.gcd(orig_sr, target_sr)
	resampled = scipy_signal.resample_poly(audio, target_sr // gcd, orig_sr // gcd)
	return resampled.astype(np.float32)


def normalize_rms(audio: np.ndarray, target_db: float = TARGET_DB) -> np.ndarray:
	"""Scale audio to a target RMS level in dBFS."""
	rms = np.sqrt(np.mean(np.square(audio)))
	if rms < 1e-8:
		return audio.astype(np.float32)
	target_rms = 10.0 ** (target_db / 20.0)
	return (audio * (target_rms / rms)).astype(np.float32)


def normalize_peak(audio: np.ndarray) -> np.ndarray:
	"""Scale so the peak absolute value equals 1.0."""
	peak = np.max(np.abs(audio)) if audio.size else 0.0
	if peak < 1e-8:
		return audio.astype(np.float32)
	return (audio / peak).astype(np.float32)


def dynamic_range_compress(
	audio: np.ndarray,
	threshold: float = 0.3,
	ratio: float = 4.0,
	makeup_gain: float = 1.2,
) -> np.ndarray:
	"""Compress amplitudes above threshold and apply makeup gain."""
	audio = audio.astype(np.float32)
	abs_a = np.abs(audio)
	compressed_abs = np.where(abs_a > threshold, threshold + (abs_a - threshold) / ratio, abs_a)
	safe_abs_a = np.where(abs_a > 1e-8, abs_a, 1.0)
	gain = np.where(abs_a > 1e-8, compressed_abs / safe_abs_a, 1.0)
	return np.clip(audio * gain * makeup_gain, -1.0, 1.0).astype(np.float32)


def log_scale_features(features: np.ndarray, floor_db: float = -80.0) -> np.ndarray:
	"""Convert magnitudes to dB scale and clip to floor_db."""
	features = np.maximum(features, 1e-8)
	db = 20.0 * np.log10(features)
	return np.maximum(db, floor_db).astype(np.float32)


def minmax_normalize(features: np.ndarray) -> np.ndarray:
	lo, hi = float(features.min()), float(features.max())
	if hi - lo < 1e-8:
		return np.zeros_like(features, dtype=np.float32)
	return ((features - lo) / (hi - lo)).astype(np.float32)


def zscore_normalize(features: np.ndarray) -> np.ndarray:
	mu, sigma = float(features.mean()), float(features.std())
	if sigma < 1e-8:
		return np.zeros_like(features, dtype=np.float32)
	return ((features - mu) / sigma).astype(np.float32)


def cmvn(features: np.ndarray) -> np.ndarray:
	"""Cepstral mean/variance normalization along feature dimension."""
	if features.ndim == 1:
		features = features.reshape(-1, 1)
	mu = features.mean(axis=0, keepdims=True)
	sigma = features.std(axis=0, keepdims=True)
	sigma = np.where(sigma < 1e-8, 1.0, sigma)
	return ((features - mu) / sigma).astype(np.float32)


def distribution_stats(data: np.ndarray) -> dict[str, float]:
	flat = data.flatten()
	return {
		"mean": float(np.mean(flat)),
		"std": float(np.std(flat)),
		"min": float(np.min(flat)),
		"max": float(np.max(flat)),
		"p25": float(np.percentile(flat, 25)),
		"p75": float(np.percentile(flat, 75)),
		"rms": float(np.sqrt(np.mean(np.square(flat)))),
		"range": float(np.max(flat) - np.min(flat)),
	}


def compare_distributions(before: np.ndarray, after: np.ndarray) -> dict[str, dict[str, float]]:
	b = distribution_stats(before)
	a = distribution_stats(after)
	delta = {k: round(a[k] - b[k], 6) for k in b}
	return {"before": b, "after": a, "delta": delta}


def evaluate_consistency(feature_list: list[np.ndarray]) -> dict[str, np.ndarray | float | int]:
	summaries = np.stack([f.mean(axis=0) for f in feature_list], axis=0)
	per_dim_var = summaries.var(axis=0)
	return {
		"per_dim_variance": per_dim_var,
		"mean_cross_variance": float(per_dim_var.mean()),
		"max_cross_variance": float(per_dim_var.max()),
		"n_recordings": len(feature_list),
	}


class NormalizationPipeline:
	"""End-to-end normalization for raw audio and FFT magnitude features."""

	def __init__(
		self,
		target_sr: int = TARGET_SR,
		target_db: float = TARGET_DB,
		n_mfcc: int = N_MFCC,
		n_fft: int = N_FFT,
		hop_length: int = HOP_LENGTH,
		use_compression: bool = True,
		feature_norm: str = "cmvn",
	) -> None:
		self.target_sr = target_sr
		self.target_db = target_db
		self.n_mfcc = n_mfcc
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.use_compression = use_compression
		self.feature_norm = feature_norm

	def normalize_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
		"""Resample -> RMS normalize -> optional compression."""
		x = _to_float_mono(audio)
		if sample_rate != self.target_sr:
			x = standardize_sample_rate(x, sample_rate, self.target_sr)
		x = normalize_rms(x, self.target_db)
		if self.use_compression:
			x = dynamic_range_compress(x)
		return x

	def normalize_fft_features(self, fft_magnitudes: np.ndarray) -> np.ndarray:
		log_features = log_scale_features(fft_magnitudes)
		return zscore_normalize(log_features)

	def extract_and_normalize_features(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
		"""Normalize waveform then extract MFCCs and apply selected normalization."""
		try:
			import librosa
		except ImportError as exc:
			raise RuntimeError("librosa is required. Install: pip install librosa") from exc

		audio_norm = self.normalize_audio(audio, sample_rate)
		mfccs = librosa.feature.mfcc(
			y=audio_norm,
			sr=self.target_sr,
			n_mfcc=self.n_mfcc,
			n_fft=self.n_fft,
			hop_length=self.hop_length,
		).T
		return self._apply_feature_norm(mfccs)

	def _apply_feature_norm(self, features: np.ndarray) -> np.ndarray:
		if self.feature_norm == "cmvn":
			return cmvn(features)
		if self.feature_norm == "zscore":
			return zscore_normalize(features)
		if self.feature_norm == "minmax":
			return minmax_normalize(features)
		return features.astype(np.float32)

	def prepare_model_input(
		self,
		audio: np.ndarray,
		sample_rate: int,
		fft_magnitudes: np.ndarray | None = None,
	) -> np.ndarray:
		"""Return a fixed-size feature vector for model input."""
		if fft_magnitudes is not None:
			features = self.normalize_fft_features(fft_magnitudes)
		else:
			features = self.extract_and_normalize_features(audio, sample_rate)
		return features.mean(axis=0)

	def process_file(self, path: str | Path) -> np.ndarray:
		"""Load one wav and return normalized model input vector."""
		sr, audio = wavfile.read(str(path))
		return self.prepare_model_input(audio, int(sr))

	def process_batch(self, paths: list[str | Path]) -> np.ndarray:
		"""Process multiple wav paths into one feature matrix."""
		return np.stack([self.process_file(p) for p in paths], axis=0)

	def process_stream(self, audio_chunks: list[tuple[np.ndarray, int]]):
		"""Yield normalized vectors from an iterable of (audio, sample_rate)."""
		for audio, sr in audio_chunks:
			yield self.prepare_model_input(audio, int(sr))

