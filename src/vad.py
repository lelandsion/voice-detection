from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TARGET_SAMPLE_RATE = 16000


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
	x = np.asarray(audio)
	if x.dtype.kind in {"i", "u"}:
		max_val = np.iinfo(x.dtype).max
		x = x.astype(np.float32) / float(max_val)
	else:
		x = x.astype(np.float32)

	if x.ndim > 1:
		x = np.mean(x, axis=1)
	return np.clip(x, -1.0, 1.0)


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
	if src_sr == dst_sr:
		return audio.astype(np.float32)
	if len(audio) < 2:
		return audio.astype(np.float32)

	src_x = np.arange(len(audio), dtype=np.float32)
	target_len = int(round(len(audio) * (dst_sr / src_sr)))
	dst_x = np.linspace(0, len(audio) - 1, target_len, dtype=np.float32)
	return np.interp(dst_x, src_x, audio).astype(np.float32)


def _frame_rms(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
	if len(audio) < frame_len:
		pad = np.pad(audio, (0, frame_len - len(audio)))
		return np.array([np.sqrt(np.mean(np.square(pad)))], dtype=np.float32)

	vals: list[float] = []
	for start in range(0, len(audio) - frame_len + 1, hop_len):
		frame = audio[start : start + frame_len]
		vals.append(float(np.sqrt(np.mean(np.square(frame)))))
	return np.array(vals, dtype=np.float32)


@dataclass
class VADResult:
	sample_rate: int
	speech_mask: np.ndarray
	threshold: float
	start_idx: int
	end_idx: int
	start_time_s: float
	end_time_s: float
	speech_ratio: float
	trimmed_audio: np.ndarray


@dataclass
class StartFreqResult:
	start_idx: int
	start_time_s: float
	start_freq_hz: float


class EnergyVAD:
	"""Simple energy-based VAD for noisy short-utterance recordings."""

	def __init__(
		self,
		sample_rate: int = TARGET_SAMPLE_RATE,
		frame_ms: int = 20,
		hop_ms: int = 10,
		threshold_multiplier: float = 4.0,
		min_threshold: float = 1e-5,
		pre_pad_ms: int = 60,
		post_pad_ms: int = 120,
	) -> None:
		self.sample_rate = sample_rate
		self.frame_ms = frame_ms
		self.hop_ms = hop_ms
		self.threshold_multiplier = threshold_multiplier
		self.min_threshold = min_threshold
		self.pre_pad_ms = pre_pad_ms
		self.post_pad_ms = post_pad_ms

	def _energy_curve(self, audio: np.ndarray) -> np.ndarray:
		frame_len = max(1, int(self.sample_rate * self.frame_ms / 1000))
		return np.convolve(audio ** 2, np.ones(frame_len, dtype=np.float32), mode="same") / frame_len

	def detect(self, audio: np.ndarray, sample_rate: int) -> VADResult:
		"""Detect voiced region and return trimmed segment with mask."""
		x = _to_mono_float32(audio)
		if sample_rate != self.sample_rate:
			x = _resample_linear(x, sample_rate, self.sample_rate)

		if len(x) == 0:
			return VADResult(
				sample_rate=self.sample_rate,
				speech_mask=np.zeros(0, dtype=bool),
				threshold=self.min_threshold,
				start_idx=0,
				end_idx=0,
				start_time_s=0.0,
				end_time_s=0.0,
				speech_ratio=0.0,
				trimmed_audio=x,
			)

		energy = self._energy_curve(x)
		baseline_window = min(len(energy), max(1, self.sample_rate // 2))
		baseline = float(np.median(energy[:baseline_window]))
		threshold = max(baseline * self.threshold_multiplier, self.min_threshold)

		speech_mask = energy > threshold
		if not np.any(speech_mask):
			return VADResult(
				sample_rate=self.sample_rate,
				speech_mask=speech_mask,
				threshold=threshold,
				start_idx=0,
				end_idx=len(x),
				start_time_s=0.0,
				end_time_s=float(len(x) / self.sample_rate),
				speech_ratio=0.0,
				trimmed_audio=x,
			)

		idx = np.where(speech_mask)[0]
		pre_pad = int(self.sample_rate * self.pre_pad_ms / 1000)
		post_pad = int(self.sample_rate * self.post_pad_ms / 1000)
		start_idx = max(0, int(idx[0]) - pre_pad)
		end_idx = min(len(x), int(idx[-1]) + 1 + post_pad)
		trimmed = x[start_idx:end_idx]

		return VADResult(
			sample_rate=self.sample_rate,
			speech_mask=speech_mask,
			threshold=threshold,
			start_idx=start_idx,
			end_idx=end_idx,
			start_time_s=float(start_idx / self.sample_rate),
			end_time_s=float(end_idx / self.sample_rate),
			speech_ratio=float(np.mean(speech_mask)),
			trimmed_audio=trimmed,
		)

	def has_speech(self, audio: np.ndarray, sample_rate: int, min_ratio: float = 0.05) -> bool:
		"""Quick boolean VAD decision."""
		result = self.detect(audio, sample_rate)
		return result.speech_ratio >= min_ratio


def find_start_index(audio: np.ndarray, sample_rate: int, frame_ms: int = 20) -> int:
	"""Find first index where smoothed energy exceeds threshold."""
	x = _to_mono_float32(audio)
	frame_len = max(1, int(sample_rate * frame_ms / 1000))
	if len(x) <= frame_len:
		return 0

	energy = np.convolve(x ** 2, np.ones(frame_len, dtype=np.float32), mode="same") / frame_len
	baseline = np.median(energy[: min(len(energy), sample_rate // 2 or 1)])
	threshold = max(baseline * 4.0, 1e-5)
	above = np.where(energy > threshold)[0]
	return int(above[0]) if above.size else 0


def start_frequency(audio: np.ndarray, sample_rate: int, start_idx: int, window_ms: int = 80) -> float:
	"""Estimate dominant start frequency near the detected speech onset."""
	x = _to_mono_float32(audio)
	win_len = max(64, int(sample_rate * window_ms / 1000))
	end_idx = min(len(x), start_idx + win_len)
	segment = x[start_idx:end_idx]
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


def detect_start_frequency(audio: np.ndarray, sample_rate: int, frame_ms: int = 20, window_ms: int = 80) -> StartFreqResult:
	"""Run onset detection and start-frequency estimation in one call."""
	start_idx = find_start_index(audio=audio, sample_rate=sample_rate, frame_ms=frame_ms)
	freq = start_frequency(audio=audio, sample_rate=sample_rate, start_idx=start_idx, window_ms=window_ms)
	return StartFreqResult(
		start_idx=start_idx,
		start_time_s=float(start_idx / sample_rate),
		start_freq_hz=freq,
	)

