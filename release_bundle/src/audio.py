from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sounddevice as sd
from scipy.io.wavfile import write as wav_write


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_OUTPUT_DIR = Path("data/raw/recordings")
DEFAULT_METADATA_CSV = Path("data/processed/recordings_metadata.csv")


def _ensure_paths(output_dir: Path, metadata_csv: Path) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	metadata_csv.parent.mkdir(parents=True, exist_ok=True)


def save_wav(audio_float32: np.ndarray, sample_rate: int, out_path: Path) -> Path:
	"""Save float32 audio in [-1, 1] to int16 wav format."""
	audio_clipped = np.clip(audio_float32, -1.0, 1.0)
	audio_int16 = (audio_clipped * 32767).astype(np.int16)
	wav_write(str(out_path), sample_rate, audio_int16)
	return out_path


def level_to_quality(level: float) -> str:
	"""Map RMS level to a simple recording quality label."""
	if level < 0.01:
		return "Too quiet"
	if level < 0.04:
		return "Good"
	if level < 0.12:
		return "Loud"
	return "Very loud / possible clipping"


def preprocess_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
	"""Apply simple noise gate and peak normalization."""
	noise_len = min(int(sample_rate * 0.5), len(audio))
	if noise_len == 0:
		return audio.astype(np.float32)

	noise_clip = audio[:noise_len]
	noise_std = np.std(noise_clip)
	threshold = noise_std * 1.5
	reduced = np.where(np.abs(audio) < threshold, 0.0, audio)
	normed = reduced / (np.max(np.abs(reduced)) + 1e-8)
	return normed.astype(np.float32)


def save_metadata(
	metadata_csv: Path,
	timestamp: str,
	duration: float,
	sample_rate: int,
	path: str,
) -> None:
	"""Append one recording row to metadata CSV."""
	row = {
		"timestamp": timestamp,
		"duration": duration,
		"sample_rate": sample_rate,
		"path": path,
	}

	if metadata_csv.exists():
		df = pd.read_csv(metadata_csv)
		df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
	else:
		df = pd.DataFrame([row])
	df.to_csv(metadata_csv, index=False)


@dataclass
class RecordingResult:
	"""Return object for one recording capture."""

	audio: np.ndarray
	sample_rate: int
	duration: float
	rms_level: float
	quality: str
	saved_path: Path | None


class MicrophoneRecorder:
	"""Reusable microphone recorder for pipeline integration."""

	def __init__(
		self,
		sample_rate: int = DEFAULT_SAMPLE_RATE,
		channels: int = DEFAULT_CHANNELS,
		output_dir: Path = DEFAULT_OUTPUT_DIR,
		metadata_csv: Path = DEFAULT_METADATA_CSV,
		preprocess: bool = True,
	) -> None:
		self.sample_rate = sample_rate
		self.channels = channels
		self.output_dir = output_dir
		self.metadata_csv = metadata_csv
		self.preprocess = preprocess

		self._frames: list[np.ndarray] = []
		self._latest_level: float = 0.0
		self._stream: sd.InputStream | None = None
		self._lock = threading.Lock()

		_ensure_paths(self.output_dir, self.metadata_csv)

	@property
	def latest_level(self) -> float:
		return self._latest_level

	@property
	def latest_quality(self) -> str:
		return level_to_quality(self._latest_level)

	def _audio_callback(self, indata, frames, time, status) -> None:  # noqa: ANN001
		del frames, time, status
		chunk = indata.copy()
		if self.channels == 1:
			chunk = chunk[:, 0]

		with self._lock:
			self._frames.append(chunk)
			self._latest_level = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0

	def start(self) -> None:
		if self._stream is not None:
			return
		self._frames = []
		self._latest_level = 0.0

		self._stream = sd.InputStream(
			samplerate=self.sample_rate,
			channels=self.channels,
			dtype="float32",
			callback=self._audio_callback,
		)
		self._stream.start()

	def stop(self, save: bool = True, file_prefix: str = "mic_monitor") -> RecordingResult:
		if self._stream is None:
			raise RuntimeError("Recorder is not running. Call start() first.")

		self._stream.stop()
		self._stream.close()
		self._stream = None

		with self._lock:
			if not self._frames:
				raise RuntimeError("No audio captured.")
			audio = np.concatenate(self._frames).astype(np.float32)

		duration = float(len(audio) / self.sample_rate)
		rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
		quality = level_to_quality(rms)

		processed = preprocess_audio(audio, self.sample_rate) if self.preprocess else audio
		saved_path: Path | None = None

		if save:
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			saved_path = self.output_dir / f"{file_prefix}_{timestamp}.wav"
			save_wav(processed, self.sample_rate, saved_path)
			save_metadata(
				metadata_csv=self.metadata_csv,
				timestamp=timestamp,
				duration=duration,
				sample_rate=self.sample_rate,
				path=str(saved_path),
			)

		return RecordingResult(
			audio=processed,
			sample_rate=self.sample_rate,
			duration=duration,
			rms_level=rms,
			quality=quality,
			saved_path=saved_path,
		)

	def record_for(self, seconds: float, save: bool = True, file_prefix: str = "mic_monitor") -> RecordingResult:
		"""Convenience API for pipeline use: start, wait, stop."""
		if seconds <= 0:
			raise ValueError("seconds must be > 0")
		self.start()
		sd.sleep(int(seconds * 1000))
		return self.stop(save=save, file_prefix=file_prefix)

