from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

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
	"""Peak normalization only. Noise gate removed — VAD already trims silence."""
	peak = np.max(np.abs(audio))
	if peak < 1e-8:
		return audio.astype(np.float32)
	normed = audio / peak
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

	def record_vad(
		self,
		calibration_s: float = 1.0,
		rolling_window_ms: int = 250,
		hang_time_ms: int = 500,
		max_seconds: float = 10.0,
		min_speech_ms: int = 300,
		threshold_multiplier: float = 1.5,
		save: bool = True,
		file_prefix: str = "mic_vad",
		on_state_change: Callable[[str], None] | None = None,
	) -> RecordingResult:
		"""Record using energy-based VAD: calibrate mic gain, then capture speech only.

		Calibrates against 95th-percentile background noise energy in the first
		`calibration_s` seconds (handles wildly different hardware gain levels).
		Uses a rolling `rolling_window_ms` energy window so transient noise doesn't
		trigger recording. Holds recording for `hang_time_ms` after energy drops so
		brief pauses between words aren't cut off. Stops when post-speech silence
		exceeds hang time or `max_seconds` is reached.

		Calls `on_state_change` with one of: "calibrating", "waiting", "speaking",
		"hang", "done" as the VAD transitions between states.
		"""
		BLOCKSIZE = 512  # ~32 ms at 16 kHz — fixed for predictable window math
		sr = self.sample_rate

		calib_samples_needed = int(calibration_s * sr)
		window_size = max(1, int(rolling_window_ms / 1000 * sr / BLOCKSIZE))
		hang_blocks = max(1, int(hang_time_ms / 1000 * sr / BLOCKSIZE))
		min_speech_blocks = max(1, int(min_speech_ms / 1000 * sr / BLOCKSIZE))
		max_blocks = int(max_seconds * sr / BLOCKSIZE)

		pre_pad_blocks = max(1, int(80 / 1000 * sr / BLOCKSIZE))  # 80 ms pre-speech pad

		lock = threading.Lock()
		calib_energies: list[float] = []
		live_chunks: list[np.ndarray] = []
		energy_window: deque[float] = deque(maxlen=window_size)

		vad_state = ["calibrating"]
		threshold = [0.0]
		calib_samples_seen = [0]
		speech_block_count = [0]
		hang_remaining = [0]
		live_blocks = [0]
		speech_start_block = [0]  # index into live_chunks where speech onset occurred
		speech_detected = [False]
		done_event = threading.Event()

		def _notify(new_state: str) -> None:
			vad_state[0] = new_state
			if on_state_change is not None:
				on_state_change(new_state)

		def _callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:  # noqa: ANN001
			chunk = (indata[:, 0] if indata.ndim > 1 else indata.ravel()).astype(np.float32)
			rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0

			with lock:
				if vad_state[0] == "done":
					return

				# ── CALIBRATING: measure background noise level ──────────────
				if vad_state[0] == "calibrating":
					calib_energies.append(rms)
					calib_samples_seen[0] += len(chunk)
					if calib_samples_seen[0] >= calib_samples_needed:
						# 95th percentile guards against a single transient spike
						# skewing the noise floor estimate.
						noise_peak = float(np.percentile(calib_energies, 95))
						threshold[0] = max(noise_peak * threshold_multiplier, 1e-5)
						print(f"[VAD] noise_peak={noise_peak:.6f}  threshold={threshold[0]:.6f}")
						_notify("waiting")
					return

				# ── POST-CALIBRATION ─────────────────────────────────────────
				live_chunks.append(chunk)
				live_blocks[0] += 1
				energy_window.append(rms)
				rolling_energy = float(np.mean(energy_window))

				if vad_state[0] == "waiting":
					if rolling_energy > threshold[0]:
						speech_block_count[0] = 1
						speech_detected[0] = True
						# Record onset with pre-pad so we don't clip the attack
						speech_start_block[0] = max(0, live_blocks[0] - 1 - pre_pad_blocks)
						_notify("speaking")

				elif vad_state[0] == "speaking":
					speech_block_count[0] += 1
					# Force hang if speech has been continuous too long (noisy mic floor)
					max_speech_blocks = max(1, int(max_seconds * 0.6 * sr / BLOCKSIZE))
					if rolling_energy <= threshold[0] or speech_block_count[0] >= max_speech_blocks:
						hang_remaining[0] = hang_blocks
						_notify("hang")

				elif vad_state[0] == "hang":
					if rolling_energy > threshold[0]:
						# Speech resumed — keep counting
						speech_block_count[0] += 1
						_notify("speaking")
					else:
						hang_remaining[0] -= 1
						if hang_remaining[0] <= 0:
							if speech_block_count[0] >= min_speech_blocks:
								_notify("done")
								done_event.set()
							else:
								# Too short — reset and wait for real speech
								speech_block_count[0] = 0
								energy_window.clear()
								_notify("waiting")

				if live_blocks[0] >= max_blocks and vad_state[0] != "done":
					_notify("done")
					done_event.set()

		with sd.InputStream(
			samplerate=sr,
			channels=self.channels,
			dtype="float32",
			blocksize=BLOCKSIZE,
			callback=_callback,
		):
			if on_state_change is not None:
				on_state_change("calibrating")
			done_event.wait()

		with lock:
			if not live_chunks or not speech_detected[0]:
				raise RuntimeError("No speech detected. Please speak clearly into the mic.")
			# Trim leading silence — only keep from speech onset (with pre-pad)
			speech_chunks = live_chunks[speech_start_block[0]:]
			audio = np.concatenate(speech_chunks if speech_chunks else live_chunks).astype(np.float32)

		duration = float(len(audio) / sr)
		rms_final = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
		quality = level_to_quality(rms_final)

		processed = preprocess_audio(audio, sr) if self.preprocess else audio
		saved_path: Path | None = None

		if save:
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			saved_path = self.output_dir / f"{file_prefix}_{timestamp}.wav"
			save_wav(processed, sr, saved_path)
			save_metadata(
				metadata_csv=self.metadata_csv,
				timestamp=timestamp,
				duration=duration,
				sample_rate=sr,
				path=str(saved_path),
			)

		return RecordingResult(
			audio=processed,
			sample_rate=sr,
			duration=duration,
			rms_level=rms_final,
			quality=quality,
			saved_path=saved_path,
		)

