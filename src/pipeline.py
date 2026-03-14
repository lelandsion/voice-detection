from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
	from .audio import MicrophoneRecorder
	from .feature_fft import extract_features_from_audio as extract_fft_features
	from .feature_startfreq import extract_features_from_audio as extract_startfreq_features
	from .model import VoiceModel
	from .normalize import NormalizationPipeline
	from .vad import EnergyVAD, detect_start_frequency
except ImportError:
	from audio import MicrophoneRecorder
	from feature_fft import extract_features_from_audio as extract_fft_features
	from feature_startfreq import extract_features_from_audio as extract_startfreq_features
	from model import VoiceModel
	from normalize import NormalizationPipeline
	from vad import EnergyVAD, detect_start_frequency


@dataclass
class PipelineConfig:
	sample_rate: int = 16000
	min_speech_ratio: float = 0.03
	include_fft_features: bool = True
	include_startfreq_features: bool = True
	include_norm_vector: bool = True


@dataclass
class PipelineResult:
	source: str
	sample_rate: int
	duration_s: float
	has_speech: bool
	speech_ratio: float
	start_time_s: float
	start_freq_hz: float
	feature_dict: dict[str, float]
	feature_vector: np.ndarray
	model_prediction: Any | None = None
	model_score: float | None = None
	model_accept: int | None = None
	recording_path: str | None = None


class VoiceVerificationPipeline:
	"""Orchestrates recording, VAD, preprocessing, feature extraction, and optional inference."""

	def __init__(
		self,
		config: PipelineConfig | None = None,
		recorder: MicrophoneRecorder | None = None,
		vad: EnergyVAD | None = None,
		normalizer: NormalizationPipeline | None = None,
		model: VoiceModel | None = None,
	) -> None:
		self.config = config or PipelineConfig()
		self.recorder = recorder or MicrophoneRecorder(sample_rate=self.config.sample_rate)
		self.vad = vad or EnergyVAD(sample_rate=self.config.sample_rate)
		self.normalizer = normalizer or NormalizationPipeline(target_sr=self.config.sample_rate)
		self.model = model

	def set_model(self, model: VoiceModel) -> None:
		self.model = model

	def load_model(self, checkpoint_dir: str | Path, model_name: str = "voice_model") -> None:
		self.model = VoiceModel.load(checkpoint_dir=checkpoint_dir, model_name=model_name)

	def save_model(self, checkpoint_dir: str | Path) -> None:
		if self.model is None:
			raise RuntimeError("No model attached.")
		self.model.save(checkpoint_dir)

	def train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
		if self.model is None:
			raise RuntimeError("No model attached. Create/set a VoiceModel first.")
		self.model.train(x_train, y_train)

	@staticmethod
	def _flatten_startfreq(prefix: str, row: dict[str, float | int | str]) -> dict[str, float]:
		out: dict[str, float] = {}
		for key, val in row.items():
			if isinstance(val, (int, float, np.floating, np.integer)):
				out[f"{prefix}{key}"] = float(val)
		return out

	@staticmethod
	def _to_sorted_vector(feature_dict: dict[str, float]) -> np.ndarray:
		keys = sorted(feature_dict.keys())
		return np.array([feature_dict[k] for k in keys], dtype=np.float32)

	def _extract_feature_bundle(self, audio: np.ndarray, sample_rate: int) -> tuple[dict[str, float], float, float]:
		vad_result = self.vad.detect(audio=audio, sample_rate=sample_rate)
		has_speech = vad_result.speech_ratio >= self.config.min_speech_ratio
		working_audio = vad_result.trimmed_audio if has_speech else audio

		startfreq = detect_start_frequency(working_audio, sample_rate=self.config.sample_rate)
		normalized_audio = self.normalizer.normalize_audio(working_audio, sample_rate=self.config.sample_rate)

		features: dict[str, float] = {
			"vad_speech_ratio": float(vad_result.speech_ratio),
			"vad_threshold": float(vad_result.threshold),
			"start_time_s": float(startfreq.start_time_s),
			"start_freq_hz": float(startfreq.start_freq_hz),
		}

		if self.config.include_norm_vector:
			norm_vector = self.normalizer.prepare_model_input(normalized_audio, sample_rate=self.config.sample_rate)
			for i, val in enumerate(norm_vector):
				features[f"norm_vec_{i:03d}"] = float(val)

		if self.config.include_fft_features:
			fft_feats = extract_fft_features(audio=normalized_audio, sample_rate=self.config.sample_rate)
			if fft_feats:
				for key, val in fft_feats.items():
					features[f"fft_{key}"] = float(val)

		if self.config.include_startfreq_features:
			sf = extract_startfreq_features(audio=normalized_audio, sample_rate=self.config.sample_rate)
			sf_row = {
				"sample_rate": sf.sample_rate,
				"duration_s": sf.duration_s,
				"start_time_s": sf.start_time_s,
				"start_freq_hz": sf.start_freq_hz,
				"rms": sf.rms,
				"zero_crossing_rate": sf.zero_crossing_rate,
			}
			for idx, val in enumerate(sf.fft_0_128):
				sf_row[f"fft_bin_{idx:03d}"] = float(val)
			features.update(self._flatten_startfreq("startfreq_", sf_row))

		return features, float(vad_result.speech_ratio), float(startfreq.start_freq_hz)

	def _maybe_infer(self, feature_vector: np.ndarray) -> tuple[Any | None, float | None, int | None]:
		if self.model is None:
			return None, None, None

		model_input = feature_vector.reshape(1, -1)
		prediction = self.model.predict(model_input)

		score: float | None = None
		accept: int | None = None
		try:
			score_arr = np.asarray(self.model.predict_scores(model_input)).reshape(-1)
			score = float(score_arr[0]) if score_arr.size else None
			if score is not None:
				accept = int(self.model.verify(model_input)[0])
		except Exception:
			score = None
			accept = None

		pred_val: Any
		pred_flat = np.asarray(prediction).reshape(-1)
		pred_val = pred_flat[0] if pred_flat.size else None
		return pred_val, score, accept

	def process_audio(self, audio: np.ndarray, sample_rate: int, source: str = "memory") -> PipelineResult:
		"""Run full pipeline for in-memory audio."""
		features, speech_ratio, start_freq_hz = self._extract_feature_bundle(audio=audio, sample_rate=sample_rate)
		feature_vector = self._to_sorted_vector(features)
		pred, score, accept = self._maybe_infer(feature_vector)

		return PipelineResult(
			source=source,
			sample_rate=self.config.sample_rate,
			duration_s=float(len(audio) / max(sample_rate, 1)),
			has_speech=speech_ratio >= self.config.min_speech_ratio,
			speech_ratio=speech_ratio,
			start_time_s=float(features.get("start_time_s", 0.0)),
			start_freq_hz=start_freq_hz,
			feature_dict=features,
			feature_vector=feature_vector,
			model_prediction=pred,
			model_score=score,
			model_accept=accept,
			recording_path=None,
		)

	def process_file(self, wav_path: str | Path) -> PipelineResult:
		"""Run full pipeline for one wav file."""
		from scipy.io import wavfile

		sr, audio = wavfile.read(str(wav_path))
		result = self.process_audio(audio=audio, sample_rate=int(sr), source=str(wav_path))
		result.recording_path = str(wav_path)
		return result

	def record_and_process(self, seconds: float, save_recording: bool = True) -> PipelineResult:
		"""Record from mic then run full pipeline."""
		rec = self.recorder.record_for(seconds=seconds, save=save_recording)
		result = self.process_audio(audio=rec.audio, sample_rate=rec.sample_rate, source="microphone")
		result.duration_s = rec.duration
		result.recording_path = str(rec.saved_path) if rec.saved_path else None
		return result

	def process_directory(self, input_dir: str | Path) -> pd.DataFrame:
		"""Process all wav files in a directory and return a summary DataFrame."""
		input_dir = Path(input_dir)
		wav_paths = sorted(input_dir.glob("*.wav"))
		if not wav_paths:
			raise FileNotFoundError(f"No .wav files found in {input_dir}")

		rows: list[dict[str, Any]] = []
		for wav_path in wav_paths:
			out = self.process_file(wav_path)
			rows.append(
				{
					"source": out.source,
					"duration_s": out.duration_s,
					"has_speech": out.has_speech,
					"speech_ratio": out.speech_ratio,
					"start_time_s": out.start_time_s,
					"start_freq_hz": out.start_freq_hz,
					"model_prediction": out.model_prediction,
					"model_score": out.model_score,
					"model_accept": out.model_accept,
					"n_features": int(len(out.feature_vector)),
				}
			)
		return pd.DataFrame(rows)

