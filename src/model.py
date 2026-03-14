from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModelArtifacts:
	"""File paths for a saved model checkpoint."""

	model_path: Path
	metadata_path: Path


@dataclass
class ModelConfig:
	"""Config shared across model implementations."""

	name: str = "voice_model"
	version: str = "0.1.0"
	task: str = "speaker_verification"
	threshold: float = 0.5
	random_state: int = 42
	extra: dict[str, Any] = field(default_factory=dict)


class VoiceModel:
	"""
	Architecture-agnostic model wrapper.

	You can inject any model object that implements `fit` and `predict`
	(and optionally `predict_proba` / `decision_function`).
	"""

	def __init__(self, config: ModelConfig | None = None, model: Any | None = None) -> None:
		self.config = config or ModelConfig()
		self.model = model
		self.is_trained = False
		self.label_names: list[str] | None = None
		self.input_dim: int | None = None

	def set_model(self, model: Any) -> None:
		"""Attach a concrete model implementation later (e.g., sklearn, torch, xgboost)."""
		self.model = model
		self.is_trained = False

	def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
		"""Fit the currently attached model."""
		if self.model is None:
			raise RuntimeError("No model implementation attached. Call set_model(...) first.")
		if not hasattr(self.model, "fit"):
			raise TypeError("Attached model must implement fit(X, y).")

		x_train = np.asarray(x_train)
		y_train = np.asarray(y_train)
		if x_train.ndim != 2:
			raise ValueError("x_train must be 2D: [n_samples, n_features].")
		if len(x_train) != len(y_train):
			raise ValueError("x_train and y_train must have the same number of rows.")

		self.input_dim = int(x_train.shape[1])
		self.label_names = sorted({str(v) for v in y_train.tolist()})

		self.model.fit(x_train, y_train)
		self.is_trained = True

	def _require_ready(self) -> None:
		if self.model is None or not self.is_trained:
			raise RuntimeError("Model is not ready. Train it or load a checkpoint first.")

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""Predict labels for input feature matrix."""
		self._require_ready()
		model = self.model
		if model is None:
			raise RuntimeError("Model is not ready.")
		x = np.asarray(x)
		if x.ndim == 1:
			x = x.reshape(1, -1)
		if self.input_dim is not None and x.shape[1] != self.input_dim:
			raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[1]}.")
		return np.asarray(model.predict(x))

	def predict_scores(self, x: np.ndarray) -> np.ndarray:
		"""
		Return verification scores.
		Uses `predict_proba` if available, otherwise `decision_function`.
		"""
		self._require_ready()
		model = self.model
		if model is None:
			raise RuntimeError("Model is not ready.")
		x = np.asarray(x)
		if x.ndim == 1:
			x = x.reshape(1, -1)

		if hasattr(model, "predict_proba"):
			probs = np.asarray(model.predict_proba(x))
			if probs.ndim == 2 and probs.shape[1] >= 2:
				return probs[:, 1]
			return probs.squeeze()

		if hasattr(model, "decision_function"):
			return np.asarray(model.decision_function(x)).squeeze()

		raise RuntimeError("Model does not expose predict_proba or decision_function.")

	def verify(self, x: np.ndarray, threshold: float | None = None) -> np.ndarray:
		"""Return binary accept/reject decisions from scores."""
		scores = self.predict_scores(x)
		th = self.config.threshold if threshold is None else threshold
		return (scores >= th).astype(np.int32)

	def evaluate_accuracy(self, x: np.ndarray, y_true: np.ndarray) -> float:
		"""Simple accuracy helper for quick baselines."""
		y_pred = self.predict(x)
		y_true = np.asarray(y_true)
		if len(y_pred) != len(y_true):
			raise ValueError("Predictions and y_true must have same length.")
		return float(np.mean(y_pred == y_true))

	def save(self, checkpoint_dir: str | Path) -> ModelArtifacts:
		"""Save model pickle + metadata JSON."""
		self._require_ready()
		checkpoint_dir = Path(checkpoint_dir)
		checkpoint_dir.mkdir(parents=True, exist_ok=True)

		model_path = checkpoint_dir / f"{self.config.name}.pkl"
		metadata_path = checkpoint_dir / f"{self.config.name}.meta.json"

		with model_path.open("wb") as f:
			pickle.dump(self.model, f)

		metadata = {
			"name": self.config.name,
			"version": self.config.version,
			"task": self.config.task,
			"threshold": self.config.threshold,
			"random_state": self.config.random_state,
			"extra": self.config.extra,
			"input_dim": self.input_dim,
			"label_names": self.label_names,
			"saved_at_utc": datetime.now(timezone.utc).isoformat(),
		}
		with metadata_path.open("w", encoding="utf-8") as f:
			json.dump(metadata, f, indent=2)

		return ModelArtifacts(model_path=model_path, metadata_path=metadata_path)

	@classmethod
	def load(cls, checkpoint_dir: str | Path, model_name: str = "voice_model") -> "VoiceModel":
		"""Load model + metadata checkpoint created by save()."""
		checkpoint_dir = Path(checkpoint_dir)
		model_path = checkpoint_dir / f"{model_name}.pkl"
		metadata_path = checkpoint_dir / f"{model_name}.meta.json"

		if not model_path.exists():
			raise FileNotFoundError(f"Missing model checkpoint: {model_path}")
		if not metadata_path.exists():
			raise FileNotFoundError(f"Missing model metadata: {metadata_path}")

		with metadata_path.open("r", encoding="utf-8") as f:
			meta = json.load(f)

		with model_path.open("rb") as f:
			model = pickle.load(f)

		config = ModelConfig(
			name=meta.get("name", model_name),
			version=meta.get("version", "0.1.0"),
			task=meta.get("task", "speaker_verification"),
			threshold=float(meta.get("threshold", 0.5)),
			random_state=int(meta.get("random_state", 42)),
			extra=meta.get("extra", {}),
		)

		instance = cls(config=config, model=model)
		instance.input_dim = meta.get("input_dim")
		instance.label_names = meta.get("label_names")
		instance.is_trained = True
		return instance

