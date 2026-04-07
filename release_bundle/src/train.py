from __future__ import annotations

import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from .normalize import NormalizationPipeline
except ImportError:
    from normalize import NormalizationPipeline


# ── constants ────────────────────────────────────────────────────────────────

TARGET_SR = 16000
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512

# number of MFCC frames per training segment (~2.7s at 16 kHz, hop=512)
SEGMENT_FRAMES = 128


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """
    Resolve a training/inference device.

    Accepted string values:
    - "auto": use CUDA when available, otherwise CPU
    - "cuda": require CUDA and raise if unavailable
    - "cpu": force CPU
    """
    if isinstance(device, torch.device):
        return device

    requested = "auto" if device is None else str(device).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available. "
                "Install a CUDA-enabled PyTorch build and confirm your NVIDIA drivers."
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device '{device}'. Use one of: auto, cuda, cpu.")


# ── config ───────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # model
    input_dim: int = N_MFCC
    hidden_dim: int = 256
    num_layers: int = 3
    embedding_dim: int = 256
    # training batch structure (N speakers × M utterances)
    n_speakers: int = 8
    n_utterances: int = 5
    # optimisation
    learning_rate: float = 1e-4
    n_epochs: int = 50
    grad_clip: float = 3.0
    # data
    segment_frames: int = SEGMENT_FRAMES
    min_utterances: int = 5        # speakers below this are excluded
    random_state: int = 42


# ── MFCC extraction ──────────────────────────────────────────────────────────

_pipeline = NormalizationPipeline(target_sr=TARGET_SR, n_mfcc=N_MFCC)


def _load_mfcc(path: str | Path, segment_frames: int) -> np.ndarray:
    """
    Load an audio file, extract normalised MFCC sequence, and return a
    fixed-length segment of shape (segment_frames, N_MFCC).

    Longer clips are randomly cropped; shorter ones are zero-padded.
    """
    import librosa

    audio, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    audio = _pipeline.normalize_audio(audio, sample_rate=TARGET_SR)
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    ).T  # (T, 13)

    T = mfccs.shape[0]
    if T >= segment_frames:
        start = random.randint(0, T - segment_frames)
        mfccs = mfccs[start : start + segment_frames]
    else:
        pad = np.zeros((segment_frames - T, N_MFCC), dtype=np.float32)
        mfccs = np.concatenate([mfccs, pad], axis=0)

    return mfccs.astype(np.float32)


# ── dataset ──────────────────────────────────────────────────────────────────

def build_speaker_map(
    csv_path: str | Path,
    audio_dir: str | Path,
    min_utterances: int = 5,
) -> dict[str, list[Path]]:
    """
    Read a Common Voice CSV and return {speaker_id: [list_of_audio_paths]}.

    Speakers with fewer than min_utterances clips are excluded.
    """
    import pandas as pd

    df = pd.read_csv(str(csv_path))
    audio_dir = Path(audio_dir)

    speaker_map: dict[str, list[Path]] = {}
    for client_id, group in df.groupby("client_id"):
        paths = [audio_dir / row for row in group["path"].tolist()]
        existing = [p for p in paths if p.exists()]
        if len(existing) >= min_utterances:
            speaker_map[str(client_id)] = existing

    return speaker_map


class SpeakerDataset(Dataset):
    """
    Samples batches of N speakers × M utterances for GE2E training.

    Each __getitem__ call returns one (N*M, segment_frames, N_MFCC) tensor.
    """

    def __init__(
        self,
        speaker_map: dict[str, list[Path]],
        config: TrainConfig,
        steps_per_epoch: int = 100,
    ) -> None:
        self.speakers = list(speaker_map.keys())
        self.paths = speaker_map
        self.config = config
        self.steps_per_epoch = steps_per_epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, _: int) -> torch.Tensor:
        N, M = self.config.n_speakers, self.config.n_utterances
        chosen = random.sample(self.speakers, N)

        segments = []
        for spk in chosen:
            files = random.choices(self.paths[spk], k=M)
            for f in files:
                seg = _load_mfcc(f, self.config.segment_frames)
                segments.append(seg)

        # (N*M, segment_frames, N_MFCC)
        return torch.from_numpy(np.stack(segments, axis=0))


# ── model ────────────────────────────────────────────────────────────────────

class LSTMEmbedder(nn.Module):
    """
    LSTM-based speaker encoder.

    Processes a sequence of MFCC frames and produces a single L2-normalised
    speaker embedding — the d-vector used for verification.
    """

    def __init__(self, config: TrainConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(config.hidden_dim, config.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x: (batch, T, input_dim)
        returns: (batch, embedding_dim) — L2 normalised embeddings
        """
        # only the final hidden state is used as the utterance embedding
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]                              # (batch, hidden_dim)
        emb = self.projection(h_last)                  # (batch, embedding_dim)
        return F.normalize(emb, p=2, dim=1)


# ── GE2E loss ─────────────────────────────────────────────────────────────────

class GE2ELoss(nn.Module):
    """
    Generalized End-to-End loss for speaker verification.

    Trains the embedding space so same-speaker embeddings cluster together
    and different-speaker embeddings are pushed apart, using cosine similarity
    with learnable scale (w) and bias (b).

    Reference: Wan et al., "Generalized End-to-End Loss for Speaker
    Verification", ICASSP 2018.
    """

    def __init__(self) -> None:
        super().__init__()
        # initialise w > 0 and b < 0 as recommended in the paper
        self.w = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(-5.0))

    def forward(self, embeddings: torch.Tensor, n_speakers: int, n_utterances: int) -> torch.Tensor:
        """
        Compute GE2E softmax loss.

        embeddings: (N*M, D) — L2 normalised speaker embeddings.
        n_speakers:  N — number of speakers in this batch.
        n_utterances: M — utterances per speaker.
        """
        N, M = n_speakers, n_utterances
        D = embeddings.shape[1]

        e = embeddings.view(N, M, D)             # (N, M, D)

        # full speaker centroids (N, D) — normalised mean of M utterances
        centroids = F.normalize(e.mean(dim=1), dim=1)   # (N, D)

        # similarity matrix: S[n, m, k] = w * cos(e[n,m], centroid[k]) + b
        # start from full-centroid similarities (N, M, N)
        S = torch.einsum("nmd,kd->nmk", e, centroids)

        # for each utterance, replace its own-speaker similarity with the
        # leave-one-out centroid (excludes that utterance from the centroid)
        for n in range(N):
            for m in range(M):
                c_loo = (centroids[n] * M - e[n, m]) / (M - 1)
                c_loo = F.normalize(c_loo, dim=0)
                S[n, m, n] = torch.dot(e[n, m], c_loo)

        S = self.w * S + self.b                    # (N, M, N) — scaled

        # softmax loss: target for utterance (n, m) is class n
        S_flat = S.view(N * M, N)
        targets = torch.arange(N, device=embeddings.device).repeat_interleave(M)
        return F.cross_entropy(S_flat, targets)


# ── training loop ─────────────────────────────────────────────────────────────

@dataclass
class TrainHistory:
    loss: list[float] = field(default_factory=list)


def train(
    speaker_map: dict[str, list[Path]],
    config: TrainConfig | None = None,
    steps_per_epoch: int = 100,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> tuple[LSTMEmbedder, TrainHistory]:
    """
    Train the LSTM speaker encoder with GE2E loss.

    Parameters
    ----------
    speaker_map:     {speaker_id: [list of audio paths]} — see build_speaker_map().
    config:          hyperparameters; uses defaults if None.
    steps_per_epoch: number of random batches sampled per epoch.
    verbose:         print per-epoch loss when True.

    Returns
    -------
    model:   trained LSTMEmbedder ready for inference.
    history: per-epoch average loss.
    """
    cfg = config or TrainConfig()
    random.seed(cfg.random_state)
    torch.manual_seed(cfg.random_state)

    device = resolve_device(device)
    if verbose:
        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    model = LSTMEmbedder(cfg).to(device)
    loss_fn = GE2ELoss().to(device)

    # GE2E paper recommends separate learning rates for model vs loss params
    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": cfg.learning_rate},
            {"params": loss_fn.parameters(), "lr": cfg.learning_rate * 0.01},
        ]
    )

    dataset = SpeakerDataset(speaker_map, cfg, steps_per_epoch=steps_per_epoch)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    history = TrainHistory()

    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            # batch: (1, N*M, T, input_dim) — squeeze the DataLoader batch dim
            x = batch.squeeze(0).to(device)            # (N*M, T, input_dim)

            # forward pass: encode each utterance to an embedding
            embeddings = model(x)                       # (N*M, embedding_dim)

            # loss computation
            loss = loss_fn(embeddings, cfg.n_speakers, cfg.n_utterances)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            # clamp w > 0 as the paper requires
            with torch.no_grad():
                loss_fn.w.clamp_(min=1e-6)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        history.loss.append(avg_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"epoch {epoch:3d}/{cfg.n_epochs}  loss={avg_loss:.4f}")

    return model, history


# ── inference helpers ─────────────────────────────────────────────────────────

def embed(
    model: LSTMEmbedder,
    paths: list[str | Path],
    config: TrainConfig | None = None,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """
    Encode a list of audio files and return their averaged embedding.

    Passing multiple utterances from the same speaker and averaging gives
    a more stable speaker template (d-vector).

    Returns: (embedding_dim,) numpy array.
    """
    cfg = config or TrainConfig()
    device = resolve_device(device)

    model.eval()
    segments = []
    with torch.no_grad():
        for p in paths:
            seg = _load_mfcc(p, cfg.segment_frames)
            x = torch.from_numpy(seg).unsqueeze(0).to(device)   # (1, T, 13)
            emb = model(x).squeeze(0).cpu().numpy()              # (D,)
            segments.append(emb)

    return np.mean(segments, axis=0).astype(np.float32)


def verify(
    model: LSTMEmbedder,
    enrollment_paths: list[str | Path],
    test_path: str | Path,
    threshold: float = 0.5,
    config: TrainConfig | None = None,
    device: str | torch.device | None = None,
) -> tuple[bool, float]:
    """
    Verify whether a test utterance matches the enrolled speaker.

    enrollment_paths: 1+ wav/mp3 files from the enrolled speaker.
    test_path:        wav/mp3 file to verify.
    threshold:        cosine similarity threshold; ≥ threshold → accepted.

    Returns (accepted, cosine_similarity_score).
    """
    cfg = config or TrainConfig()
    device = resolve_device(device)

    template = embed(model, enrollment_paths, cfg, device)
    test_emb = embed(model, [test_path], cfg, device)

    score = float(
        np.dot(template, test_emb)
        / (np.linalg.norm(template) * np.linalg.norm(test_emb) + 1e-8)
    )
    return score >= threshold, score


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_model(model: LSTMEmbedder, path: str | Path) -> None:
    """Save model weights and config to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.lstm.input_size}, str(path))


def load_model(path: str | Path, config: TrainConfig | None = None) -> LSTMEmbedder:
    """Restore a model saved by save_model()."""
    cfg = config or TrainConfig()
    state = torch.load(str(path), map_location="cpu")
    model = LSTMEmbedder(cfg)
    model.load_state_dict(state["state_dict"])
    return model
