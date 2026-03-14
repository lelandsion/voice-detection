"""
Audio normalization for voice verification
Lilly Ko 

Handles preprocessing from raw audio to model-ready MFCC features.
Works with raw .wav files from the recorder, or FFT magnitude output
once that step is done.

Usage:
    from normalize import NormalizationPipeline
    pipeline = NormalizationPipeline()
    features = pipeline.prepare_model_input(audio, sample_rate=16000)
    prediction = svm.predict(features.reshape(1, -1))
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile

# keep these in sync with voice_input.ipynb
TARGET_SR = 16000
TARGET_DB = -20.0
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512


# PI1 — amplitude and sample rate standardization

def standardize_sample_rate(audio, orig_sr, target_sr=TARGET_SR):
    """Resample audio to target_sr using polyphase resampling."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    gcd = np.gcd(orig_sr, target_sr)
    resampled = scipy_signal.resample_poly(audio, target_sr // gcd, orig_sr // gcd)
    return resampled.astype(np.float32)


def normalize_rms(audio, target_db=TARGET_DB):
    """Scale audio to a target RMS level in dBFS (default -20)."""
    rms = np.sqrt(np.mean(np.square(audio)))
    if rms < 1e-8:
        return audio.astype(np.float32)
    target_rms = 10.0 ** (target_db / 20.0)
    return (audio * (target_rms / rms)).astype(np.float32)


def normalize_peak(audio):
    """Scale so the peak absolute value equals 1.0."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio.astype(np.float32)
    return (audio / peak).astype(np.float32)


# PI2 — dynamic range compression and feature scaling

def dynamic_range_compress(audio, threshold=0.3, ratio=4.0, makeup_gain=1.2):
    """
    Apply dynamic range compression. Samples above threshold are
    attenuated by ratio (4:1 here). makeup_gain offsets the level drop.
    """
    audio = audio.astype(np.float32)
    abs_a = np.abs(audio)
    compressed_abs = np.where(
        abs_a > threshold,
        threshold + (abs_a - threshold) / ratio,
        abs_a,
    )
    safe_abs_a = np.where(abs_a > 1e-8, abs_a, 1.0)  # avoid div-by-zero before masking
    gain = np.where(abs_a > 1e-8, compressed_abs / safe_abs_a, 1.0)
    return np.clip(audio * gain * makeup_gain, -1.0, 1.0).astype(np.float32)


def log_scale_features(features, floor_db=-80.0):
    """Convert spectral magnitudes to dB. Clips at floor_db to avoid -inf."""
    features = np.maximum(features, 1e-8)
    db = 20.0 * np.log10(features)
    return np.maximum(db, floor_db).astype(np.float32)


def minmax_normalize(features):
    """Scale features to [0, 1]."""
    lo, hi = features.min(), features.max()
    if hi - lo < 1e-8:
        return np.zeros_like(features, dtype=np.float32)
    return ((features - lo) / (hi - lo)).astype(np.float32)


def zscore_normalize(features):
    """Zero mean, unit variance normalization."""
    mu, sigma = features.mean(), features.std()
    if sigma < 1e-8:
        return np.zeros_like(features, dtype=np.float32)
    return ((features - mu) / sigma).astype(np.float32)


def cmvn(features):
    """
    Cepstral Mean and Variance Normalization.
    Normalizes each MFCC dimension independently — standard for speaker
    verification, also helps remove mic/channel differences.
    Input: (n_frames, n_features)
    """
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    mu = features.mean(axis=0, keepdims=True)
    sigma = features.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return ((features - mu) / sigma).astype(np.float32)


# PI3 — distribution comparison for validation + see normalize_pipeline.ipynb

def distribution_stats(data):
    """Basic descriptive stats over a flattened array."""
    flat = data.flatten()
    return {
        'mean': float(np.mean(flat)),
        'std': float(np.std(flat)),
        'min': float(np.min(flat)),
        'max': float(np.max(flat)),
        'p25': float(np.percentile(flat, 25)),
        'p75': float(np.percentile(flat, 75)),
        'rms': float(np.sqrt(np.mean(np.square(flat)))),
        'range': float(np.max(flat) - np.min(flat)),
    }


def compare_distributions(before, after):
    """Compare stats before and after normalization. Returns before/after/delta."""
    b = distribution_stats(before)
    a = distribution_stats(after)
    delta = {k: round(a[k] - b[k], 6) for k in b}
    return {'before': b, 'after': a, 'delta': delta}


# PI4 — consistency evaluation + see normalize_pipeline.ipynb

def evaluate_consistency(feature_list):
    """
    Check how consistent feature vectors are across recordings.
    Computes variance per MFCC dimension across recordings — lower
    means the model sees more uniform inputs, which helps training.
    """
    summaries = np.stack([f.mean(axis=0) for f in feature_list], axis=0)
    per_dim_var = summaries.var(axis=0)
    return {
        'per_dim_variance': per_dim_var,
        'mean_cross_variance': float(per_dim_var.mean()),
        'max_cross_variance': float(per_dim_var.max()),
        'n_recordings': len(feature_list),
    }


# PI5 — full pipeline

class NormalizationPipeline:
    """
    End-to-end normalization pipeline for speaker verification.

    Takes raw audio from the recorder (or FFT magnitudes once that step is
    done) and outputs a normalized feature vector ready for the SVM.

    To plug in FFT output later:
        features = pipeline.prepare_model_input(audio, sr, fft_magnitudes=fft_out)
    """

    def __init__(
        self,
        target_sr=TARGET_SR,
        target_db=TARGET_DB,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        use_compression=True,
        feature_norm='cmvn',
    ):
        self.target_sr = target_sr
        self.target_db = target_db
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_compression = use_compression
        self.feature_norm = feature_norm

    def normalize_audio(self, audio, sample_rate):
        """Resample → RMS normalize → compress. Returns float32 waveform."""
        audio = audio.astype(np.float32)
        if sample_rate != self.target_sr:
            audio = standardize_sample_rate(audio, sample_rate, self.target_sr)
        audio = normalize_rms(audio, self.target_db)
        if self.use_compression:
            audio = dynamic_range_compress(audio)
        return audio

    def normalize_fft_features(self, fft_magnitudes):
        """Log-scale then z-score normalize FFT magnitude features."""
        log_features = log_scale_features(fft_magnitudes)
        return zscore_normalize(log_features)

    def extract_and_normalize_features(self, audio, sample_rate):
        """Normalize audio then extract MFCCs and apply CMVN."""
        try:
            import librosa
        except ImportError:
            raise RuntimeError('librosa is required. Install: pip install librosa')
        audio_norm = self.normalize_audio(audio, sample_rate)
        mfccs = librosa.feature.mfcc(
            y=audio_norm,
            sr=self.target_sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        ).T  # (n_frames, n_mfcc)
        return self._apply_feature_norm(mfccs)

    def _apply_feature_norm(self, features):
        if self.feature_norm == 'cmvn':
            return cmvn(features)
        elif self.feature_norm == 'zscore':
            return zscore_normalize(features)
        elif self.feature_norm == 'minmax':
            return minmax_normalize(features)
        return features.astype(np.float32)

    def prepare_model_input(self, audio, sample_rate, fft_magnitudes=None):
        """
        Main entry point. Returns a (n_mfcc,) feature vector for the SVM.
        Pass fft_magnitudes to use FFT output directly instead of extracting MFCCs.
        """
        if fft_magnitudes is not None:
            features = self.normalize_fft_features(fft_magnitudes)
        else:
            features = self.extract_and_normalize_features(audio, sample_rate)
        return features.mean(axis=0)

    def process_file(self, path):
        """Load a .wav and return its feature vector."""
        sr, audio = wavfile.read(str(path))
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / 32768.0
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self.prepare_model_input(audio, sr)

    def process_batch(self, paths):
        """Process a list of .wav files, returns (n_files, n_features) array."""
        return np.stack([self.process_file(p) for p in paths], axis=0)

    def process_stream(self, audio_chunks):
        """
        Process a live audio stream. Accepts an iterator of (audio, sample_rate)
        tuples from the sounddevice callback and yields feature vectors.
        """
        for audio, sr in audio_chunks:
            yield self.prepare_model_input(audio, sr)
