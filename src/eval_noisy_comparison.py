from __future__ import annotations

# PI5: Compare clean-trained vs noise-augmented LSTM speaker encoder
# Loads two checkpoints, enrolls speakers from clean audio, then tests
# identification accuracy at each SNR level using MUSAN noise.
#
# Usage (from project root):
#   python src/eval_noisy_comparison.py \
#       --clean-model checkpoints/model_clean.pt \
#       --noisy-model checkpoints/model_noisy.pt \
#       --csv "data/commonvoice-v24_en-AU 3/commonvoice-v24_en-AU.csv" \
#       --audio-dir "data/commonvoice-v24_en-AU 3/audio_files" \
#       --musan-noise-dir musan/noise
#
# Prints a side-by-side accuracy table to stdout.

import argparse
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import TrainConfig, LSTMEmbedder, load_model, _load_mfcc, build_speaker_map


TARGET_SR = 16000
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512


def load_noise_clips(noise_dir):
    clips = []
    for wav in sorted(Path(noise_dir).rglob("*.wav")):
        try:
            y, _ = librosa.load(str(wav), sr=TARGET_SR, mono=True)
            if len(y) > TARGET_SR * 0.5:
                clips.append(y.astype(np.float32))
        except Exception:
            continue
    return clips


def mix_noise(signal, noise_clips, snr_db, rng):
    noise_src = noise_clips[rng.randint(len(noise_clips))]
    if len(noise_src) < len(signal):
        noise_src = np.tile(noise_src, int(np.ceil(len(signal) / len(noise_src))))
    start = rng.randint(0, len(noise_src) - len(signal) + 1)
    noise = noise_src[start : start + len(signal)].copy()

    sig_rms = float(np.sqrt(np.mean(signal ** 2)))
    if sig_rms < 1e-8:
        return signal.copy()
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    if noise_rms < 1e-8:
        return signal.copy()

    noise = noise * (sig_rms * 10.0 ** (-snr_db / 20.0) / noise_rms)
    mixed = signal + noise
    peak = float(np.abs(mixed).max())
    if peak > 1.0:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def get_embedding(model, audio_path, config, device):
    seg = _load_mfcc(audio_path, config.segment_frames)
    x = torch.from_numpy(seg).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x).squeeze(0).cpu().numpy()
    return emb


def get_noisy_embedding(model, audio_path, noise_clips, snr_db, config, device, rng):
    from normalize import NormalizationPipeline
    pipeline = NormalizationPipeline(target_sr=TARGET_SR, n_mfcc=N_MFCC)

    audio, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    audio = pipeline.normalize_audio(audio, sample_rate=TARGET_SR)
    audio = mix_noise(audio, noise_clips, snr_db, rng)

    mfccs = librosa.feature.mfcc(
        y=audio, sr=TARGET_SR, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    ).T

    T = mfccs.shape[0]
    seg_len = config.segment_frames
    if T >= seg_len:
        start = random.randint(0, T - seg_len)
        mfccs = mfccs[start : start + seg_len]
    else:
        pad = np.zeros((seg_len - T, N_MFCC), dtype=np.float32)
        mfccs = np.concatenate([mfccs, pad], axis=0)

    x = torch.from_numpy(mfccs.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(x).squeeze(0).cpu().numpy()
    return emb


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def evaluate_model(model, enroll_embs, test_speakers, test_files, noise_clips,
                   snr_db, config, device, rng):
    speaker_ids = list(enroll_embs.keys())
    correct = 0
    top5_correct = 0
    total = 0

    for spk, path in zip(test_speakers, test_files):
        if snr_db is None:
            emb = get_embedding(model, path, config, device)
        else:
            emb = get_noisy_embedding(model, path, noise_clips, snr_db,
                                      config, device, rng)

        scores = []
        for sid in speaker_ids:
            sim = cosine_sim(emb, enroll_embs[sid])
            scores.append((sid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)

        pred = scores[0][0]
        top5 = [s[0] for s in scores[:5]]

        total += 1
        if pred == spk:
            correct += 1
        if spk in top5:
            top5_correct += 1

    acc = correct / total if total > 0 else 0.0
    top5_acc = top5_correct / total if total > 0 else 0.0
    return acc, top5_acc, total


def parse_args():
    p = argparse.ArgumentParser(description="compare clean vs noisy model at each SNR")
    p.add_argument("--clean-model", required=True, help="clean model checkpoint")
    p.add_argument("--noisy-model", required=True, help="noisy model checkpoint")
    p.add_argument("--csv", required=True, help="Common Voice TSV/CSV")
    p.add_argument("--audio-dir", required=True, help="root dir for audio files")
    p.add_argument("--musan-noise-dir", default="musan/noise")
    p.add_argument("--n-speakers", type=int, default=20, help="number of test speakers")
    p.add_argument("--n-enroll", type=int, default=5, help="enrollment utterances per speaker")
    p.add_argument("--n-test", type=int, default=3, help="test utterances per speaker")
    p.add_argument("--snr-levels", type=float, nargs="+", default=[-5, 0, 5, 10, 15, 20])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()

    print("loading clean model...")
    clean_model = load_model(args.clean_model, cfg).to(device)
    clean_model.eval()

    print("loading noisy model...")
    noisy_model = load_model(args.noisy_model, cfg).to(device)
    noisy_model.eval()

    print("building speaker map...")
    speaker_map = build_speaker_map(args.csv, args.audio_dir)
    print(f"  {len(speaker_map)} speakers total")

    min_files = args.n_enroll + args.n_test
    usable = {spk: files for spk, files in speaker_map.items() if len(files) >= min_files}
    print(f"  {len(usable)} speakers with >= {min_files} utterances")

    chosen = random.sample(list(usable.keys()), min(args.n_speakers, len(usable)))
    print(f"  using {len(chosen)} speakers for evaluation")

    enroll_files = {}
    test_speakers = []
    test_files = []

    for spk in chosen:
        files = list(usable[spk])
        random.shuffle(files)
        enroll_files[spk] = files[:args.n_enroll]
        for f in files[args.n_enroll : args.n_enroll + args.n_test]:
            test_speakers.append(spk)
            test_files.append(f)

    print(f"  {len(test_files)} test utterances")

    print(f"loading MUSAN noise from {args.musan_noise_dir}...")
    noise_clips = load_noise_clips(args.musan_noise_dir)
    print(f"  {len(noise_clips)} clips")

    print("\ncomputing enrollment embeddings...")
    clean_enroll = {}
    noisy_enroll = {}
    for spk in chosen:
        c_embs = [get_embedding(clean_model, f, cfg, device) for f in enroll_files[spk]]
        n_embs = [get_embedding(noisy_model, f, cfg, device) for f in enroll_files[spk]]
        clean_enroll[spk] = np.mean(c_embs, axis=0)
        noisy_enroll[spk] = np.mean(n_embs, axis=0)

    conditions = [("clean", None)] + [(f"{s:+d}dB", s) for s in args.snr_levels]
    rows = []

    for label, snr_db in conditions:
        rng_c = np.random.RandomState(args.seed + 1)
        rng_n = np.random.RandomState(args.seed + 1)

        c_acc, c_top5, c_n = evaluate_model(
            clean_model, clean_enroll, test_speakers, test_files,
            noise_clips, snr_db, cfg, device, rng_c,
        )
        n_acc, n_top5, n_n = evaluate_model(
            noisy_model, noisy_enroll, test_speakers, test_files,
            noise_clips, snr_db, cfg, device, rng_n,
        )

        delta = n_acc - c_acc
        rows.append({
            "condition": label,
            "snr_db": snr_db if snr_db is not None else "inf",
            "clean_acc": round(c_acc, 4),
            "clean_top5": round(c_top5, 4),
            "noisy_acc": round(n_acc, 4),
            "noisy_top5": round(n_top5, 4),
            "delta_acc": round(delta, 4),
            "n_evaluated": c_n,
        })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
