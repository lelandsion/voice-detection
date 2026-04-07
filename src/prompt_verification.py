from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


try:
    from wonderwords import RandomWord
except ImportError:
    RandomWord = None

try:
    from .audio import MicrophoneRecorder
    from .train import LSTMEmbedder, TrainConfig, embed
except ImportError:
    from audio import MicrophoneRecorder
    from train import LSTMEmbedder, TrainConfig, embed


def _slugify(text: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return clean or "prompt"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec
    return vec / norm


def _infer_train_config_from_checkpoint(model_state: dict[str, torch.Tensor]) -> TrainConfig:
    ih0 = model_state["lstm.weight_ih_l0"]
    hh0 = model_state["lstm.weight_hh_l0"]
    proj = model_state["projection.weight"]

    input_dim = int(ih0.shape[1])
    hidden_dim = int(hh0.shape[1])
    embedding_dim = int(proj.shape[0])
    num_layers = len([k for k in model_state.keys() if k.startswith("lstm.weight_ih_l")])

    return TrainConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
    )


def load_embedding_model(model_path: str | Path) -> tuple[LSTMEmbedder, TrainConfig]:
    model_path = Path(model_path)
    checkpoint = torch.load(str(model_path), map_location="cpu")
    state_dict = checkpoint["state_dict"]
    cfg = _infer_train_config_from_checkpoint(state_dict)

    model = LSTMEmbedder(cfg)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


@dataclass
class EnrollmentPromptEntry:
    prompt: str
    vector_path: str
    count: int
    updated_at_utc: str


class EnrollmentDB:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)
        self.index_path = self.root_dir / "index.json"
        self.vectors_dir = self.root_dir / "vectors"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_dir.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            self.data = json.loads(self.index_path.read_text(encoding="utf-8"))
        else:
            self.data = {"version": 1, "speakers": {}}

    def save(self) -> None:
        self.index_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def _get_prompt_map(self, speaker_id: str) -> dict[str, dict[str, Any]]:
        speakers = self.data.setdefault("speakers", {})
        spk = speakers.setdefault(speaker_id, {"prompts": {}})
        return spk.setdefault("prompts", {})

    def upsert_prompt_embedding(self, speaker_id: str, prompt: str, embedding: np.ndarray, count: int) -> None:
        if count < 1:
            raise ValueError("count must be >= 1")

        prompt_key = _slugify(prompt)
        prompt_map = self._get_prompt_map(speaker_id)
        timestamp = datetime.now(timezone.utc).isoformat()
        vec_file = f"{_slugify(speaker_id)}__{prompt_key}.npy"
        vec_path = self.vectors_dir / vec_file

        if prompt_key in prompt_map:
            prev = prompt_map[prompt_key]
            prev_vec = np.load(self.root_dir / prev["vector_path"])
            prev_count = int(prev.get("count", 1))
            merged = ((prev_vec * prev_count) + (embedding * count)) / float(prev_count + count)
            merged = _unit_vector(merged)
            np.save(vec_path, merged.astype(np.float32))

            prev["count"] = int(prev_count + count)
            prev["updated_at_utc"] = timestamp
            prev["prompt"] = prompt
            prev["vector_path"] = (Path("vectors") / vec_file).as_posix()
            return

        np.save(vec_path, _unit_vector(embedding).astype(np.float32))
        prompt_map[prompt_key] = EnrollmentPromptEntry(
            prompt=prompt,
            vector_path=(Path("vectors") / vec_file).as_posix(),
            count=int(count),
            updated_at_utc=timestamp,
        ).__dict__

    def speakers(self) -> list[str]:
        return sorted(self.data.get("speakers", {}).keys())

    def prompts_for_speaker(self, speaker_id: str) -> dict[str, dict[str, Any]]:
        spk = self.data.get("speakers", {}).get(speaker_id, {})
        return dict(spk.get("prompts", {}))

    def load_prompt_vector(self, speaker_id: str, prompt_key: str) -> np.ndarray:
        prompt = self.prompts_for_speaker(speaker_id).get(prompt_key)
        if prompt is None:
            raise KeyError(f"Prompt {prompt_key} not found for speaker {speaker_id}")
        return np.load(self.root_dir / prompt["vector_path"]).astype(np.float32)

    def global_prompt_inventory(self) -> dict[str, str]:
        prompts: dict[str, str] = {}
        for speaker_id in self.speakers():
            for key, meta in self.prompts_for_speaker(speaker_id).items():
                prompts[key] = str(meta.get("prompt", key))
        return prompts


def _record_live_utterance(recorder: MicrophoneRecorder, seconds: float, file_prefix: str) -> Path:
    rec = recorder.record_for(seconds=seconds, save=True, file_prefix=file_prefix)
    if rec.saved_path is None:
        raise RuntimeError("Live capture did not produce a saved file.")
    return rec.saved_path


def _extract_embedding_from_file(
    model: LSTMEmbedder,
    cfg: TrainConfig,
    audio_path: str | Path,
    device: torch.device,
) -> np.ndarray:
    vec = embed(model=model, paths=[audio_path], config=cfg, device=device)
    return _unit_vector(vec.astype(np.float32))


def _generate_random_word_prompt(num_words: int, rng: random.Random) -> str:
    if num_words <= 0:
        raise ValueError("num_words must be > 0")
    if RandomWord is None:
        raise RuntimeError(
            "wonderwords is required for random word prompts. Install it with: pip install wonderwords"
        )

    generator = RandomWord()
    words: list[str] = []
    seen: set[str] = set()
    max_attempts = max(num_words * 10, 20)

    for _ in range(max_attempts):
        word = generator.word(word_min_length=3, word_max_length=10)
        if not word:
            continue
        word = str(word).strip().lower()
        if not word or not word.isalpha() or word in seen:
            continue
        seen.add(word)
        words.append(word)
        if len(words) == num_words:
            break

    if len(words) != num_words:
        raise RuntimeError("Unable to generate enough random words from wonderwords.")

    rng.shuffle(words)
    return " ".join(words)


def enroll_voice(args: argparse.Namespace) -> None:
    model, cfg = load_embedding_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    db = EnrollmentDB(args.db_dir)
    recorder = MicrophoneRecorder(sample_rate=args.sample_rate)
    rng = random.Random(args.seed)

    prompt_to_paths: dict[str, list[Path]] = {}

    if args.mode == "recorded":
        if not args.audio_paths:
            raise ValueError("Recorded mode requires one or more --audio-paths values.")

        paths = [Path(p) for p in args.audio_paths]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Missing recorded file: {p}")

        if args.prompt_texts:
            prompt_texts = [p.strip() for p in args.prompt_texts.split("|") if p.strip()]
            if len(prompt_texts) == 1:
                prompt_to_paths[prompt_texts[0]] = paths
            elif len(prompt_texts) == len(paths):
                for prompt, path in zip(prompt_texts, paths):
                    prompt_to_paths.setdefault(prompt, []).append(path)
            else:
                raise ValueError("When provided, --prompt-texts must be one prompt or one per audio path.")
        else:
            sampled_prompts = [
                _generate_random_word_prompt(args.num_words, rng)
                for _ in range(min(len(paths), args.num_prompts))
            ]
            for idx, path in enumerate(paths):
                prompt = sampled_prompts[idx % len(sampled_prompts)]
                prompt_to_paths.setdefault(prompt, []).append(path)

    else:
        prompts = [p.strip() for p in args.prompt_texts.split("|") if p.strip()] if args.prompt_texts else []
        if not prompts:
            prompts = [
                _generate_random_word_prompt(args.num_words, rng)
                for _ in range(args.num_prompts)
            ]

        for prompt in prompts:
            for take_idx in range(args.takes_per_prompt):
                prefix = f"enroll_{_slugify(args.speaker_id)}_{_slugify(prompt)}_{take_idx + 1}"
                print(f"Speak these words: {prompt}")
                input("Press Enter to start recording...")
                wav_path = _record_live_utterance(recorder, args.seconds, prefix)
                prompt_to_paths.setdefault(prompt, []).append(wav_path)

    for prompt, paths in prompt_to_paths.items():
        vectors: list[np.ndarray] = []
        for path in paths:
            vectors.append(_extract_embedding_from_file(model, cfg, path, device))
        prompt_vector = _unit_vector(np.mean(np.stack(vectors, axis=0), axis=0))
        db.upsert_prompt_embedding(args.speaker_id, prompt, prompt_vector, count=len(vectors))

    db.save()
    print(
        json.dumps(
            {
                "status": "enrolled",
                "speaker_id": args.speaker_id,
                "mode": args.mode,
                "prompts": list(prompt_to_paths.keys()),
                "samples": int(sum(len(v) for v in prompt_to_paths.values())),
                "db_dir": str(Path(args.db_dir)),
            },
            indent=2,
        )
    )


def _speaker_vector_for_prompt(
    db: EnrollmentDB,
    speaker_id: str,
    prompt_key: str,
    reference_mode: str = "auto",
) -> tuple[np.ndarray, str]:
    prompt_map = db.prompts_for_speaker(speaker_id)
    mode = reference_mode.strip().lower()
    if mode not in {"auto", "prompt", "speaker-average"}:
        raise ValueError("reference_mode must be one of: auto, prompt, speaker-average")

    if mode in {"auto", "prompt"} and prompt_key in prompt_map:
        return db.load_prompt_vector(speaker_id, prompt_key), "prompt"

    if mode == "prompt":
        raise ValueError(
            f"Prompt '{prompt_key}' not enrolled for speaker '{speaker_id}' while using strict prompt mode."
        )

    vectors = [db.load_prompt_vector(speaker_id, key) for key in prompt_map.keys()]
    if not vectors:
        raise ValueError(f"Speaker {speaker_id} has no enrollment vectors.")
    return _unit_vector(np.mean(np.stack(vectors, axis=0), axis=0)), "speaker_average"


def verify_voice(args: argparse.Namespace) -> None:
    model, cfg = load_embedding_model(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    db = EnrollmentDB(args.db_dir)
    if not db.speakers():
        raise RuntimeError("Enrollment database is empty. Run enroll first.")

    rng = random.Random(args.seed)
    recorder = MicrophoneRecorder(sample_rate=args.sample_rate)

    if args.prompt_text:
        challenge_prompt = args.prompt_text.strip()
        prompt_key = _slugify(challenge_prompt)
    else:
        if args.claimed_speaker:
            prompt_inventory = db.prompts_for_speaker(args.claimed_speaker)
            if not prompt_inventory:
                raise ValueError(f"No enrolled prompts found for claimed speaker: {args.claimed_speaker}")
            prompt_key = rng.choice(list(prompt_inventory.keys()))
            challenge_prompt = str(prompt_inventory[prompt_key]["prompt"])
        else:
            inventory = db.global_prompt_inventory()
            prompt_key = rng.choice(list(inventory.keys()))
            challenge_prompt = inventory[prompt_key]

    if args.mode == "recorded":
        if not args.audio_path:
            raise ValueError("Recorded mode requires --audio-path")
        challenge_path = Path(args.audio_path)
        if not challenge_path.exists():
            raise FileNotFoundError(f"Missing recorded verification file: {challenge_path}")
    else:
        prefix = f"verify_{_slugify(args.claimed_speaker or 'unknown')}_{prompt_key}"
        print(f"Speak these words: {challenge_prompt}")
        input("Press Enter to start recording...")
        challenge_path = _record_live_utterance(recorder, args.seconds, prefix)

    query_vec = _extract_embedding_from_file(model, cfg, challenge_path, device)

    scored: list[dict[str, Any]] = []
    for speaker_id in db.speakers():
        ref_vec, source = _speaker_vector_for_prompt(
            db,
            speaker_id,
            prompt_key,
            reference_mode=args.reference_mode,
        )
        score = _cosine_similarity(query_vec, ref_vec)
        scored.append({"speaker_id": speaker_id, "score": score, "reference": source})

    scored.sort(key=lambda x: float(x["score"]), reverse=True)
    top = scored[0]

    accepted: bool
    claimed_score: float | None = None

    if args.claimed_speaker:
        claimed = next((s for s in scored if s["speaker_id"] == args.claimed_speaker), None)
        claimed_score = float(claimed["score"]) if claimed else None
        accepted = bool(
            claimed is not None
            and claimed_score is not None
            and claimed_score >= args.threshold
            and top["speaker_id"] == args.claimed_speaker
        )
    else:
        accepted = float(top["score"]) >= args.threshold

    print(
        json.dumps(
            {
                "status": "verified",
                "accepted": accepted,
                "mode": args.mode,
                "challenge_prompt": challenge_prompt,
                "reference_mode": args.reference_mode,
                "challenge_audio": str(challenge_path),
                "threshold": float(args.threshold),
                "claimed_speaker": args.claimed_speaker,
                "claimed_score": claimed_score,
                "top_match": top,
                "ranked_scores": scored,
            },
            indent=2,
        )
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Random-prompt speaker verification pipeline for live or recorded audio."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    enroll = sub.add_parser("enroll", help="Enroll one speaker with prompt-aware embeddings.")
    enroll.add_argument("--speaker-id", type=str, required=True)
    enroll.add_argument("--model-path", type=Path, required=True)
    enroll.add_argument("--db-dir", type=Path, default=Path("data/processed/enrollment_db"))
    enroll.add_argument("--mode", choices=("live", "recorded"), default="live")
    enroll.add_argument("--audio-paths", nargs="*", default=[])
    enroll.add_argument(
        "--prompt-texts",
        type=str,
        default="",
        help="Optional explicit prompt list separated by '|'. In recorded mode use 1 prompt or one per audio file.",
    )
    enroll.add_argument("--num-prompts", type=int, default=3)
    enroll.add_argument(
        "--num-words",
        type=int,
        default=3,
        help="Number of random words per generated enrollment prompt.",
    )
    enroll.add_argument("--takes-per-prompt", type=int, default=2)
    enroll.add_argument("--seconds", type=float, default=3.0)
    enroll.add_argument("--sample-rate", type=int, default=16000)
    enroll.add_argument("--seed", type=int, default=42)
    enroll.set_defaults(func=enroll_voice)

    verify = sub.add_parser("verify", help="Run random-prompt verification against enrolled voices.")
    verify.add_argument("--model-path", type=Path, required=True)
    verify.add_argument("--db-dir", type=Path, default=Path("data/processed/enrollment_db"))
    verify.add_argument("--mode", choices=("live", "recorded"), default="live")
    verify.add_argument("--audio-path", type=Path)
    verify.add_argument("--claimed-speaker", type=str, default="")
    verify.add_argument("--prompt-text", type=str, default="")
    verify.add_argument(
        "--num-words",
        type=int,
        default=3,
        help="Reserved for explicit random-word verification flows; enrolled prompts remain the source of truth.",
    )
    verify.add_argument("--threshold", type=float, default=0.60)
    verify.add_argument("--seconds", type=float, default=3.0)
    verify.add_argument("--sample-rate", type=int, default=16000)
    verify.add_argument("--seed", type=int, default=42)
    verify.add_argument(
        "--reference-mode",
        choices=("auto", "prompt", "speaker-average"),
        default="auto",
        help="How enrolled references are selected during scoring.",
    )
    verify.set_defaults(func=verify_voice)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if hasattr(args, "claimed_speaker"):
        args.claimed_speaker = args.claimed_speaker.strip() if args.claimed_speaker else ""
    if hasattr(args, "prompt_text"):
        args.prompt_text = args.prompt_text.strip() if args.prompt_text else ""
    args.func(args)


if __name__ == "__main__":
    main()
