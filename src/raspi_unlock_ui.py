from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import random
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import torch

ENROLL_PROMPTS = 3
ENROLL_TAKES = 2
REFERENCE_MODE = "speaker-average"
THRESHOLD = 0.12

try:
    from .audio import MicrophoneRecorder
    from .prompt_verification import EnrollmentDB, load_embedding_model
    from .train import TrainConfig, embed
except ImportError:
    from audio import MicrophoneRecorder
    from prompt_verification import EnrollmentDB, load_embedding_model
    from train import TrainConfig, embed


@dataclass
class ModelBundle:
    model: Any
    cfg: TrainConfig
    device: torch.device


class PasswordStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def has_password(self) -> bool:
        if not self.path.exists():
            return False
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return all(k in data for k in ["salt", "hash"])
        except Exception:
            return False

    def set_password(self, password: str) -> None:
        if not password:
            raise ValueError("Password cannot be empty")

        salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
        payload = {
            "salt": base64.b64encode(salt).decode("ascii"),
            "hash": base64.b64encode(digest).decode("ascii"),
            "iterations": 200_000,
            "algo": "pbkdf2_hmac_sha256",
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def verify(self, password: str) -> bool:
        if not self.has_password():
            return False

        data = json.loads(self.path.read_text(encoding="utf-8"))
        salt = base64.b64decode(data["salt"])
        expected = base64.b64decode(data["hash"])
        iters = int(data.get("iterations", 200_000))
        got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
        return hmac.compare_digest(got, expected)


def slugify(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in text.strip())
    return "_".join(part for part in cleaned.split("_") if part) or "prompt"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def unit_vector(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= 1e-12:
        return vec.astype(np.float32)
    return (vec / n).astype(np.float32)


class RaspiVoiceUnlockUI:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rng = random.Random(args.seed)
        self.db = EnrollmentDB(args.db_dir)
        self.password_store = PasswordStore(args.password_store)
        self.recorder = MicrophoneRecorder(
            sample_rate=args.sample_rate,
            output_dir=Path(args.recordings_dir),
        )
        self.model_cache: dict[str, ModelBundle] = {}

        self.root = tk.Tk()
        self.root.title("Voice Unlock")
        self.root.configure(bg="#0f172a")

        if args.fullscreen:
            self.root.attributes("-fullscreen", True)
        if args.always_on_top:
            self.root.attributes("-topmost", True)

        if args.allow_exit_key:
            self.root.bind("<Escape>", lambda _: self.root.destroy())

        self.model_var = tk.StringVar()
        self.speaker_var = tk.StringVar()
        self.ref_mode_var = tk.StringVar(value=args.reference_mode)
        self.threshold_var = tk.StringVar(value=f"{args.threshold:.2f}")
        self.seconds_var = tk.StringVar(value=f"{args.seconds:.1f}")
        self.enroll_prompts_var = tk.StringVar(value=str(args.enroll_prompts))
        self.enroll_takes_var = tk.StringVar(value=str(args.enroll_takes))

        self.prompt_text_var = tk.StringVar(value="Press 'New Prompt' to begin.")
        self.status_var = tk.StringVar(value="Locked")
        self.score_var = tk.StringVar(value="-")

        self._is_busy = False
        self._build_ui()
        self._refresh_models()
        self._refresh_speakers()
        self._new_prompt()

    def _build_ui(self) -> None:
        self.root.geometry("1000x700")
        self.root.configure(bg="#0b1220")  # deep dark

        style = ttk.Style()
        style.theme_use("clam")

        # Colors
        BG = "#0b1220"
        CARD = "#111827"
        ACCENT = "#60a5fa"  # light blue
        TEXT = "#e5e7eb"
        MUTED = "#9ca3af"
        BUTTON = "#1d4ed8"
        BUTTON_HOVER = "#2563eb"

        # Fonts
        TITLE = ("Segoe UI", 26, "bold")
        PROMPT = ("Segoe UI", 24, "bold")
        BODY = ("Segoe UI", 11)

        # Base styles
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=CARD, foreground=TEXT, font=BODY)

        # Fake "rounded" button look
        style.configure(
            "Modern.TButton",
            background=BUTTON,
            foreground="white",
            padding=10,
            borderwidth=0,
            focusthickness=3,
            focuscolor="none",
            font=("Segoe UI", 11, "bold")
        )

        style.map(
            "Modern.TButton",
            background=[("active", BUTTON_HOVER)],
        )

        # MAIN CONTAINER (center everything)
        container = ttk.Frame(self.root)
        container.place(relx=0.5, rely=0.5, anchor="center")

        # ===== TITLE =====
        title = tk.Label(
            container,
            text="Voice Unlock",
            font=TITLE,
            fg="white",
            bg=BG
        )
        title.pack(pady=(0, 20))

        # ===== CARD =====
        card = tk.Frame(
            container,
            bg=CARD,
            bd=0,
            highlightthickness=0
        )
        card.pack(padx=20, pady=10, fill="both")

        inner = tk.Frame(card, bg=CARD)
        inner.pack(padx=30, pady=30)

        # ===== MODEL + SPEAKER ROW =====
        top_row = tk.Frame(inner, bg=CARD)
        top_row.pack(fill="x", pady=(0, 15))

        self.model_box = ttk.Combobox(
            top_row,
            textvariable=self.model_var,
            state="readonly",
            width=45
        )
        self.model_box.pack(side="left", padx=(0, 10))

        self.speaker_box = ttk.Combobox(
            top_row,
            textvariable=self.speaker_var,
            width=20
        )
        self.speaker_box.pack(side="left")

        # ===== PROMPT =====
        tk.Label(
            inner,
            text="Say this phrase",
            fg=MUTED,
            bg=CARD,
            font=("Segoe UI", 11)
        ).pack(anchor="w")

        tk.Label(
            inner,
            textvariable=self.prompt_text_var,
            fg=ACCENT,
            bg=CARD,
            font=PROMPT,
            wraplength=700,
            justify="left"
        ).pack(anchor="w", pady=(5, 20))

        # ===== BUTTONS =====
        button_row = tk.Frame(inner, bg=CARD)
        button_row.pack(fill="x", pady=(0, 20))

        ttk.Button(
            button_row,
            text="New Prompt",
            style="Modern.TButton",
            command=self._new_prompt
        ).pack(side="left", padx=5)

        ttk.Button(
            button_row,
            text= "Verify",
            style="Modern.TButton",
            command=self._verify_async
        ).pack(side="left", padx=5)

        ttk.Button(
            button_row,
            text="Enroll",
            style="Modern.TButton",
            command=self._enroll_async
        ).pack(side="left", padx=5)

        # ===== STATUS =====
        status_row = tk.Frame(inner, bg=CARD)
        status_row.pack(fill="x", pady=(10, 0))

        tk.Label(
            status_row,
            text="Status:",
            fg=MUTED,
            bg=CARD
        ).pack(side="left")

        self.status_label = tk.Label(
            status_row,
            textvariable=self.status_var,
            fg="#fbbf24",
            bg=CARD,
            font=("Segoe UI", 14, "bold")
        )
        self.status_label.pack(side="left", padx=10)

        # ===== SCORE =====
        score_row = tk.Frame(inner, bg=CARD)
        score_row.pack(fill="x", pady=(5, 0))

        tk.Label(
            score_row,
            text="Confidence:",
            fg=MUTED,
            bg=CARD
        ).pack(side="left")

        tk.Label(
            score_row,
            textvariable=self.score_var,
            fg=TEXT,
            bg=CARD
        ).pack(side="left", padx=10)

        # ===== RECORD TIME =====
        bottom_row = tk.Frame(inner, bg=CARD)
        bottom_row.pack(fill="x", pady=(15, 0))

        tk.Label(
            bottom_row,
            text="Recording (seconds):",
            fg=MUTED,
            bg=CARD
        ).pack(side="left")

        tk.Entry(
            bottom_row,
            textvariable=self.seconds_var,
            width=6,
            bg="#1f2937",
            fg="white",
            insertbackground="white",
            relief="flat"
        ).pack(side="left", padx=10)

    def _refresh_models(self) -> None:
        checkpoints = sorted(Path(self.args.checkpoints_dir).glob("*.pt"))
        values = [str(p) for p in checkpoints]

        # Also accept direct model paths passed by --default-model.
        if self.args.default_model:
            direct = Path(self.args.default_model)
            candidate: Path | None = None
            if direct.exists():
                candidate = direct
            else:
                joined = Path(self.args.checkpoints_dir) / direct
                if joined.exists():
                    candidate = joined
            if candidate is not None and str(candidate) not in values:
                values.insert(0, str(candidate))

        self.model_box["values"] = values

        if not values:
            self.model_var.set("")
            self.status_var.set("No checkpoints found (check --checkpoints-dir or --default-model)")
            return

        current = self.model_var.get()
        if current not in values:
            preferred = str(Path(self.args.default_model)) if self.args.default_model else ""
            self.model_var.set(preferred if preferred in values else values[0])

    def _refresh_speakers(self) -> None:
        self.db = EnrollmentDB(self.args.db_dir)
        speakers = self.db.speakers()
        self.speaker_box["values"] = speakers

        if not speakers:
            default_spk = self.args.default_speaker.strip() if self.args.default_speaker else ""
            self.speaker_var.set(default_spk)
            self.status_var.set("No enrolled speakers (use Enroll Voice)")
            return

        if self.speaker_var.get() not in speakers:
            preferred = self.args.default_speaker.strip() if self.args.default_speaker else ""
            self.speaker_var.set(preferred if preferred in speakers else speakers[0])

    def _new_prompt(self) -> None:
        speaker = self.speaker_var.get().strip()
        if not speaker:
            self.prompt_text_var.set("Enroll a speaker first.")
            return

        prompts = self.db.prompts_for_speaker(speaker)
        if not prompts:
            self.prompt_text_var.set("No prompts enrolled for selected speaker.")
            return

        key = self.rng.choice(list(prompts.keys()))
        self.prompt_text_var.set(str(prompts[key].get("prompt", key)))

    def _generate_enroll_prompt(self) -> str:
        words = [
            "apple", "river", "lantern", "storm", "piano", "silver",
            "forest", "window", "saturn", "candle", "orange", "velvet",
        ]
        picked = self.rng.sample(words, k=3)
        return " ".join(picked)

    def _load_model_bundle(self, model_path: str) -> ModelBundle:
        if model_path in self.model_cache:
            return self.model_cache[model_path]

        model, cfg = load_embedding_model(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        bundle = ModelBundle(model=model, cfg=cfg, device=device)
        self.model_cache[model_path] = bundle
        return bundle

    def _speaker_reference_vector(self, speaker: str, prompt_key: str, mode: str) -> tuple[np.ndarray, str]:
        prompt_map = self.db.prompts_for_speaker(speaker)
        if not prompt_map:
            raise ValueError(f"Speaker '{speaker}' has no enrollment prompts.")

        mode = mode.strip().lower()
        if mode not in {"auto", "prompt", "speaker-average"}:
            raise ValueError("Reference mode must be auto, prompt, or speaker-average")

        if mode in {"auto", "prompt"} and prompt_key in prompt_map:
            return self.db.load_prompt_vector(speaker, prompt_key), "prompt"

        if mode == "prompt":
            raise ValueError(f"Prompt '{prompt_key}' not found for speaker '{speaker}'")

        vectors = [self.db.load_prompt_vector(speaker, k) for k in prompt_map.keys()]
        avg = unit_vector(np.mean(np.stack(vectors, axis=0), axis=0))
        return avg, "speaker_average"

    def _verify_async(self) -> None:
        if self._is_busy:
            return
        thread = threading.Thread(target=self._verify_worker, daemon=True)
        thread.start()

    def _enroll_async(self) -> None:
        if self._is_busy:
            return
        thread = threading.Thread(target=self._enroll_worker, daemon=True)
        thread.start()

    def _verify_worker(self) -> None:
        self._is_busy = True
        try:
            self._set_status("Listening...")

            model_path = self.model_var.get().strip()
            speaker = self.speaker_var.get().strip()
            prompt_text = self.prompt_text_var.get().strip()
            ref_mode = REFERENCE_MODE

            if not model_path:
                raise ValueError("Select a model checkpoint first.")
            if not speaker:
                raise ValueError("Select a speaker first.")
            if not prompt_text or prompt_text == "Press 'New Prompt' to begin.":
                raise ValueError("Generate a prompt first.")

            threshold = THRESHOLD
            seconds = float(self.seconds_var.get())
            if seconds <= 0:
                raise ValueError("Record seconds must be > 0")

            bundle = self._load_model_bundle(model_path)

            prompt_key = slugify(prompt_text)
            prefix = f"unlock_{slugify(speaker)}_{prompt_key}"
            result = self.recorder.record_for(seconds=seconds, save=True, file_prefix=prefix)
            if result.saved_path is None:
                raise RuntimeError("No audio file captured")

            query = embed(bundle.model, [result.saved_path], bundle.cfg, bundle.device)
            query = unit_vector(query)

            scored: list[dict[str, Any]] = []
            for speaker_id in self.db.speakers():
                ref_vec, source = self._speaker_reference_vector(speaker_id, prompt_key, ref_mode)
                score = cosine(query, ref_vec)
                scored.append({"speaker_id": speaker_id, "score": score, "reference": source})

            scored.sort(key=lambda x: float(x["score"]), reverse=True)
            top = scored[0]

            claimed = next((s for s in scored if s["speaker_id"] == speaker), None)
            claimed_score = float(claimed["score"]) if claimed else -1.0
            accepted = bool(claimed and claimed_score >= threshold and top["speaker_id"] == speaker)

            self._set_score(f"{claimed_score:.4f} (top={top['speaker_id']} {top['score']:.4f})")

            if accepted:
                self._set_status("Unlocked")
                self._handle_unlock_action()
                self._new_prompt()
            else:
                self._set_status("Denied")

        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Error: {exc}")
        finally:
            self._is_busy = False

    def _enroll_worker(self) -> None:
        self._is_busy = True
        try:
            self._set_status("Enrollment in progress...")

            model_path = self.model_var.get().strip()
            speaker = self.speaker_var.get().strip()

            if not model_path:
                raise ValueError("Select a model checkpoint first.")

            if not speaker:
                raise ValueError("Select a speaker first (or create one by typing it in the speaker box).")

            prompts_n = ENROLL_PROMPTS
            takes_n = ENROLL_TAKES
            seconds = float(self.seconds_var.get())

            if prompts_n <= 0 or takes_n <= 0:
                raise ValueError("Enroll prompts and takes must be > 0")
            if seconds <= 0:
                raise ValueError("Record seconds must be > 0")

            bundle = self._load_model_bundle(model_path)
            new_prompts: dict[str, list[Path]] = {}

            for prompt_idx in range(prompts_n):
                prompt_text = self._generate_enroll_prompt()
                self._set_prompt(prompt_text)

                for take_idx in range(takes_n):
                    self._set_status(
                        f"Enroll: prompt {prompt_idx + 1}/{prompts_n}, take {take_idx + 1}/{takes_n}. Speak now..."
                    )
                    prefix = f"enroll_{slugify(speaker)}_{slugify(prompt_text)}_{take_idx + 1}"
                    rec = self.recorder.record_for(seconds=seconds, save=True, file_prefix=prefix)
                    if rec.saved_path is None:
                        raise RuntimeError("No audio file captured during enrollment")
                    new_prompts.setdefault(prompt_text, []).append(rec.saved_path)

            for prompt_text, paths in new_prompts.items():
                vecs = []
                for p in paths:
                    emb = embed(bundle.model, [p], bundle.cfg, bundle.device)
                    vecs.append(unit_vector(emb))
                prompt_vec = unit_vector(np.mean(np.stack(vecs, axis=0), axis=0))
                self.db.upsert_prompt_embedding(speaker, prompt_text, prompt_vec, count=len(vecs))

            self.db.save()
            self._set_status("Enrollment complete")
            self.root.after(0, lambda: self._finalize_enrollment_ui(speaker))

        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Error: {exc}")
        finally:
            self._is_busy = False

    def _handle_unlock_action(self) -> None:
        if self.args.on_unlock_command:
            try:
                subprocess.Popen(self.args.on_unlock_command, shell=True)
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Unlocked (hook failed: {exc})")
                return

        if self.args.close_on_unlock:
            self.root.after(500, self.root.destroy)

    def _set_password_dialog(self) -> None:
        pwd = simpledialog.askstring("Set Password", "Enter fallback password:", show="*", parent=self.root)
        if pwd is None:
            return
        confirm = simpledialog.askstring("Set Password", "Confirm fallback password:", show="*", parent=self.root)
        if confirm is None:
            return

        if pwd != confirm:
            messagebox.showerror("Password", "Passwords do not match.")
            return

        if len(pwd) < 4:
            messagebox.showerror("Password", "Password must be at least 4 characters.")
            return

        self.password_store.set_password(pwd)
        self._set_status("Fallback password set")
        messagebox.showinfo("Password", "Fallback password saved.")

    def _password_unlock_dialog(self) -> None:
        if not self.password_store.has_password():
            messagebox.showerror("Password", "No fallback password is set.")
            return

        pwd = simpledialog.askstring("Unlock", "Enter fallback password:", show="*", parent=self.root)
        if pwd is None:
            return

        if self.password_store.verify(pwd):
            self._set_status("Unlocked (password)")
            self._set_score("password")
            self._handle_unlock_action()
        else:
            self._set_status("Denied (bad password)")

    def _set_prompt(self, text: str) -> None:
        self.root.after(0, lambda: self.prompt_text_var.set(text))

    def _finalize_enrollment_ui(self, speaker: str) -> None:
        self._refresh_speakers()
        self.speaker_var.set(speaker)
        self._new_prompt()
        messagebox.showinfo("Enrollment", f"Enrollment saved for '{speaker}'.")

    def _set_status(self, text: str) -> None:
        def update():
            self.status_var.set(text)

            if "Unlocked" in text:
                self.status_label.config(foreground="#22c55e")  # green
            elif "Denied" in text:
                self.status_label.config(foreground="#ef4444")  # red
            elif "Error" in text:
                self.status_label.config(foreground="#f97316")  # orange
            else:
                self.status_label.config(foreground="#eab308")  # yellow

        self.root.after(0, update)

    def _set_score(self, text: str) -> None:
        self.root.after(0, lambda: self.score_var.set(text))

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi fullscreen voice unlock UI")
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--default-model", type=Path, default=Path("checkpoints/model_noisy.pt"))
    parser.add_argument("--db-dir", type=Path, default=Path("data/processed/enrollment_db"))
    parser.add_argument("--recordings-dir", type=Path, default=Path("data/raw/recordings"))
    parser.add_argument("--default-speaker", type=str, default="degan_v2")
    parser.add_argument("--reference-mode", choices=("auto", "prompt", "speaker-average"), default="speaker-average")
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--password-store", type=Path, default=Path("data/processed/unlock_password.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enroll-prompts", type=int, default=3)
    parser.add_argument("--enroll-takes", type=int, default=2)

    parser.add_argument("--fullscreen", action="store_true", default=True)
    parser.add_argument("--windowed", action="store_false", dest="fullscreen")
    parser.add_argument("--always-on-top", action="store_true", default=True)
    parser.add_argument("--allow-exit-key", action="store_true", default=True)
    parser.add_argument("--close-on-unlock", action="store_true", default=False)
    parser.add_argument("--on-unlock-command", type=str, default="")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = RaspiVoiceUnlockUI(args)
    app.run()


if __name__ == "__main__":
    main()
