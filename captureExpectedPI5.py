import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import tkinter as tk
from tkinter import ttk, messagebox

SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_DIR = Path("data/raw/recordings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_CSV = Path("data/processed/recordings_metadata.csv")
METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)

def _save_wav(audio_float32: np.ndarray, sample_rate: int, out_path: Path) -> Path:
    audio_clipped = np.clip(audio_float32, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    wav_write(str(out_path), sample_rate, audio_int16)
    return out_path

def _level_to_quality(level: float) -> str:
    if level < 0.01:
        return "Too quiet"
    if level < 0.04:
        return "Good"
    if level < 0.12:
        return "Loud"
    return "Very loud / possible clipping"

def _save_metadata(timestamp: str, duration: float, sample_rate: int, path: str):
    row = {"timestamp": timestamp, "duration": duration, "sample_rate": sample_rate, "path": path}
    if METADATA_CSV.exists():
        df = pd.read_csv(METADATA_CSV)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(METADATA_CSV, index=False)

def _preprocess_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    # Simple noise reduction: estimate noise from first 0.5s, apply spectral gating
    noise_len = min(int(sample_rate * 0.5), len(audio))
    noise_clip = audio[:noise_len]
    noise_std = np.std(noise_clip)
    # Spectral gating: zero out values below threshold
    threshold = noise_std * 1.5
    reduced = np.where(np.abs(audio) < threshold, 0, audio)
    # Normalization
    normed = reduced / (np.max(np.abs(reduced)) + 1e-8)
    return normed.astype(np.float32)

class MonitorRecorderUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Mic Recorder + Preprocessing + Metadata")

        self.is_recording = False
        self.frames: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.latest_level = 0.0

        self._build_ui()
        self._poll_level()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(row=0, column=0, sticky="nsew")

        self.status_var = tk.StringVar(value="Status: Idle")
        ttk.Label(frame, textvariable=self.status_var).grid(row=0, column=0, columnspan=2, sticky="w")

        self.start_btn = ttk.Button(frame, text="Start Recording", command=self.start_recording)
        self.start_btn.grid(row=1, column=0, padx=(0, 6), pady=6, sticky="ew")

        self.stop_btn = ttk.Button(frame, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.grid(row=1, column=1, pady=6, sticky="ew")

        self.save_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Save recording to disk", variable=self.save_var).grid(row=2, column=0, columnspan=2, sticky="w")

        ttk.Label(frame, text="Level").grid(row=3, column=0, sticky="w")
        self.level_bar = ttk.Progressbar(frame, orient="horizontal", length=220, mode="determinate", maximum=0.2)
        self.level_bar.grid(row=3, column=1, sticky="ew")

        ttk.Label(frame, text="Quality").grid(row=4, column=0, sticky="w")
        self.quality_var = tk.StringVar(value="N/A")
        ttk.Label(frame, textvariable=self.quality_var).grid(row=4, column=1, sticky="w")

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def _audio_callback(self, indata, frames, time, status):
        chunk = indata.copy()
        if CHANNELS == 1:
            chunk = chunk[:, 0]
        self.frames.append(chunk)
        self.latest_level = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0

    def start_recording(self) -> None:
        if self.is_recording:
            return
        self.frames = []
        self.latest_level = 0.0
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as exc:  # pragma: no cover - UI path
            messagebox.showerror("Audio Error", f"Could not start recording: {exc}")
            self.stream = None
            return

        self.is_recording = True
        self.status_var.set("Status: Recording...")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop_recording(self) -> None:
        if not self.is_recording:
            return

        stream = self.stream
        if stream is not None:
            stream.stop()
            stream.close()
        self.stream = None
        self.is_recording = False

        if not self.frames:
            self.status_var.set("Status: No audio captured")
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            return

        audio = np.concatenate(self.frames)
        duration = len(audio) / SAMPLE_RATE

        # Automated preprocessing
        processed = _preprocess_audio(audio, SAMPLE_RATE)

        if self.save_var.get():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"mic_monitor_{timestamp}.wav"
            _save_wav(processed, SAMPLE_RATE, out_path)
            _save_metadata(timestamp, duration, SAMPLE_RATE, str(out_path))
            self.status_var.set(f"Status: Saved {out_path.name} ({duration:.2f}s)")
        else:
            self.status_var.set(f"Status: Discarded capture ({duration:.2f}s)")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _poll_level(self):
        # Refresh UI every 100ms with latest level and quality label.
        self.level_bar['value'] = min(self.latest_level, self.level_bar['maximum'])
        self.quality_var.set(_level_to_quality(self.latest_level))
        self.root.after(100, self._poll_level)

def main():
    root = tk.Tk()
    MonitorRecorderUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
