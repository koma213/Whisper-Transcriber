# Whisper Transcriber

A desktop transcription tool for Windows built on faster-whisper and PySide6. Drag in video or audio files and get back plain text, timestamped text, or SRT/VTT/ASS subtitles — with optional speaker identification and labeling (diarization) and subtitle burn-in.

**Current version:** 1.01

## Features

- Batch transcription of video and audio files (mp4, mkv, mov, avi, webm, mp3, wav, m4a, flac, and more)
- Output formats: plain text, timestamped text, SRT, WebVTT, ASS subtitles
- Optional subtitle burn-in to a new video file (via ffmpeg)
- Optional speaker diarization (who-said-what) using pyannote.audio, with color-coded live preview
- Batch multi-model mode — run the same file through several Whisper models in one pass
- CUDA GPU acceleration with automatic CPU fallback
- Custom hotwords/dictionary to bias transcription toward expected terms
- Resizable log + live colored transcript preview
- Collapsible file queue, per-file progress and status

## Requirements

- **Windows 10/11** (the app targets Windows; see note below for other platforms)
- **Python 3.11+** (3.13 recommended)
- **[ffmpeg](https://ffmpeg.org/)** — used to extract audio from video and for subtitle burn-in.
  Not strictly required to launch the app, since the `imageio-ffmpeg` package provides a
  bundled fallback binary, but installing ffmpeg separately (`winget install Gyan.FFmpeg`)
  is more reliable, especially for burn-in.

### Python libraries

Installed automatically via `requirements.txt`:

| Library | Purpose |
|---|---|
| `PySide6` | GUI framework |
| `faster-whisper` | Whisper transcription engine (CTranslate2 backend) |
| `imageio-ffmpeg` | Fallback ffmpeg binary if none is on PATH |

Optional, in `requirements-optional.txt`:

| Library | Purpose |
|---|---|
| `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` | Enables CUDA/GPU transcription |
| `pyannote.audio`, `torch` | Enables speaker diarization |

## Installation

### Option A — one-command setup (Windows, recommended)

1. [Download](../../archive/refs/heads/main.zip) or `git clone` this repository.
2. Double-click **`install.bat`**.
3. It will:
   - find (or tell you to install) Python
   - create an isolated virtual environment in `.venv`
   - install the required packages
   - ask if you want GPU (CUDA) support
   - ask if you want speaker diarization support
   - check whether ffmpeg is on your PATH and warn you if not
4. Once it finishes, launch the app with **`run.bat`**.

### Option B — manual setup

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

REM optional — GPU acceleration
pip install -r requirements-optional.txt

REM then run
pythonw transcriber_1.01.pyw
```

### macOS / Linux

The app was built and tested on Windows (the CUDA DLL fix and taskbar App User Model ID
are Windows-specific and simply no-op elsewhere). The core transcription features should
still work on macOS/Linux with a standard `pip install -r requirements.txt` in a venv and
`python3 transcriber_1.01.pyw`, provided `ffmpeg` is installed via your package manager
(`brew install ffmpeg` / `apt install ffmpeg`). This hasn't been extensively tested outside Windows.

## Usage

1. Launch the app (`run.bat`, or `pythonw transcriber_1.01.pyw` from an activated venv).
2. Click **Add Files…** (or just click **Start** with an empty queue — it'll prompt you).
3. Pick a model, output format, language (or leave as `auto`), and device.
4. Optional: set hotwords, enable diarization, or select multiple models under **Batch Models…**.
5. Click **Start**. Progress, live transcript preview, and per-file status show as it runs.
6. Output files save next to the source file, or to the folder set under **Output folder**.

### Speaker diarization setup

1. `pip install pyannote.audio` (or answer "y" during `install.bat`).
2. Create a free account at [huggingface.co](https://huggingface.co) and accept the model
   license at [hf.co/pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1).
3. Generate an access token at huggingface.co/settings/tokens.
4. In the app, click **Diarize…**, paste the token, and check **Speaker diarization**.

Diarization fails safely — if it can't run for any reason, transcription continues normally without speaker labels.

## Configuration

Settings persist to `config.json` in the same folder as the script (created automatically on first run — not tracked in git).

## Known limitations

- GPU acceleration requires an NVIDIA card; CUDA library errors fall back to CPU automatically.
- Diarization is English-agnostic but requires the HuggingFace token step above.
- Very long files can take a while depending on model size and hardware — the `large-v3-turbo` or `distil-large-v3` models are good speed/accuracy tradeoffs.

## License

MIT — see [LICENSE](LICENSE).
