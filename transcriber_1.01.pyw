import sys
import os
import json
import time
import base64
import subprocess
import tempfile
from pathlib import Path
from shutil import which

# ── CUDA DLL fix (Windows) ────────────────────────────────────────────────────
# Adds nvidia wheel bin dirs to the DLL search path before anything touches
# ctranslate2/faster-whisper, so CUDA is actually findable at runtime.
if sys.platform == "win32":
    _nvidia_base = os.path.join(
        os.environ.get("USERPROFILE", ""),
        "AppData", "Local", "Programs", "Python", "Python313",
        "Lib", "site-packages", "nvidia"
    )
    for _sub in ("cublas\\bin", "cudnn\\bin"):
        _p = os.path.join(_nvidia_base, _sub)
        if os.path.isdir(_p):
            os.add_dll_directory(_p)

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QIcon, QPixmap, QColor, QTextCharFormat, QTextCursor, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QListWidget, QListWidgetItem, QPushButton, QComboBox, QLineEdit, QLabel,
    QProgressBar, QPlainTextEdit, QTextEdit, QFileDialog, QDialog, QDialogButtonBox,
    QGroupBox, QMessageBox, QCheckBox, QSplitter, QFrame, QScrollArea,
    QToolButton, QSizePolicy
)

VERSION = "1.01"

# ── Embedded icon (64×64 RGBA microphone, base64 PNG ~495 bytes) ──────────────
_ICON_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAABtklEQVR42u2bzZHCMAyFIw11cIZG"
    "KGYL2mJoBM40AqfMwG5sZPtJtmLpyCRY79NPYiVZlrC5jSwXOx5PT8lxj8eddgFAKrgnEBpVtB"
    "UM8iJcCwR5E44GwZ7FI3xgz+IRvpBn4YiSoF7iL7+3f79df87mEMha/JZwDRBSCDya+JLjED6z"
    "VW2WikJAgABARL9WTCsEie/sueMjIPAodd+rH/AyufFeU1+qKTJghujntEUGBIBJ0j+lMTIgAASA"
    "eep/S2tkQAAIAAEgAGha64QXNSpP2aHlZM3ZvtW6JiVQ65wFTLMeUCrGQvwHgJrHzFtO5oaYUlG5"
    "4xDp/671sBjb6myv/gFtgrkofRNTI1bjaRFr1Tba2dT/tWYNp2pjJAhI8X81Um6riExRdMrXRl4V"
    "gDTqpV0emfZfAaCGIxoNq7Xet0pcDQAShOarM8mmpzEiK4GBvidINXhTAD03UikAXHqCR8tp4doT"
    "9yBedCfoGYLE9xiJoUh6jH72KtB6Zeh5I1QSMNag6iHy1fOAdYHRniXWBoetFxxJfPNVYAQIrT7E"
    "R1Nox6b9bE4ThqsPJxFA9rQRCxvdXsRC+R4xpHi1AAAAAElFTkSuQmCC"
)

def _app_icon():
    pm = QPixmap()
    pm.loadFromData(base64.b64decode(_ICON_B64))
    return QIcon(pm)

# ── Constants ─────────────────────────────────────────────────────────────────
VIDEO_EXTS = ["mp4","mkv","mov","avi","webm","flv","m4v","wmv","mpg","mpeg","ts"]
AUDIO_EXTS = ["mp3","wav","m4a","aac","flac","ogg","opus","wma"]
ALL_EXTS   = VIDEO_EXTS + AUDIO_EXTS

# Model list — includes turbo and distil variants worth trying
MODELS = [
    "large-v3",           # best overall accuracy
    "large-v3-turbo",     # ~8x faster than large-v3, nearly same accuracy
    "large-v2",           # previous gen; sometimes better on accents/literary content
    "distil-large-v3",    # distilled large-v3: fast, very accurate, English-only
    "distil-large-v2",    # distilled large-v2: fast, English-only
    "medium",
    "medium.en",          # English-only medium; slightly faster/more accurate for EN
    "small",
    "small.en",
    "base",
]

DEVICE_LABELS = ["Auto (CUDA, fallback CPU)", "CUDA", "CPU"]
DEVICE_VALUES = ["auto", "cuda", "cpu"]

OUTPUT_MODES = {
    "plain":       ("Plain text",       ".txt"),
    "timestamped": ("Timestamped text", ".txt"),
    "srt":         ("SRT subtitles",    ".srt"),
    "vtt":         ("WebVTT subtitles", ".vtt"),
    "ass":         ("ASS subtitles",    ".ass"),
}
OUTPUT_KEYS   = list(OUTPUT_MODES.keys())
OUTPUT_LABELS = [v[0] for v in OUTPUT_MODES.values()]

WATCHDOG_SECONDS = 60
EXTRACT_TIMEOUT  = 600
BURNIN_TIMEOUT   = 1800

# Preview text colors
COLOR_TIMESTAMP = "#6ab0f5"   # soft blue for timestamps
COLOR_TEXT      = "#e0e0e0"   # near-white for transcript text
COLOR_DIVIDER   = "#888888"   # grey for file dividers

# Speaker diarization colors — male palette / female palette / fallback cycle
SPEAKER_COLORS_M = ["#5b9bd5","#4fc3f7","#80cbc4","#aed581","#7986cb"]
SPEAKER_COLORS_F = ["#f48fb1","#f06292","#ce93d8","#ffb74d","#ff8a65"]
SPEAKER_COLORS_U = ["#a5d6a7","#fff176","#b0bec5","#ef9a9a","#80deea"]

ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

DEFAULT_CONFIG = {
    "model":          "large-v3",
    "batch_models":   [],
    "output_mode":    "plain",
    "language":       "auto",
    "output_folder":  "",
    "hotwords":       [],
    "device":         "auto",
    "use_vad":        False,
    "burn_subs":      False,
    "use_diarize":    False,
    "hf_token":       "",
}


# ── Config ────────────────────────────────────────────────────────────────────
def load_config():
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg = dict(DEFAULT_CONFIG)
            cfg.update({k: data[k] for k in DEFAULT_CONFIG if k in data})
            return cfg
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

def save_config(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── Timestamp formatters ──────────────────────────────────────────────────────
def _ts_srt(s):
    ms = int(round(s * 1000))
    h, ms = divmod(ms, 3_600_000); m, ms = divmod(ms, 60_000); sec, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

def _ts_vtt(s):  return _ts_srt(s).replace(",", ".")

def _ts_ass(s):
    cs = int(round(s * 100))
    h, cs = divmod(cs, 360_000); m, cs = divmod(cs, 6_000); sec, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"

def _ts_readable(s, long_form):
    s = int(s); h, r = divmod(s, 3600); m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if long_form else f"{m:02d}:{sec:02d}"


# ── Subtitle builder ──────────────────────────────────────────────────────────
def build_subtitle_text(segments, mode):
    """segments: list of (start, end, text[, speaker])"""
    lines = []
    if mode == "srt":
        for i, seg in enumerate(segments, 1):
            lines += [str(i), f"{_ts_srt(seg[0])} --> {_ts_srt(seg[1])}", seg[2], ""]
    elif mode == "vtt":
        lines.append("WEBVTT\n")
        for i, seg in enumerate(segments, 1):
            lines += [str(i), f"{_ts_vtt(seg[0])} --> {_ts_vtt(seg[1])}", seg[2], ""]
    elif mode == "ass":
        lines.append(ASS_HEADER)
        for seg in segments:
            safe = seg[2].replace("\\","\\\\").replace("{","\\{").replace("\n","\\N")
            lines.append(f"Dialogue: 0,{_ts_ass(seg[0])},{_ts_ass(seg[1])},Default,,0,0,0,,{safe}")
    return "\n".join(lines)


# ── ffmpeg ────────────────────────────────────────────────────────────────────
def resolve_ffmpeg():
    ff = which("ffmpeg")
    if ff: return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# ── Diarization (optional, fail-safe) ────────────────────────────────────────
def diarize_available():
    try:
        from pyannote.audio import Pipeline
        return True
    except Exception:
        return False

def run_diarization(wav_path, hf_token, num_speakers=None):
    """Returns a list of (start, end, speaker_label) or None on any failure."""
    try:
        from pyannote.audio import Pipeline
        import torch
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        if torch.cuda.is_available():
            pipe = pipe.to(torch.device("cuda"))
        kwargs = {}
        if num_speakers:
            kwargs["num_speakers"] = num_speakers
        diarization = pipe(wav_path, **kwargs)
        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append((turn.start, turn.end, speaker))
        return result
    except Exception:
        return None

def assign_speaker(diarization, seg_start, seg_end):
    """Find the speaker with most overlap for a given segment."""
    if not diarization:
        return None
    best_speaker, best_overlap = None, 0.0
    for d_start, d_end, speaker in diarization:
        overlap = max(0.0, min(seg_end, d_end) - max(seg_start, d_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


# ── Worker thread ─────────────────────────────────────────────────────────────
ST_WAIT = "wait"; ST_RUN = "run"; ST_DONE = "done"; ST_ERR = "err"
_PREFIX = {ST_WAIT:"[  ]  ", ST_RUN:"[ > ] ", ST_DONE:"[ OK ] ", ST_ERR:"[ERR] "}


class TranscribeWorker(QThread):
    device_chosen = Signal(str)
    log           = Signal(str)
    file_started  = Signal(int)
    file_progress = Signal(int, float)
    # segment: (timestamp_str, text, speaker_or_None)
    segment       = Signal(str, str, object)
    file_done     = Signal(int, str)
    file_error    = Signal(int, str)
    all_done      = Signal()

    def __init__(self, files, cfg, models_override=None):
        super().__init__()
        self.files           = files
        self.cfg             = cfg
        self.models_override = models_override  # list of model names for batch-model mode
        self._stop           = False
        self.last_activity   = time.time()
        self.current_stage   = "idle"

    def request_stop(self):
        self._stop = True

    def _stage(self, name):
        self.current_stage = name
        self.last_activity = time.time()
        self.log.emit(name)

    def _ping(self):
        self.last_activity = time.time()

    @staticmethod
    def _cuda_available():
        try:
            import ctranslate2
            return len(ctranslate2.get_supported_compute_types("cuda")) > 0
        except Exception:
            return False

    def _load_model(self, model_name):
        from faster_whisper import WhisperModel
        dev = self.cfg.get("device", "auto")
        if dev == "cpu":
            return WhisperModel(model_name, device="cpu", compute_type="int8"), f"CPU (int8)"
        if dev == "cuda":
            if not self._cuda_available():
                raise RuntimeError(
                    "CUDA selected but cuBLAS/cuDNN libraries not found.\n"
                    "Install: py -3.13 -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12\n"
                    "Or switch Device to CPU.")
            return WhisperModel(model_name, device="cuda", compute_type="float16"), "CUDA (float16)"
        if self._cuda_available():
            try:
                return WhisperModel(model_name, device="cuda", compute_type="float16"), "CUDA (float16)"
            except Exception as e:
                self.log.emit(f"CUDA init failed ({type(e).__name__}); falling back to CPU.")
        else:
            self.log.emit("CUDA libraries not found; using CPU.")
        return WhisperModel(model_name, device="cpu", compute_type="int8"), "CPU (int8)"

    def run(self):
        try:
            self._run_inner()
        except Exception:
            import traceback
            self.file_error.emit(-1, f"Unexpected crash:\n{traceback.format_exc()}")
            self.all_done.emit()

    def _run_inner(self):
        models_to_run = self.models_override or [self.cfg["model"]]

        lang      = self.cfg["language"].strip().lower()
        language  = None if lang in ("", "auto") else lang
        hot       = ", ".join(self.cfg.get("hotwords", []) or [])
        hotwords  = hot if hot.strip() else None
        mode      = self.cfg["output_mode"]
        use_vad   = bool(self.cfg.get("use_vad", False))
        burn_subs = bool(self.cfg.get("burn_subs", False))
        do_burn   = burn_subs and mode in ("srt","vtt","ass")
        use_diar  = bool(self.cfg.get("use_diarize", False))
        hf_token  = self.cfg.get("hf_token","").strip()

        # Load first model — subsequent models in batch reloaded as needed
        try:
            self._stage(f"Loading model: {models_to_run[0]}...")
            model, dev_info = self._load_model(models_to_run[0])
            current_model_name = models_to_run[0]
            self.device_chosen.emit(f"{dev_info} — model: {current_model_name}")
            self.log.emit(f"Model ready.")
        except Exception as e:
            self.file_error.emit(-1, f"Could not load model:\n{e}")
            self.all_done.emit()
            return

        # Outer loop: files; inner loop: models (for batch-model mode)
        for file_idx, src in enumerate(self.files):
            if self._stop: break
            for model_name in models_to_run:
                if self._stop: break
                # Switch model if needed
                if model_name != current_model_name:
                    try:
                        self._stage(f"Loading model: {model_name}...")
                        model, dev_info = self._load_model(model_name)
                        current_model_name = model_name
                        self.device_chosen.emit(f"{dev_info} — model: {current_model_name}")
                    except Exception as e:
                        self.file_error.emit(file_idx, f"Could not load {model_name}: {e}")
                        continue

                queue_idx = file_idx * len(models_to_run) + models_to_run.index(model_name)
                self.file_started.emit(queue_idx)
                try:
                    out = self._transcribe_one(
                        model, model_name, src, language, hotwords,
                        mode, use_vad, do_burn, use_diar, hf_token, queue_idx
                    )
                    if out and not self._stop:
                        self.file_done.emit(queue_idx, out)
                except Exception as e:
                    self.file_error.emit(queue_idx, f"{type(e).__name__}: {e}")

        self.current_stage = "done"
        self.all_done.emit()

    def _extract_audio(self, src_path):
        ff = resolve_ffmpeg()
        if not ff:
            raise RuntimeError("No ffmpeg found. Install: winget install Gyan.FFmpeg")
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        cmd = [ff, "-y", "-i", str(src_path), "-vn", "-ar", "16000", "-ac", "1", tmp]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            proc = subprocess.run(cmd, capture_output=True, creationflags=flags,
                                  timeout=EXTRACT_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg audio extraction timed out.")
        if proc.returncode != 0:
            raise RuntimeError("ffmpeg extraction failed:\n" +
                               proc.stderr.decode("utf-8","ignore")[-500:])
        return tmp

    def _burn_subtitles(self, src_path, sub_path):
        ff = resolve_ffmpeg()
        if not ff: raise RuntimeError("No ffmpeg for burn-in.")
        out = src_path.with_stem(src_path.stem + "_subtitled")
        sub_esc = str(sub_path).replace("\\","/").replace(":","\\:")
        cmd = [ff, "-y", "-i", str(src_path),
               "-vf", f"subtitles='{sub_esc}'", "-c:a", "copy", str(out)]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._stage(f"Burning subtitles: {src_path.name}")
        try:
            proc = subprocess.run(cmd, capture_output=True, creationflags=flags,
                                  timeout=BURNIN_TIMEOUT)
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffmpeg burn-in timed out.")
        if proc.returncode != 0:
            raise RuntimeError("Burn-in failed:\n" +
                               proc.stderr.decode("utf-8","ignore")[-800:])
        self.log.emit(f"Burn-in: {out.name}")
        return out

    def _transcribe_one(self, model, model_name, src, language, hotwords,
                        mode, use_vad, do_burn, use_diar, hf_token, queue_idx):
        src_path = Path(src)
        temp_wav = None
        try:
            self._stage(f"Extracting audio: {src_path.name}")
            temp_wav = self._extract_audio(src_path)
            self.log.emit("Audio extracted.")

            # Optional diarization — runs before transcription, fails safely
            diarization = None
            if use_diar:
                if not diarize_available():
                    self.log.emit("⚠ Diarization skipped: pyannote.audio not installed.")
                elif not hf_token:
                    self.log.emit("⚠ Diarization skipped: no HuggingFace token set.")
                else:
                    self._stage("Running speaker diarization...")
                    diarization = run_diarization(temp_wav, hf_token)
                    if diarization is None:
                        self.log.emit("⚠ Diarization failed — continuing without speaker labels.")
                    else:
                        self.log.emit(f"Diarization found {len(set(d[2] for d in diarization))} speakers.")

            self._stage(f"Transcribing [{model_name}]: {src_path.name} (VAD: {'on' if use_vad else 'off'})")
            segs_gen, info = model.transcribe(
                temp_wav, language=language, hotwords=hotwords,
                vad_filter=use_vad, beam_size=5,
            )
            total = getattr(info, "duration", 0) or 0
            self.log.emit(f"Length {_ts_readable(total, total>=3600)}; "
                          f"lang: {getattr(info,'language','?')}.")
            long_form = total >= 3600
            collected = []

            for seg in segs_gen:
                if self._stop: break
                self._ping()
                text    = seg.text.strip()
                speaker = assign_speaker(diarization, seg.start, seg.end)
                ts_str  = _ts_srt(seg.start) if mode in ("srt","vtt","ass") \
                          else _ts_readable(seg.start, long_form)
                collected.append((seg.start, seg.end, text, speaker))
                self.segment.emit(ts_str, text, speaker)
                if total > 0:
                    self.file_progress.emit(queue_idx, min(seg.end / total, 1.0))

            if self._stop: return ""

            # Build output
            segs_out = [(s,e,t) for s,e,t,_ in collected]
            if mode == "plain":
                content = "\n".join(t for _,_,t,_ in collected) + "\n"
            elif mode == "timestamped":
                content = "\n".join(
                    f"[{_ts_readable(s,long_form)}] {t}" for s,_,t,_ in collected
                ) + "\n"
            else:
                content = build_subtitle_text(segs_out, mode)

            # If batch-model mode, suffix the model name to avoid overwrites
            suffix = f"_{model_name.replace('/','_')}" if self.models_override and len(self.models_override) > 1 else ""
            ext = OUTPUT_MODES[mode][1]
            out_path = self._output_path_for(src_path, ext, suffix)
            Path(out_path).write_text(content, encoding="utf-8")
            self.file_progress.emit(queue_idx, 1.0)
            self.log.emit(f"Saved: {Path(out_path).name}")

            if do_burn and not self._stop:
                self._burn_subtitles(src_path, Path(out_path))

            return out_path
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try: os.remove(temp_wav)
                except Exception: pass

    def _output_path_for(self, src_path, ext, suffix=""):
        folder = self.cfg.get("output_folder","").strip()
        name   = src_path.stem + suffix + ext
        return str(Path(folder) / name) if folder else str(src_path.parent / name)


# ── Hotwords dialog ───────────────────────────────────────────────────────────
class HotwordsDialog(QDialog):
    def __init__(self, parent, hotwords):
        super().__init__(parent)
        self.setWindowTitle("Edit Hotwords / Custom Dictionary")
        self.resize(440, 460)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "One term or phrase per line.\n"
            "These bias Whisper toward expected words (jargon, proper nouns, etc).\n"
            "Saved between sessions. Leave empty for generic use."
        ))
        self.edit = QPlainTextEdit("\n".join(hotwords))
        lay.addWidget(self.edit)
        row = QHBoxLayout()
        clr = QPushButton("Clear all"); clr.clicked.connect(self.edit.clear)
        row.addWidget(clr); row.addStretch()
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def hotwords(self):
        return [l.strip() for l in self.edit.toPlainText().splitlines() if l.strip()]


# ── Diarization settings dialog ───────────────────────────────────────────────
class DiarizeDialog(QDialog):
    def __init__(self, parent, token):
        super().__init__(parent)
        self.setWindowTitle("Speaker Diarization Settings")
        self.resize(480, 280)
        lay = QVBoxLayout(self)
        avail = diarize_available()
        status = "✓ pyannote.audio installed" if avail else "✗ pyannote.audio not installed"
        status_lbl = QLabel(status)
        status_lbl.setStyleSheet(f"color: {'#80c080' if avail else '#c08080'};")
        lay.addWidget(status_lbl)
        if not avail:
            lay.addWidget(QLabel(
                "To enable diarization:\n"
                "  py -3.13 -m pip install pyannote.audio\n\n"
                "You also need a free HuggingFace account and must accept\n"
                "the pyannote/speaker-diarization-3.1 model license at:\n"
                "  https://hf.co/pyannote/speaker-diarization-3.1"
            ))
        lay.addWidget(QLabel("HuggingFace access token:"))
        self.token_edit = QLineEdit(token)
        self.token_edit.setPlaceholderText("hf_...")
        self.token_edit.setEchoMode(QLineEdit.Password)
        lay.addWidget(self.token_edit)
        lay.addWidget(QLabel(
            "If diarization is enabled but fails for any reason,\n"
            "transcription continues normally without speaker labels."
        ))
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def token(self):
        return self.token_edit.text().strip()


# ── Batch model selector dialog ───────────────────────────────────────────────
class BatchModelDialog(QDialog):
    def __init__(self, parent, selected):
        super().__init__(parent)
        self.setWindowTitle("Batch Model Output — Select Models")
        self.resize(340, 380)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "Select multiple models. Each source file will be\n"
            "transcribed by every selected model and output\n"
            "separately (filename includes model name)."
        ))
        self.checks = {}
        for m in MODELS:
            cb = QCheckBox(m)
            cb.setChecked(m in selected)
            lay.addWidget(cb)
            self.checks[m] = cb
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def selected_models(self):
        return [m for m, cb in self.checks.items() if cb.isChecked()]


# ── Main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Whisper Transcriber  v{VERSION}")
        self.setWindowIcon(_app_icon())
        self.resize(820, 860)
        self.cfg      = load_config()
        self.worker   = None
        self.queue    = []   # list of (src_path, model_name) tuples
        self._wd_last = None
        self._speaker_color_map = {}

        # App User Model ID — makes Windows treat this as its own taskbar entry
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    f"jer.whispertranscriber.{VERSION}")
            except Exception:
                pass

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)

        # ── Files section (collapsible) ───────────────────────────────────────
        files_header = QHBoxLayout()
        self._files_toggle = QToolButton()
        self._files_toggle.setText("▶  Files")
        self._files_toggle.setCheckable(True)
        self._files_toggle.setChecked(False)
        self._files_toggle.setStyleSheet("QToolButton { border:none; font-weight:bold; }")
        self._files_toggle.clicked.connect(self._toggle_files)
        self._files_count_lbl = QLabel("(no files)")
        files_header.addWidget(self._files_toggle)
        files_header.addWidget(self._files_count_lbl)
        files_header.addStretch()
        root.addLayout(files_header)

        self._files_panel = QWidget()
        fp_lay = QVBoxLayout(self._files_panel)
        fp_lay.setContentsMargins(0,0,0,0)
        self.list = QListWidget()
        self.list.setMaximumHeight(160)
        fp_lay.addWidget(self.list)
        qbtns = QHBoxLayout()
        for text, fn in (("Add Files…", self.add_files),
                         ("Remove Selected", self.remove_selected),
                         ("Clear", self.clear_queue)):
            b = QPushButton(text); b.clicked.connect(fn); qbtns.addWidget(b)
        qbtns.addStretch()
        fp_lay.addLayout(qbtns)
        self._files_panel.setVisible(False)
        root.addWidget(self._files_panel)

        # ── Settings ──────────────────────────────────────────────────────────
        sbox = QGroupBox("Settings")
        grid = QGridLayout(sbox)

        grid.addWidget(QLabel("Model:"), 0, 0)
        self.model_cb = QComboBox(); self.model_cb.addItems(MODELS)
        self.model_cb.setCurrentText(self.cfg["model"])
        self.model_cb.currentTextChanged.connect(self._on_settings_changed)
        grid.addWidget(self.model_cb, 0, 1)

        grid.addWidget(QLabel("Output:"), 0, 2)
        self.out_cb = QComboBox(); self.out_cb.addItems(OUTPUT_LABELS)
        cur = self.cfg.get("output_mode","plain")
        self.out_cb.setCurrentIndex(OUTPUT_KEYS.index(cur) if cur in OUTPUT_KEYS else 0)
        self.out_cb.currentIndexChanged.connect(self._on_output_mode_changed)
        grid.addWidget(self.out_cb, 0, 3)

        grid.addWidget(QLabel("Language:"), 1, 0)
        self.lang_edit = QLineEdit(self.cfg["language"])
        self.lang_edit.setPlaceholderText("auto (or e.g. en, fr, de)")
        self.lang_edit.editingFinished.connect(self._on_settings_changed)
        grid.addWidget(self.lang_edit, 1, 1)

        grid.addWidget(QLabel("Device:"), 1, 2)
        self.dev_cb = QComboBox(); self.dev_cb.addItems(DEVICE_LABELS)
        self.dev_cb.setCurrentIndex(DEVICE_VALUES.index(self.cfg.get("device","auto")))
        self.dev_cb.currentIndexChanged.connect(self._on_settings_changed)
        grid.addWidget(self.dev_cb, 1, 3)

        grid.addWidget(QLabel("Output folder:"), 2, 0)
        self.outdir_edit = QLineEdit(self.cfg["output_folder"])
        self.outdir_edit.setPlaceholderText("(empty = save next to source files)")
        self.outdir_edit.editingFinished.connect(self._on_settings_changed)
        grid.addWidget(self.outdir_edit, 2, 1, 1, 2)
        ob = QPushButton("Browse…"); ob.clicked.connect(self.browse_outdir)
        grid.addWidget(ob, 2, 3)

        # Option checkboxes row
        opt_row = QHBoxLayout()
        self.vad_chk = QCheckBox("VAD filter")
        self.vad_chk.setToolTip("Skip silence before transcribing. Off by default — can cause a long pre-scan on some files.")
        self.vad_chk.setChecked(bool(self.cfg.get("use_vad",False)))
        self.vad_chk.stateChanged.connect(self._on_settings_changed)

        self.burn_chk = QCheckBox("Burn subtitles into video")
        self.burn_chk.setToolTip("Creates a _subtitled copy with subs burned in. Only active for SRT/VTT/ASS output modes.")
        self.burn_chk.setChecked(bool(self.cfg.get("burn_subs",False)))
        self.burn_chk.stateChanged.connect(self._on_settings_changed)

        self.diar_chk = QCheckBox("Speaker diarization")
        self.diar_chk.setToolTip(
            "Color transcript by speaker. Requires pyannote.audio + HuggingFace token.\n"
            "Fails safely — transcript still works if diarization can't run.")
        self.diar_chk.setChecked(bool(self.cfg.get("use_diarize",False)))
        self.diar_chk.stateChanged.connect(self._on_settings_changed)

        hw_btn    = QPushButton("Hotwords…");   hw_btn.clicked.connect(self.edit_hotwords)
        diar_btn  = QPushButton("Diarize…");    diar_btn.clicked.connect(self.edit_diarize)
        batch_btn = QPushButton("Batch Models…"); batch_btn.clicked.connect(self.edit_batch_models)

        opt_row.addWidget(self.vad_chk)
        opt_row.addWidget(self.burn_chk)
        opt_row.addWidget(self.diar_chk)
        opt_row.addStretch()
        opt_row.addWidget(hw_btn)
        opt_row.addWidget(diar_btn)
        opt_row.addWidget(batch_btn)
        grid.addLayout(opt_row, 3, 0, 1, 4)

        self._update_burn_state()
        root.addWidget(sbox)

        # ── Status ────────────────────────────────────────────────────────────
        self.device_lbl = QLabel("Device: (determined when you start)")
        root.addWidget(self.device_lbl)
        self.progress = QProgressBar(); self.progress.setRange(0,100)
        root.addWidget(self.progress)

        # ── Log + Preview (resizable splitter) ────────────────────────────────
        splitter = QSplitter(Qt.Vertical)

        log_widget = QWidget()
        log_lay = QVBoxLayout(log_widget); log_lay.setContentsMargins(0,0,0,0)
        log_lay.addWidget(QLabel("Log:"))
        self.log_panel = QPlainTextEdit(); self.log_panel.setReadOnly(True)
        self.log_panel.setFont(QFont("Consolas", 9))
        log_lay.addWidget(self.log_panel)
        splitter.addWidget(log_widget)

        prev_widget = QWidget()
        prev_lay = QVBoxLayout(prev_widget); prev_lay.setContentsMargins(0,0,0,0)
        prev_lay.addWidget(QLabel("Live transcript preview:"))
        self.preview = QTextEdit(); self.preview.setReadOnly(True)
        self.preview.setFont(QFont("Consolas", 10))
        prev_lay.addWidget(self.preview)
        splitter.addWidget(prev_widget)

        splitter.setSizes([180, 400])
        root.addWidget(splitter, stretch=1)

        # ── Run controls ──────────────────────────────────────────────────────
        run_row = QHBoxLayout()
        self.start_btn = QPushButton("Start"); self.start_btn.clicked.connect(self.start)
        self.stop_btn  = QPushButton("Stop");  self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        run_row.addWidget(self.start_btn); run_row.addWidget(self.stop_btn); run_row.addStretch()
        root.addLayout(run_row)

        self.watchdog = QTimer(self)
        self.watchdog.setInterval(5000)
        self.watchdog.timeout.connect(self._check_watchdog)

    # ── Files collapse ────────────────────────────────────────────────────────
    def _toggle_files(self, checked):
        self._files_panel.setVisible(checked)
        self._files_toggle.setText(("▼" if checked else "▶") + "  Files")

    def _update_files_label(self):
        n = len(self.queue)
        self._files_count_lbl.setText(f"({n} file{'s' if n!=1 else ''})" if n else "(no files)")
        # Auto-expand when multiple files
        if n > 1 and not self._files_toggle.isChecked():
            self._files_toggle.setChecked(True)
            self._toggle_files(True)

    # ── Settings ──────────────────────────────────────────────────────────────
    def _gather_cfg(self):
        self.cfg["model"]         = self.model_cb.currentText()
        self.cfg["output_mode"]   = OUTPUT_KEYS[self.out_cb.currentIndex()]
        self.cfg["language"]      = self.lang_edit.text().strip() or "auto"
        self.cfg["output_folder"] = self.outdir_edit.text().strip()
        self.cfg["device"]        = DEVICE_VALUES[self.dev_cb.currentIndex()]
        self.cfg["use_vad"]       = self.vad_chk.isChecked()
        self.cfg["burn_subs"]     = self.burn_chk.isChecked()
        self.cfg["use_diarize"]   = self.diar_chk.isChecked()
        return self.cfg

    def _on_settings_changed(self, *_):
        save_config(self._gather_cfg())

    def _on_output_mode_changed(self, *_):
        self._update_burn_state()
        self._on_settings_changed()

    def _update_burn_state(self):
        mode = OUTPUT_KEYS[self.out_cb.currentIndex()]
        ok = mode in ("srt","vtt","ass")
        self.burn_chk.setEnabled(ok)
        if not ok: self.burn_chk.setChecked(False)

    # ── Queue ─────────────────────────────────────────────────────────────────
    def add_files(self):
        patterns = " ".join(f"*.{e}" for e in ALL_EXTS)
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add media files", "", f"Media files ({patterns});;All files (*.*)")
        for f in files:
            if f not in self.queue:
                self.queue.append(f)
                item = QListWidgetItem(_PREFIX[ST_WAIT] + Path(f).name)
                item.setData(Qt.UserRole, ST_WAIT); item.setToolTip(f)
                self.list.addItem(item)
        self._update_files_label()

    def remove_selected(self):
        for item in self.list.selectedItems():
            row = self.list.row(item)
            self.list.takeItem(row); del self.queue[row]
        self._update_files_label()

    def clear_queue(self):
        self.list.clear(); self.queue.clear()
        self._update_files_label()

    def _set_item_state(self, idx, state, suffix=""):
        if 0 <= idx < self.list.count():
            item = self.list.item(idx)
            # In batch-model mode the list has file×model entries
            src_idx = idx // max(1, len(self.cfg.get("batch_models",[]) or [self.cfg["model"]]))
            if src_idx < len(self.queue):
                name = Path(self.queue[src_idx]).name
                item.setText(_PREFIX[state] + name + (f"  {suffix}" if suffix else ""))
            item.setData(Qt.UserRole, state)

    def browse_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d:
            self.outdir_edit.setText(d); self._on_settings_changed()

    def edit_hotwords(self):
        dlg = HotwordsDialog(self, self.cfg.get("hotwords",[]))
        if dlg.exec() == QDialog.Accepted:
            self.cfg["hotwords"] = dlg.hotwords(); save_config(self.cfg)

    def edit_diarize(self):
        dlg = DiarizeDialog(self, self.cfg.get("hf_token",""))
        if dlg.exec() == QDialog.Accepted:
            self.cfg["hf_token"] = dlg.token(); save_config(self.cfg)

    def edit_batch_models(self):
        dlg = BatchModelDialog(self, self.cfg.get("batch_models",[]))
        if dlg.exec() == QDialog.Accepted:
            self.cfg["batch_models"] = dlg.selected_models(); save_config(self.cfg)

    # ── Colored preview ───────────────────────────────────────────────────────
    def _speaker_color(self, speaker):
        """Assign a consistent color to each speaker label."""
        if speaker is None:
            return COLOR_TEXT
        if speaker not in self._speaker_color_map:
            idx = len(self._speaker_color_map)
            # Guess gender from pyannote label convention (SPEAKER_00, etc.)
            # — no reliable gender from label alone, so cycle through neutral palette
            # unless diarization provides gender hints (future extension).
            palette = SPEAKER_COLORS_U
            self._speaker_color_map[speaker] = palette[idx % len(palette)]
        return self._speaker_color_map[speaker]

    def _append_preview(self, ts_str, text, speaker):
        cursor = self.preview.textCursor()
        cursor.movePosition(QTextCursor.End)

        def append_colored(s, color):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            cursor.insertText(s, fmt)

        if ts_str:
            append_colored(f"[{ts_str}] ", COLOR_TIMESTAMP)
        if speaker:
            append_colored(f"[{speaker}] ", self._speaker_color(speaker))
        append_colored(text + "\n", self._speaker_color(speaker) if speaker else COLOR_TEXT)
        self.preview.setTextCursor(cursor)
        self.preview.ensureCursorVisible()

    def _append_divider(self, name):
        cursor = self.preview.textCursor()
        cursor.movePosition(QTextCursor.End)
        fmt = QTextCharFormat(); fmt.setForeground(QColor(COLOR_DIVIDER))
        cursor.insertText(f"\n─── {name} ───\n", fmt)
        self.preview.setTextCursor(cursor)

    # ── Log ───────────────────────────────────────────────────────────────────
    def _log(self, msg):
        self.log_panel.appendPlainText(time.strftime("%H:%M:%S  ") + msg)

    # ── Run ───────────────────────────────────────────────────────────────────
    def start(self):
        if not self.queue:
            # If no files in list, open file dialog directly
            self.add_files()
            if not self.queue: return

        outdir = self.outdir_edit.text().strip()
        if outdir and not Path(outdir).is_dir():
            QMessageBox.warning(self, "Bad output folder",
                                "The output folder doesn't exist.")
            return

        self._gather_cfg(); save_config(self.cfg)

        batch_models = self.cfg.get("batch_models",[])
        models_to_run = batch_models if batch_models else [self.cfg["model"]]

        # Build flat queue: file × model entries in list widget
        self.list.clear()
        for src in self.queue:
            for mname in models_to_run:
                label = f"{_PREFIX[ST_WAIT]}{Path(src).name}"
                if len(models_to_run) > 1:
                    label += f"  [{mname}]"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, ST_WAIT)
                self.list.addItem(item)

        if not self._files_toggle.isChecked():
            self._files_toggle.setChecked(True)
            self._toggle_files(True)

        self.preview.clear(); self.log_panel.clear()
        self.progress.setValue(0); self._wd_last = None
        self._speaker_color_map.clear()

        self._set_controls_running(True)
        self.worker = TranscribeWorker(
            list(self.queue), dict(self.cfg),
            models_override=models_to_run if batch_models else None
        )
        self.worker.device_chosen.connect(lambda s: self.device_lbl.setText(f"Device: {s}"))
        self.worker.log.connect(self._log)
        self.worker.file_started.connect(self._on_file_started)
        self.worker.file_progress.connect(self._on_file_progress)
        self.worker.segment.connect(self._on_segment)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.file_error.connect(self._on_file_error)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()
        self.watchdog.start()

    def stop(self):
        if self.worker:
            self.worker.request_stop()
            self._log("Stop requested...")
            self.stop_btn.setEnabled(False)

    def _set_controls_running(self, running):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        for w in (self.model_cb, self.out_cb, self.lang_edit, self.dev_cb,
                  self.outdir_edit, self.vad_chk, self.burn_chk, self.diar_chk):
            w.setEnabled(not running)
        if not running: self._update_burn_state()

    def _check_watchdog(self):
        w = self.worker
        if not w or not w.isRunning(): return
        idle  = time.time() - w.last_activity
        stage = w.current_stage
        if idle >= WATCHDOG_SECONDS and stage not in ("idle","done"):
            key = (stage, int(idle // 30))
            if key != self._wd_last:
                self._wd_last = key
                self._log(f"⚠ No progress for ~{int(idle)}s in: {stage} — you can Stop.")

    def _on_file_started(self, idx):
        self._set_item_state(idx, ST_RUN, "(working)")
        self.progress.setValue(0); self._wd_last = None
        # figure out which src
        models_to_run = self.cfg.get("batch_models",[]) or [self.cfg["model"]]
        src_idx = idx // len(models_to_run)
        if src_idx < len(self.queue):
            self._append_divider(Path(self.queue[src_idx]).name +
                                 (f" [{models_to_run[idx % len(models_to_run)]}]"
                                  if len(models_to_run) > 1 else ""))

    def _on_file_progress(self, idx, frac):
        self.progress.setValue(int(frac * 100))

    def _on_segment(self, ts_str, text, speaker):
        self._append_preview(ts_str, text, speaker)

    def _on_file_done(self, idx, out_path):
        self._set_item_state(idx, ST_DONE, f"→ {Path(out_path).name}")

    def _on_file_error(self, idx, msg):
        if idx < 0:
            QMessageBox.critical(self, "Error", msg)
            self._log("Failed: " + msg.splitlines()[0])
            return
        self._set_item_state(idx, ST_ERR, "(see log)")
        self._log(f"ERROR: {msg.splitlines()[0]}")

    def _on_all_done(self):
        self.watchdog.stop()
        self._set_controls_running(False)
        self._log("All files processed.")

    def _on_worker_finished(self):
        self.worker = None

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(3000)
        save_config(self._gather_cfg())
        super().closeEvent(event)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(_app_icon())
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
