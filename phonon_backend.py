"""Phonon-2 Speech-to-Text backend for GetShorts.

Integrates FermionResearch/Phonon-2 via the official fermion-research CPU engine
running inside the dedicated virtual environment (/home/zeeshan/models/phonon-env).

Provides:
  - Configurable model paths (PHONON_MODEL_PATH, PHONON_ENV_PYTHON, PHONON_CLI_BIN).
  - High-performance persistent worker (reuses loaded model across multiple videos).
  - Robust one-shot fallback if the persistent worker is disabled or restarted.
  - Full adherence to the GetShorts transcript data contract:
      {
        "text": str,
        "language": str,
        "segments": [
          {"start": float, "end": float, "text": str,
           "words": [{"word": str, "start": float, "end": float}, ...]}
        ]
      }
    including word leading-space convention, native Python floats, and chronological ordering.
"""
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from subtitles import merge_continuation_words

logger = logging.getLogger(__name__)

# Default configuration paths
DEFAULT_PHONON_MODEL_PATH = "/home/zeeshan/models/Phonon-2"
DEFAULT_PHONON_ENV_PYTHON = "/home/zeeshan/models/phonon-env/bin/python"
DEFAULT_PHONON_CLI_BIN = "/home/zeeshan/models/phonon-env/bin/fermion"

# Path to the helper worker script in this project
WORKER_SCRIPT_PATH = str(Path(__file__).parent / "phonon_worker.py")


def get_phonon_model_path() -> str:
    return os.environ.get("PHONON_MODEL_PATH", DEFAULT_PHONON_MODEL_PATH).strip()


def get_phonon_env_python() -> str:
    return os.environ.get("PHONON_ENV_PYTHON", DEFAULT_PHONON_ENV_PYTHON).strip()


def get_phonon_cli_bin() -> str:
    return os.environ.get("PHONON_CLI_BIN", DEFAULT_PHONON_CLI_BIN).strip()


def is_worker_enabled() -> bool:
    return os.environ.get("PHONON_WORKER_ENABLED", "1").lower() in ("1", "true", "yes")


def is_fallback_to_whisper_enabled() -> bool:
    """Return whether automatic fallback to Whisper is permitted.

    Enabled by default for production reliability (e.g. non-English audio or silent/music tracks).
    Can be disabled for strict Phonon-only mode by setting:
        PHONON_STRICT=1 or PHONON_FALLBACK_TO_WHISPER=0
    """
    if os.environ.get("PHONON_STRICT", "0").strip().lower() in ("1", "true", "yes"):
        return False
    return os.environ.get("PHONON_FALLBACK_TO_WHISPER", "1").strip().lower() in ("1", "true", "yes")


def resolve_phonon_model_dir(model_path: str = None) -> Path:
    """Resolve and validate the local Phonon-2 model directory.

    Checks for:
      1. An unpacked 'model_phonon2_c4c_int6' directory containing model.fermion and config.json.
      2. The model path itself if it directly contains model.fermion with joint/vocabulary in config.json.
      3. Automatic unpacking of 'phonon-2.bps.tar.zst' if found.
    """
    raw_path = model_path or get_phonon_model_path()
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configured PHONON_MODEL_PATH does not exist: {path}")

    # Case 1: Root repo dir with unpacked subdirectory 'model_phonon2_c4c_int6'
    unpacked_sub = path / "model_phonon2_c4c_int6"
    if unpacked_sub.is_dir() and (unpacked_sub / "model.fermion").is_file():
        return unpacked_sub

    # Case 2: Path itself is the unpacked directory with model.fermion and joint/vocab in config.json
    if (path / "model.fermion").is_file() and (path / "config.json").is_file():
        try:
            cfg = json.loads((path / "config.json").read_text())
            if "joint" in cfg or "vocabulary" in cfg:
                return path
        except Exception:
            pass

    # Case 3: Need to unpack phonon-2.bps.tar.zst if present
    tar_zst = path / "phonon-2.bps.tar.zst"
    if not tar_zst.is_file() and path.parent.is_dir():
        tar_zst = path.parent / "phonon-2.bps.tar.zst"

    if tar_zst.is_file():
        print(f"🎙️ [Phonon-2] Unpacking model weights from {tar_zst} to {unpacked_sub}...", flush=True)
        unpacked_sub.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "--zstd", "-xf", str(tar_zst), "-C", str(unpacked_sub)],
            check=True
        )
        if (unpacked_sub / "model.fermion").is_file():
            return unpacked_sub

    raise FileNotFoundError(
        f"Could not locate valid Phonon-2 model weights in {path} or {unpacked_sub}. "
        f"Expected model.fermion and config.json."
    )


class _PhononWorkerClient:
    """Manages a persistent Phonon-2 worker process for high-throughput transcription."""

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()
        self._model_dir = None

    def _ensure_proc(self):
        """Ensure the background worker process is running and ready."""
        python_bin = get_phonon_env_python()
        if not Path(python_bin).is_file():
            raise FileNotFoundError(
                f"Phonon virtual environment Python not found at {python_bin}. "
                f"Please verify PHONON_ENV_PYTHON configuration."
            )

        model_dir = resolve_phonon_model_dir()

        if self._proc is not None:
            if self._proc.poll() is None and self._model_dir == model_dir:
                return  # Process is already running with correct model
            self.terminate()

        threads = os.environ.get("PHONON_THREADS") or os.environ.get("FERMION_CPU_THREADS")
        cmd = [python_bin, WORKER_SCRIPT_PATH, "--worker", "--model-dir", str(model_dir)]
        if threads:
            cmd.extend(["--threads", str(int(threads))])

        print(f"🎙️ [Phonon-2] Launching persistent model worker from {model_dir}…", flush=True)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Wait for handshake line {"status": "ready", ...}
        t0 = time.time()
        ready_line = ""
        while time.time() - t0 < 60:
            if proc.poll() is not None:
                stderr_out = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(
                    f"Phonon worker process exited prematurely (exit code {proc.returncode}):\n{stderr_out}"
                )
            ready_line = proc.stdout.readline()
            if ready_line:
                break
            time.sleep(0.05)

        if not ready_line:
            proc.kill()
            raise TimeoutError("Phonon worker process timed out waiting for ready handshake")

        try:
            ready_data = json.loads(ready_line)
            if ready_data.get("status") != "ready":
                raise ValueError(f"Unexpected worker handshake response: {ready_line}")
            print(f"🎙️ [Phonon-2] Worker ready ({ready_data.get('engine', 'cpu')} engine, "
                  f"{ready_data.get('init_seconds', 0):.2f}s load time)", flush=True)
        except json.JSONDecodeError as exc:
            proc.kill()
            raise RuntimeError(f"Malformed worker handshake: {ready_line} ({exc})")

        self._proc = proc
        self._model_dir = model_dir

    def transcribe(self, audio_path: str, timeout: float = 1800.0) -> dict:
        """Send a transcription request to the persistent worker."""
        with self._lock:
            self._ensure_proc()
            req = {"cmd": "transcribe", "audio_path": str(audio_path)}
            try:
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                print(f"⚠️ [Phonon-2] Worker pipe broken ({exc}), restarting…", flush=True)
                self.terminate()
                self._ensure_proc()
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()

            # Read result JSON line
            resp_line = self._proc.stdout.readline()
            if not resp_line:
                stderr_out = self._proc.stderr.read() if self._proc.stderr else ""
                self.terminate()
                raise RuntimeError(
                    f"Phonon worker returned empty response (process died):\n{stderr_out}"
                )

            try:
                res = json.loads(resp_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSON from Phonon worker: {resp_line[:200]} ({exc})")

            if not res.get("success"):
                err_msg = res.get("error", "Unknown transcription error")
                raise RuntimeError(f"Phonon-2 transcription failed: {err_msg}")

            return res

    def terminate(self):
        """Cleanly terminate the worker process and release memory."""
        with self._lock:
            if self._proc is not None:
                try:
                    if self._proc.poll() is None:
                        self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                        self._proc.stdin.flush()
                        self._proc.wait(timeout=3)
                except Exception:
                    pass
                if self._proc.poll() is None:
                    try:
                        self._proc.terminate()
                        self._proc.wait(timeout=2)
                    except Exception:
                        self._proc.kill()
                self._proc = None
                self._model_dir = None
                print("🧹 [Phonon-2] Resident worker process terminated", flush=True)


# Global worker client singleton
_WORKER_CLIENT = _PhononWorkerClient()


def terminate_phonon_worker():
    """Explicitly terminate the background Phonon worker process."""
    _WORKER_CLIENT.terminate()


def _transcribe_oneshot(audio_path: str, model_dir: Path) -> dict:
    """Fallback one-shot transcription using python phonon_worker.py or CLI."""
    python_bin = get_phonon_env_python()
    if Path(python_bin).is_file():
        cmd = [python_bin, WORKER_SCRIPT_PATH, "--transcribe", str(audio_path), "--model-dir", str(model_dir)]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode == 0:
            for line in reversed(proc.stdout.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        res = json.loads(line)
                        if res.get("success"):
                            return res
                    except json.JSONDecodeError:
                        pass
        print(f"⚠️ [Phonon-2] Python one-shot failed (exit code {proc.returncode}):\n{proc.stderr}", flush=True)

    # CLI fallback via fermion binary
    cli_bin = get_phonon_cli_bin()
    if Path(cli_bin).is_file():
        cmd = [cli_bin, "transcribe", str(model_dir), str(audio_path), "--json"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode == 0:
            for line in reversed(proc.stdout.splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        raw = json.loads(line)
                        return {
                            "success": True,
                            "text": raw.get("text", ""),
                            "duration_seconds": raw.get("duration_seconds", 0.0),
                            "decode_seconds": raw.get("decode_seconds", 0.0),
                            "segments": raw.get("segments", []),
                            "words": raw.get("words", []),
                            "truncated": raw.get("truncated", False),
                        }
                    except json.JSONDecodeError:
                        pass
        raise RuntimeError(f"Fermion CLI transcription failed (code {proc.returncode}): {proc.stderr}")

    raise RuntimeError("No available method to run Phonon-2 one-shot transcription")


def _detect_language(text: str) -> str:
    """Classify the transcribed text into an ISO 639-1 language code."""
    sample = (text or "").strip()
    if len(sample) < 20:
        return "en"
    try:
        import py3langid
        lang, _score = py3langid.classify(sample[:4000])
        return lang
    except Exception:
        return "en"


def adapt_phonon_transcript(raw_result: dict) -> dict:
    """Transform raw Phonon-2 output into GetShorts internal transcript format.

    Contract:
      {
        "text": str,
        "language": str,
        "segments": [
          {
            "start": float, "end": float, "text": str,
            "words": [{"word": str, "start": float, "end": float}, ...]
          }
        ]
      }

    Guarantees:
      - word["word"] has leading space on true word starts.
      - Words are partitioned into their respective segment spans.
      - All timestamps are native Python floats.
      - Segments and words are chronological.
    """
    raw_text = str(raw_result.get("text") or "").strip()
    raw_segments = raw_result.get("segments") or []
    raw_words = raw_result.get("words") or []

    # Map words to segments
    out_segments = []
    text_parts = []

    # Prepare segments structure
    for seg in raw_segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        seg_text = str(seg.get("text") or "").strip()
        out_segments.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg_text,
            "words": [],
        })
        if seg_text:
            text_parts.append(seg_text)

    # Distribute words into segments based on timestamp containment
    if out_segments and raw_words:
        seg_idx = 0
        num_segs = len(out_segments)
        for w in raw_words:
            w_start = float(w["start"])
            w_end = float(w["end"])
            w_text = str(w.get("text") or "").strip()
            if not w_text:
                continue

            word_obj = {
                "word": " " + w_text,
                "start": w_start,
                "end": w_end,
            }

            # Advance seg_idx if word is beyond current segment
            while seg_idx < num_segs - 1 and w_start > out_segments[seg_idx]["end"] + 0.05:
                seg_idx += 1

            # Check containment in current segment or adjacent
            target_seg = out_segments[seg_idx]
            target_seg["words"].append(word_obj)

    # Post-process segments: apply merge_continuation_words and float validations
    cleaned_segments = []
    for s in out_segments:
        words = merge_continuation_words(s["words"])
        # Ensure native floats
        for w in words:
            w["start"] = float(w["start"])
            w["end"] = float(w["end"])
        cleaned_segments.append({
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": s["text"],
            "words": words,
        })

    full_text = raw_text if raw_text else " ".join(text_parts)
    detected_lang = _detect_language(full_text)

    return {
        "text": full_text,
        "language": detected_lang,
        "segments": cleaned_segments,
    }


def phonon_fallback_reason(transcript: dict, duration_hint: float = None) -> str | None:
    """Check if the Phonon-2 transcript is valid or requires fallback."""
    segments = transcript.get("segments") or []
    total_words = sum(len(s.get("words") or []) for s in segments)
    if total_words == 0:
        return "no words recognized"
    duration = duration_hint or (segments[-1]["end"] if segments else 0)
    if duration > 60 and total_words < duration * 0.15:
        return f"only {total_words} words in {duration:.0f}s of audio (word rate too low)"
    return None
