#!/usr/bin/env python3
"""Phonon-2 worker script for GetShorts.

Runs inside the dedicated Phonon virtual environment (/home/zeeshan/models/phonon-env).
Supports two execution modes:
  1. Persistent IPC worker mode (--worker):
     Loads Phonon-2 into memory once, then communicates over stdin/stdout via
     newline-delimited JSON. Avoids model reloading overhead between videos.
  2. One-shot CLI mode (--transcribe <audio_path>):
     Transcribes a single audio file and prints JSON to stdout.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Fix terminal encoding issues if any
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")


def resolve_model_dir(spec: str) -> Path:
    """Resolve the directory holding the unpacked Phonon-2 model weights."""
    path = Path(str(spec)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Phonon model path does not exist: {path}")

    # Case 1: Root repo dir with unpacked subdirectory 'model_phonon2_c4c_int6'
    unpacked_sub = path / "model_phonon2_c4c_int6"
    if unpacked_sub.is_dir() and (unpacked_sub / "model.fermion").is_file():
        return unpacked_sub

    # Case 2: Path itself is the unpacked directory (has model.fermion and valid config)
    if (path / "model.fermion").is_file() and (path / "config.json").is_file():
        try:
            cfg_data = json.loads((path / "config.json").read_text())
            if "joint" in cfg_data or "vocabulary" in cfg_data:
                return path
        except Exception:
            pass

    # Case 3: Need to unpack phonon-2.bps.tar.zst if present
    tar_zst = path / "phonon-2.bps.tar.zst"
    if not tar_zst.is_file() and path.parent.is_dir():
        tar_zst = path.parent / "phonon-2.bps.tar.zst"

    if tar_zst.is_file():
        import subprocess
        print(f"[phonon_worker] Unpacking {tar_zst} to {unpacked_sub}...", file=sys.stderr)
        unpacked_sub.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "--zstd", "-xf", str(tar_zst), "-C", str(unpacked_sub)],
            check=True
        )
        if (unpacked_sub / "model.fermion").is_file():
            return unpacked_sub

    raise FileNotFoundError(
        f"Could not locate model.fermion inside {path} or {unpacked_sub}."
    )


def load_phonon_model(model_dir: Path, threads: int = None):
    """Load the Phonon-2 model using official fermion-research CPU engine."""
    if threads:
        os.environ["FERMION_CPU_THREADS"] = str(int(threads))

    from fermion._speech import backends
    engine_kind = backends.resolve("phonon_worker")
    model_dir_str = str(model_dir)

    speech = backends.load(
        engine_kind,
        model_dir_str,
        profile="five-value",
        backend="phonon2-five-value",
        quiet=True,
    )
    return speech, engine_kind


def transcribe_audio(speech, audio_path: str, repetition_penalty: float = 1.05):
    """Perform transcription and return structured dict with segments & words."""
    p = Path(audio_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {p}")

    t0 = time.perf_counter()
    res = speech.transcribe_detailed(
        str(p),
        repetition_penalty=repetition_penalty,
    )
    text, decode_s, duration_s = res.triple()
    wall_s = time.perf_counter() - t0

    return {
        "success": True,
        "text": text,
        "duration_seconds": round(float(duration_s), 3),
        "decode_seconds": round(float(decode_s), 3),
        "wall_seconds": round(float(wall_s), 3),
        "segment_count": len(res.segments),
        "segments": res.segments,
        "words": res.words if res.timed else [],
        "truncated": bool(res.truncated),
    }


def run_worker_loop(model_dir: Path, threads: int = None):
    """Persistent stdin/stdout JSON protocol worker."""
    # Notify stderr of initialization
    print(f"[phonon_worker] Initializing Phonon-2 from {model_dir}...", file=sys.stderr, flush=True)
    t_start = time.perf_counter()
    speech, engine_kind = load_phonon_model(model_dir, threads=threads)
    init_time = time.perf_counter() - t_start
    print(f"[phonon_worker] Loaded in {init_time:.2f}s ({engine_kind} engine). Ready for requests.", file=sys.stderr, flush=True)

    # Handshake ready line
    ready_msg = json.dumps({
        "status": "ready",
        "model_dir": str(model_dir),
        "engine": engine_kind,
        "init_seconds": round(init_time, 3),
    })
    sys.stdout.write(ready_msg + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            err_resp = {"success": False, "error": f"Invalid JSON request: {exc}"}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()
            continue

        cmd = req.get("cmd", "transcribe")
        if cmd == "ping":
            sys.stdout.write(json.dumps({"status": "pong"}) + "\n")
            sys.stdout.flush()
            continue
        elif cmd == "quit":
            print("[phonon_worker] Received quit command. Exiting.", file=sys.stderr, flush=True)
            break
        elif cmd == "transcribe":
            audio_path = req.get("audio_path")
            rep_pen = float(req.get("repetition_penalty", 1.05))
            try:
                result = transcribe_audio(speech, audio_path, repetition_penalty=rep_pen)
                sys.stdout.write(json.dumps(result) + "\n")
                sys.stdout.flush()
            except Exception as exc:
                err_resp = {
                    "success": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()
        else:
            err_resp = {"success": False, "error": f"Unknown command: {cmd}"}
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Phonon-2 GetShorts Worker")
    parser.add_argument("--model-dir", default=os.environ.get("PHONON_MODEL_PATH", "/home/zeeshan/models/Phonon-2"),
                        help="Path to Phonon-2 model root or unpacked directory")
    parser.add_argument("--threads", type=int, default=None,
                        help="Number of CPU decoding threads")
    parser.add_argument("--worker", action="store_true",
                        help="Run in persistent IPC worker mode (reads requests from stdin)")
    parser.add_argument("--transcribe", type=str, default=None,
                        help="Transcribe a single audio file and exit")
    parser.add_argument("--repetition-penalty", type=float, default=1.05,
                        help="Repetition penalty (default: 1.05)")

    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)

    if args.worker:
        run_worker_loop(model_dir, threads=args.threads)
    elif args.transcribe:
        speech, _ = load_phonon_model(model_dir, threads=args.threads)
        res = transcribe_audio(speech, args.transcribe, repetition_penalty=args.repetition_penalty)
        print(json.dumps(res))
    else:
        parser.print_help(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
