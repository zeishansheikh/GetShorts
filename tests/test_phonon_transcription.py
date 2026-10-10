import os
import json
import pytest
from pathlib import Path

import phonon_backend as pb
import transcribe_backends as tb


# --- 1. Model Resolution & Path Discovery Tests ------------------------------

def test_resolve_phonon_model_dir_default():
    """Verify local Phonon-2 model files exist at configured path."""
    model_dir = pb.resolve_phonon_model_dir("/home/zeeshan/models/Phonon-2")
    assert model_dir.is_dir()
    assert (model_dir / "model.fermion").is_file()
    assert (model_dir / "config.json").is_file()


def test_resolve_phonon_model_dir_nonexistent():
    """Verify FileNotFoundError on nonexistent model path."""
    with pytest.raises(FileNotFoundError):
        pb.resolve_phonon_model_dir("/nonexistent/model/path")


def test_resolve_phonon_model_dir_direct_subdir():
    """Verify passing the unpacked subdirectory directly works."""
    model_dir = pb.resolve_phonon_model_dir("/home/zeeshan/models/Phonon-2/model_phonon2_c4c_int6")
    assert model_dir.is_dir()
    assert model_dir.name == "model_phonon2_c4c_int6"


# --- 2. Transcript Adaptation Contract Tests ---------------------------------

def test_adapt_phonon_transcript_matches_contract():
    """Verify that adapt_phonon_transcript strictly satisfies the GetShorts transcript format."""
    raw_mock = {
        "text": "Hello world. This is Phonon speaking.",
        "duration_seconds": 5.0,
        "decode_seconds": 0.3,
        "segments": [
            {"id": 0, "start": 0.5, "end": 2.0, "text": "Hello world."},
            {"id": 1, "start": 2.2, "end": 4.8, "text": "This is Phonon speaking."}
        ],
        "words": [
            {"text": "Hello", "start": 0.5, "end": 1.0},
            {"text": "world.", "start": 1.1, "end": 1.9},
            {"text": "This", "start": 2.2, "end": 2.6},
            {"text": "is", "start": 2.7, "end": 2.9},
            {"text": "Phonon", "start": 3.0, "end": 3.8},
            {"text": "speaking.", "start": 3.9, "end": 4.7}
        ],
        "truncated": False
    }

    transcript = pb.adapt_phonon_transcript(raw_mock)

    assert transcript["text"] == "Hello world. This is Phonon speaking."
    assert transcript["language"] == "en"
    assert len(transcript["segments"]) == 2

    # Invariant: Segments have start, end, text, words
    seg0 = transcript["segments"][0]
    assert type(seg0["start"]) is float
    assert type(seg0["end"]) is float
    assert seg0["start"] == 0.5
    assert seg0["end"] == 2.0
    assert seg0["text"] == "Hello world."

    # Invariant: Words in segment
    assert len(seg0["words"]) == 2
    w0 = seg0["words"][0]
    # Invariant: leading space on word starts
    assert w0["word"] == " Hello"
    assert type(w0["start"]) is float
    assert type(w0["end"]) is float
    assert w0["start"] == 0.5
    assert w0["end"] == 1.0

    seg1 = transcript["segments"][1]
    assert len(seg1["words"]) == 4
    assert [w["word"] for w in seg1["words"]] == [" This", " is", " Phonon", " speaking."]

    # Must be valid JSON
    json_str = json.dumps(transcript)
    assert len(json_str) > 0


def test_adapt_phonon_transcript_words_survive_merge_continuation():
    """Verify that adapted words conform to merge_continuation_words invariant."""
    from subtitles import merge_continuation_words

    raw_mock = {
        "text": "GetShorts audio pipeline test.",
        "segments": [
            {"id": 0, "start": 0.0, "end": 3.0, "text": "GetShorts audio pipeline test."}
        ],
        "words": [
            {"text": "GetShorts", "start": 0.1, "end": 0.8},
            {"text": "audio", "start": 0.9, "end": 1.4},
            {"text": "pipeline", "start": 1.5, "end": 2.2},
            {"text": "test.", "start": 2.3, "end": 2.8}
        ]
    }

    transcript = pb.adapt_phonon_transcript(raw_mock)
    seg = transcript["segments"][0]
    assert merge_continuation_words(seg["words"]) == seg["words"]


# --- 3. Fallback Reason Policy Tests -----------------------------------------

def test_phonon_fallback_reason_valid():
    transcript = {
        "text": "Valid speech here.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Valid speech here.",
             "words": [{"word": " Valid", "start": 0.0, "end": 0.5},
                       {"word": " speech", "start": 0.6, "end": 1.2},
                       {"word": " here.", "start": 1.3, "end": 1.8}]}
        ]
    }
    assert pb.phonon_fallback_reason(transcript) is None


def test_phonon_fallback_reason_no_words():
    transcript = {
        "text": "",
        "language": "en",
        "segments": [{"start": 0.0, "end": 5.0, "text": "", "words": []}]
    }
    assert "no words recognized" in pb.phonon_fallback_reason(transcript)


def test_phonon_fallback_reason_word_rate_too_low():
    transcript = {
        "text": "word",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 100.0, "text": "word",
             "words": [{"word": " word", "start": 1.0, "end": 1.5}]}
        ]
    }
    assert "word rate too low" in pb.phonon_fallback_reason(transcript, duration_hint=100.0)


# --- 4. Live Audio Transcription Tests ---------------------------------------

def test_live_phonon_transcription_sample():
    """Verify official Phonon-2 execution on a real local audio file."""
    audio_sample = "/tmp/test_10s.wav"
    assert os.path.isfile(audio_sample), f"Test audio {audio_sample} must exist"

    raw_res = pb._WORKER_CLIENT.transcribe(audio_sample)
    assert raw_res.get("success") is True
    assert len(raw_res.get("text", "")) > 10
    assert len(raw_res.get("segments", [])) >= 1
    assert len(raw_res.get("words", [])) >= 5

    transcript = pb.adapt_phonon_transcript(raw_res)
    assert transcript["language"] == "en"
    assert len(transcript["text"]) > 0
    assert len(transcript["segments"]) > 0
    for seg in transcript["segments"]:
        assert seg["start"] >= 0.0
        assert seg["end"] > seg["start"]
        assert len(seg["words"]) > 0
        for w in seg["words"]:
            assert w["word"].startswith(" ")
            assert w["start"] >= 0.0
            assert w["end"] >= w["start"]


def test_transcribe_media_e2e_with_phonon(monkeypatch):
    """Verify transcribe_media end-to-end integration with Phonon-2."""
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "phonon")
    audio_sample = "/tmp/test_10s.wav"

    transcript = tb.transcribe_media(audio_sample)
    assert isinstance(transcript, dict)
    assert "text" in transcript
    assert "language" in transcript
    assert "segments" in transcript
    assert len(transcript["segments"]) > 0

    # Ensure model cleanup runs without error
    tb.release_models()


def test_transcribe_media_falls_back_on_phonon_failure(monkeypatch):
    """Verify transcribe_media gracefully falls back to whisper on error when fallback is enabled."""
    sentinel = {"text": "whisper fallback", "language": "en", "segments": []}
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "phonon")
    monkeypatch.setenv("PHONON_FALLBACK_TO_WHISPER", "1")
    monkeypatch.delenv("PHONON_STRICT", raising=False)
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)

    def boom(path):
        raise RuntimeError("Phonon simulated crash")

    monkeypatch.setattr(tb, "_transcribe_with_phonon", boom)
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: sentinel)

    res = tb.transcribe_media("test_video.mp4")
    assert res is sentinel


def test_transcribe_media_strict_mode_prevents_fallback_on_error(monkeypatch):
    """Verify PHONON_STRICT=1 disables Whisper fallback and raises RuntimeError on error."""
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "phonon")
    monkeypatch.setenv("PHONON_STRICT", "1")
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)

    def boom(path):
        raise RuntimeError("Phonon simulated crash")

    monkeypatch.setattr(tb, "_transcribe_with_phonon", boom)
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: pytest.fail("Whisper should not be called in strict mode"))

    with pytest.raises(RuntimeError, match="Phonon simulated crash"):
        tb.transcribe_media("test_video.mp4")


def test_transcribe_media_strict_mode_prevents_fallback_on_rejected_transcript(monkeypatch):
    """Verify strict mode raises RuntimeError if Phonon transcript is rejected."""
    bad_transcript = {
        "text": "",
        "language": "en",
        "segments": [{"start": 0.0, "end": 5.0, "text": "", "words": []}]
    }
    monkeypatch.setenv("TRANSCRIBE_BACKEND", "phonon")
    monkeypatch.setenv("PHONON_FALLBACK_TO_WHISPER", "0")
    monkeypatch.setattr(tb, "_has_audio_stream", lambda path: True)
    monkeypatch.setattr(tb, "_transcribe_with_phonon", lambda path: bad_transcript)
    monkeypatch.setattr(tb, "_transcribe_with_whisper", lambda path: pytest.fail("Whisper should not be called in strict mode"))

    with pytest.raises(RuntimeError, match="strict mode"):
        tb.transcribe_media("test_video.mp4")

