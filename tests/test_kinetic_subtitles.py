"""Tests for viral-style kinetic subtitles."""
import pytest
from subtitles import (
    generate_ass,
    generate_kinetic_ass,
    _detect_kinetic_punch_word,
    _collect_kinetic_blocks,
    CAPTION_PRESETS,
)


def _w(text, start, end):
    return {"word": text, "start": start, "end": end}


def _transcript(words):
    return {"segments": [{"start": 0, "end": 99, "text": "", "words": words}]}


class TestKineticDetectionAndGrouping:
    def test_single_word_is_punch_word(self):
        words = [_w(" EVERYTHING", 0.0, 0.8)]
        top, punch = _detect_kinetic_punch_word(words)
        assert top == ""
        assert punch == "EVERYTHING"

    def test_phrase_detects_punch_word_at_end(self):
        words = [_w(" learn", 0.0, 0.3), _w(" new", 0.3, 0.6), _w(" skills", 0.6, 1.1)]
        top, punch = _detect_kinetic_punch_word(words)
        assert top == "learn new"
        assert punch == "SKILLS"

    def test_high_impact_words_get_emphasis(self):
        words = [_w(" make", 0.0, 0.3), _w(" more", 0.3, 0.5), _w(" money", 0.5, 0.9)]
        top, punch = _detect_kinetic_punch_word(words)
        assert top == "make more"
        assert punch == "MONEY"

    def test_groups_into_short_phrases_under_four_words(self):
        words = [
            _w(" In", 0.0, 0.2), _w(" this", 0.2, 0.4), _w(" video", 0.4, 0.8),
            _w(" you", 0.9, 1.1), _w(" will", 1.1, 1.3), _w(" learn", 1.3, 1.6), _w(" everything", 1.6, 2.2),
        ]
        blocks = _collect_kinetic_blocks(_transcript(words), 0, 10)
        assert len(blocks) >= 2
        for b in blocks:
            assert len(b["words"]) <= 4
            assert b["punch_word"] != ""


class TestKineticAssGeneration:
    def test_kinetic_layout_and_typography(self, tmp_path):
        out = tmp_path / "kinetic.ass"
        words = [
            _w(" learn", 0.0, 0.3),
            _w(" new", 0.3, 0.6),
            _w(" skills", 0.6, 1.2),
        ]
        ok = generate_kinetic_ass(_transcript(words), 0, 10, str(out), font_name="Montserrat ExtraBold")
        assert ok is True
        content = out.read_text(encoding="utf-8-sig")

        # Resolution 1080x1920
        assert "PlayResX: 1080" in content
        assert "PlayResY: 1920" in content

        # Style specification
        assert "Style: Kinetic,Montserrat ExtraBold,90,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,5,108,108,0,1" in content

        # Dialogue positioning: x=540 (50%), y=1085 (~56.5% height in 1060-1110 range), \an5 (center anchor)
        assert r"\an5\pos(540,1085)" in content

        # Fast scale pop over 80ms
        assert r"\fscx104\fscy104\t(0,80,\fscx100\fscy100)" in content

        # Two-line stack: top line \fs90, bottom punch line \fs210
        assert r"\fs90}learn new\N{\fs210}SKILLS" in content

    def test_single_word_caption_rendering(self, tmp_path):
        out = tmp_path / "kinetic_single.ass"
        words = [_w(" EVERYTHING", 0.0, 0.9)]
        ok = generate_kinetic_ass(_transcript(words), 0, 10, str(out))
        assert ok is True
        content = out.read_text(encoding="utf-8-sig")

        # Single word has no \N line break and renders with punch font size
        assert r"\pos(540,1085)" in content
        assert r"\N" not in content
        assert "EVERYTHING" in content

    def test_generate_ass_routes_kinetic_preset(self, tmp_path):
        out = tmp_path / "route_kinetic.ass"
        words = [_w(" build", 0.0, 0.4), _w(" skills", 0.4, 0.9)]
        ok = generate_ass(_transcript(words), 0, 10, str(out), style="kinetic")
        assert ok is True
        content = out.read_text(encoding="utf-8-sig")
        assert "PlayResX: 1080" in content
        assert r"\an5\pos(540,1085)" in content
        assert "SKILLS" in content

    def test_long_keyword_auto_downscales(self, tmp_path):
        out = tmp_path / "long_keyword.ass"
        words = [_w(" total", 0.0, 0.3), _w(" transformation", 0.3, 1.1)]
        ok = generate_kinetic_ass(_transcript(words), 0, 10, str(out))
        assert ok is True
        content = out.read_text(encoding="utf-8-sig")

        # "TRANSFORMATION" (14 chars) would exceed 80% width at 210px; font size should be clamped
        # 860 / (14 * 0.58) = 105 -> \fs105
        assert r"\fs105}TRANSFORMATION" in content
