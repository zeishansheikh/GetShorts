import os
import re
import subprocess
import sys

from ffmpeg_utils import (video_encode_args, escape_filter_value, QUALITY,
                          METADATA_SCRUB)


_STDIO_CONFIGURED = False

# Shared faster-whisper config so both transcription paths (this module and
# main.transcribe_video) behave identically. "small" is meaningfully better at
# German than "base" without being much slower on CPU.
DEFAULT_WHISPER_MODEL = "small"


def get_whisper_config():
    """Return the faster-whisper model config, overridable via env vars."""
    return {
        "model_size": os.environ.get("WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
        "device": os.environ.get("WHISPER_DEVICE", "cpu"),
        "compute_type": os.environ.get("WHISPER_COMPUTE", "int8"),
    }


# Decode params shared by both transcription paths. condition_on_previous_text
# is off to avoid repetition/hallucination loops; vad_filter drops silence.
WHISPER_TRANSCRIBE_PARAMS = {
    "beam_size": 5,
    "vad_filter": True,
    "condition_on_previous_text": False,
    "word_timestamps": True,
}


def merge_continuation_words(words):
    """Merge faster-whisper continuation fragments into their base word.

    faster-whisper marks a word boundary with a LEADING SPACE on each token.
    Compound-word fragments (e.g. "-Kanal.", ".200") arrive WITHOUT a leading
    space and belong to the preceding word. Without merging, "YouTube" and
    "-Kanal." get space-joined into "YouTube -Kanal." or split across subtitle
    blocks. We concatenate such fragments onto the previous word and extend its
    end time. Normal words keep their leading space, so real word boundaries
    (e.g. "ich habe") are never glued together.

    Returns a new list; the input dicts are not mutated.
    """
    merged = []
    for word in words:
        text = word.get("word", "")
        if merged and isinstance(text, str) and text and not text.startswith(" "):
            prev = merged[-1]
            prev["word"] = f"{prev.get('word', '')}{text}"
            if word.get("end") is not None:
                prev["end"] = word["end"]
        else:
            merged.append(dict(word))
    return merged


def _configure_stdio():
    global _STDIO_CONFIGURED
    if _STDIO_CONFIGURED:
        return
    _STDIO_CONFIGURED = True
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if not stream or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _log(message):
    _configure_stdio()
    stream = sys.stdout
    text = str(message)
    try:
        stream.write(text + "\n")
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        stream.write(safe_text + "\n")
    stream.flush()


def _escape_ffmpeg_filter_value(value):
    """Escape a path/value for use inside a quoted FFmpeg filter argument.

    NOTE: an apostrophe in the path cannot be made safe here. ffmpeg's
    filtergraph parser is not a shell — the shell idiom ``'\\''`` was tried on
    29-jul-2026 and is worse than doing nothing: it drops the apostrophe AND
    swallows the following option, so ``ass='…Earth'\\''s.ass':fontsdir='…'``
    resolved to a filename of "…Earths.ass:fontsdir=…" and failed to open.

    The only reliable answer is to keep apostrophes OUT of any path that is
    interpolated into a filter. Callers generate their own subtitle filenames,
    so they control this: use a neutral name (``subs_<i>_<ts>.ass``), never one
    derived from a video title.

    The implementation now lives in ffmpeg_utils so the reframe engine can use
    it too: it was building `sendcmd=f='<abs path>'` unescaped, which is the
    same bug this function was written for.
    """
    return escape_filter_value(value)


def _normalize_subtitle_word(value):
    return " ".join(str(value or "").split())


def transcribe_audio(video_path):
    """
    Transcribe audio from a video file via the configured ASR backend.
    Returns transcript in the same format as main.py for compatibility.
    """
    # Lazy import: transcribe_backends imports helpers from this module.
    from transcribe_backends import transcribe_media

    _log(f"🎙️  Transcribing audio from: {video_path}")
    transcript = transcribe_media(video_path)
    _log(f"✅ Transcription complete. Language: {transcript['language']}")
    return transcript


def generate_srt_from_video(video_path, output_path, max_chars=20, max_duration=2.0,
                            style="classic", **style_opts):
    """
    Transcribe a video and generate a subtitle file directly (SRT, or karaoke
    ASS when style="karaoke"). Used for dubbed videos without a transcript.
    """
    transcript = transcribe_audio(video_path)

    # Get video duration to use as clip_end
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0
    cap.release()

    if style == "karaoke":
        return generate_ass(transcript, 0, duration, output_path, max_chars, max_duration, **style_opts)
    return generate_srt(transcript, 0, duration, output_path, max_chars, max_duration)


def _collect_word_blocks(transcript, clip_start, clip_end, max_chars=20, max_duration=2.0):
    """
    Flatten transcript words for a clip range and group them into short blocks
    suitable for vertical video. Returns a list of blocks; each block is a list
    of {'word', 'start', 'end'} dicts with times relative to the clip.

    Continuation fragments are merged defensively here too, because transcripts
    from old jobs on disk store unmerged tokens (the leading space is still
    present, so the boundary signal survives).
    """
    flat_words = []
    for segment in transcript.get('segments', []):
        flat_words.extend(segment.get('words', []))
    flat_words = merge_continuation_words(flat_words)

    words = []
    for word_info in flat_words:
        if word_info.get('end', 0) > clip_start and word_info.get('start', 0) < clip_end:
            cleaned_word = _normalize_subtitle_word(word_info.get('word', ''))
            if not cleaned_word:
                continue
            words.append({
                'word': cleaned_word,
                'start': max(0, word_info['start'] - clip_start),
                'end': max(0, word_info['end'] - clip_start),
            })

    blocks = []
    current_block = []
    block_start = None

    for word in words:
        if not current_block:
            current_block = [word]
            block_start = word['start']
            continue

        current_text_len = sum(len(w['word']) + 1 for w in current_block)
        duration = word['end'] - block_start

        if current_text_len + len(word['word']) > max_chars or duration > max_duration:
            blocks.append(current_block)
            current_block = [word]
            block_start = word['start']
        else:
            current_block.append(word)

    if current_block:
        blocks.append(current_block)
    return blocks


def generate_srt(transcript, clip_start, clip_end, output_path, max_chars=20, max_duration=2.0):
    """
    Generates an SRT file from the transcript for a specific time range.
    Groups words into short lines suitable for vertical video.
    """
    blocks = _collect_word_blocks(transcript, clip_start, clip_end, max_chars, max_duration)
    if not blocks:
        return False

    srt_content = ""
    for index, block in enumerate(blocks, 1):
        text = " ".join(w['word'] for w in block).strip()
        srt_content += format_srt_block(index, block[0]['start'], block[-1]['end'], text)

    # Write UTF-8 with BOM so Windows/FFmpeg subtitle readers reliably detect Unicode text.
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(srt_content)

    return True


# Vertical margin for burned captions, in PlayResY=288 units (so ~15% of the
# frame height). The old hardcoded 25 (8.7%) put captions underneath TikTok's
# and Reels' own bottom UI — the caption/username block and the music ticker —
# where they were partly covered on the platform even though the exported file
# looked fine.
SAFE_MARGIN_V = 43


# The caption look applied automatically to every generated clip. Chosen by
# rendering four candidates on a real clip and comparing them (25-jul-2026):
# white Anton uppercase with a yellow active word, heavy black outline, gentle
# pop. Yellow because it is the one colour that almost never occurs in footage,
# so the active word reads instantly on any background; the base text stays
# fully opaque (dimming it tested worse over bright scenes). This is a starting
# point, not a cage — the subtitle modal still overrides every field.
AUTO_CAPTION_STYLE = {
    "style": "karaoke",
    "alignment": "bottom",
    "font_name": "Barlow-ExtraLight",
    "font_size": 8,
    "font_color": "#FFFFFF",
    "highlight_color": "#FFE500",
    "border_color": "#000000",
    "border_width": 1,
    "effect": "pop",
    "base_opacity": 1.0,
    "uppercase": True,
    "max_chars": 16,
    "max_duration": 1.4,
}


# Named looks for /api/subtitle `preset` (and the MCP add_subtitles tool), in
# request-field names. Mirrors the dashboard's CAPTION_PRESETS in
# SubtitleModal.jsx (a test checks the ids): "default" is what every clip
# ships with, the rest are the short-form looks trending in 2026.
_PRESET_BASE = {"style": "karaoke", "font_color": "#FFFFFF", "bg_opacity": 0.0,
                "base_opacity": 1.0, "reveal": False, "shadow": 0,
                "max_duration": 1.4}
CAPTION_PRESETS = {
    # Viral kinetic subtitles (phrase-by-phrase, two-line punch keyword stack, pure white)
    "kinetic": {
        "style": "kinetic", "font_name": "Montserrat ExtraBold", "font_size": 8,
        "font_color": "#FFFFFF", "border_width": 0, "shadow": 0, "effect": "kinetic",
        "uppercase": True, "max_chars": 12, "max_duration": 1.4, "bg_opacity": 0.0,
        "base_opacity": 1.0, "reveal": False,
    },
    "default": {**_PRESET_BASE, "font_name": "Barlow-ExtraLight", "font_size": 8,
                "highlight_color": "#FFE500", "border_width": 1, "effect": "pop",
                "uppercase": True, "max_chars": 16},
    # Words appear as they are spoken, yellow active word, shadow.
    "hormozi": {**_PRESET_BASE, "font_name": "Montserrat ExtraBold", "font_size": 8,
                "highlight_color": "#FFE500", "border_width": 1, "shadow": 1,
                "effect": "pop", "uppercase": True, "reveal": True, "max_chars": 12},
    # Solid box behind the active word (CapCut / Submagic).
    "pill": {**_PRESET_BASE, "font_name": "Montserrat ExtraBold", "font_size": 8,
             "highlight_color": "#7C3AED", "border_width": 1, "effect": "highlight",
             "uppercase": True, "max_chars": 12},
    "lime": {**_PRESET_BASE, "font_name": "Montserrat ExtraBold", "font_size": 8,
             "highlight_color": "#A3FF12", "border_width": 1, "effect": "highlight",
             "uppercase": True, "max_chars": 12},
    # One big word at a time.
    "oneword": {**_PRESET_BASE, "font_name": "Barlow-ExtraLight", "font_size": 10,
                "highlight_color": "#FFFFFF", "border_width": 2, "effect": "pop",
                "uppercase": True, "max_chars": 1},
    # No outline, soft shadow, sentence case.
    "clean": {**_PRESET_BASE, "font_name": "Montserrat ExtraBold", "font_size": 7,
              "highlight_color": "#FFFFFF", "border_width": 0, "shadow": 1,
              "effect": "none", "uppercase": False, "base_opacity": 0.7,
              "max_chars": 14},
}

# Characters per line at font size 44, per font.
# Mirrors lineBudget in SubtitleModal.jsx.
_LINE_CHARS = {
    "Barlow-ExtraLight": 14,
    "Barlow ExtraLight": 14,
    "Anton": 16,
    "Montserrat ExtraBold": 9,
    "Impact": 16,
}


def line_budget(font_name, font_size):
    """max_chars that keeps one line inside the 9:16 frame at this size."""
    size = _clamp_number(font_size, 4, 200, 8)
    if size <= 12:
        return max(6, min(24, round(_LINE_CHARS.get(font_name, 14) * 8 / size)))
    return max(6, round(_LINE_CHARS.get(font_name, 14) * 44 / size))


def _ass_time(seconds):
    """Format seconds as ASS timestamp H:MM:SS.cc (centiseconds)."""
    seconds = max(0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _hex_to_ass_inline_color(hex_color, fallback="FFFFFF"):
    """Convert #RRGGBB to the &HBBGGRR& form used by inline \\c override tags."""
    hex_digits = str(hex_color or "").lstrip('#')
    if not _HEX_COLOR_RE.match(hex_digits):
        hex_digits = fallback
    r = hex_digits[0:2]
    g = hex_digits[2:4]
    b = hex_digits[4:6]
    return f"&H{b}{g}{r}&".upper()


def _escape_ass_text(text):
    """Neutralize characters that would start ASS override blocks."""
    return str(text).replace('\\', '/').replace('{', '(').replace('}', ')')


def _dim_hex_color(hex_color, opacity, fallback="FFFFFF"):
    """Fully-opaque 'dimmed' variant of a color (scaled toward black).

    Dimming via alpha looks muddy in ASS: libass draws the outline as a
    filled shape UNDER the fill, so a semi-transparent white fill blends
    with its own black outline into dark grey. Scaling the RGB instead
    keeps the text crisp on every player."""
    hex_digits = str(hex_color or "").lstrip('#')
    if not _HEX_COLOR_RE.match(hex_digits):
        hex_digits = fallback
    # Gentle curve: even strong dimming stays a readable light silver, matching
    # the airy look of browser-alpha dimming over bright video.
    factor = 0.5 + 0.5 * _clamp_number(opacity, 0.05, 1.0, 1.0)
    r = min(255, round(int(hex_digits[0:2], 16) * factor))
    g = min(255, round(int(hex_digits[2:4], 16) * factor))
    b = min(255, round(int(hex_digits[4:6], 16) * factor))
    return f"{r:02X}{g:02X}{b:02X}"


_KINETIC_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

_KINETIC_HIGH_IMPACT_WORDS = {
    "everything", "nothing", "never", "always", "skills", "money", "million", "billion",
    "dollar", "dollars", "secret", "truth", "viral", "huge", "insane", "crazy", "fast",
    "power", "powerful", "success", "life", "future", "world", "stop", "danger", "win",
    "winner", "lose", "free", "dead", "kill", "die", "best", "worst", "mistake", "fix",
    "rule", "rules", "game", "change", "changed", "business", "growth", "watch", "now",
    "everybody", "everyone", "nobody", "forever", "impossible", "simple", "easy", "hard"
}


def _detect_kinetic_punch_word(words):
    """
    Given a list of word dicts [{'word': ..., 'start': ..., 'end': ...}],
    returns (top_text, punch_word).
    """
    if not words:
        return "", ""

    def clean_text(t):
        return re.sub(r'[^a-z0-9]', '', str(t or '').lower())

    if len(words) == 1:
        raw = re.sub(r'^[^a-zA-Z0-9$!%?]+|[^a-zA-Z0-9$!%?]+$', '', words[0]['word'].strip())
        return "", (raw or words[0]['word'].strip()).upper()

    best_score = -9999
    best_idx = len(words) - 1

    for idx, w in enumerate(words):
        clean = clean_text(w['word'])
        if not clean:
            continue
        score = len(clean) * 1.5
        if clean in _KINETIC_HIGH_IMPACT_WORDS:
            score += 15
        if re.search(r'\d', clean):
            score += 12
        if w['word'].strip().isupper() and len(clean) > 1:
            score += 8
        if clean in _KINETIC_STOP_WORDS:
            score -= 10
        score += idx * 2.0  # End-focus bias

        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx == len(words) - 1:
        top_text = " ".join(w['word'].strip() for w in words[:best_idx])
        punch_raw = re.sub(r'^[^a-zA-Z0-9$!%?]+|[^a-zA-Z0-9$!%?]+$', '', words[best_idx]['word'].strip())
        return top_text, (punch_raw or words[best_idx]['word'].strip()).upper()

    last_clean = clean_text(words[-1]['word'])
    if last_clean not in _KINETIC_STOP_WORDS and len(last_clean) >= 3:
        top_text = " ".join(w['word'].strip() for w in words[:-1])
        punch_raw = re.sub(r'^[^a-zA-Z0-9$!%?]+|[^a-zA-Z0-9$!%?]+$', '', words[-1]['word'].strip())
        return top_text, (punch_raw or words[-1]['word'].strip()).upper()

    top_text = " ".join(w['word'].strip() for w in words[:best_idx])
    punch_words = " ".join(w['word'].strip() for w in words[best_idx:])
    return top_text, punch_words.upper()


def _collect_kinetic_blocks(transcript, clip_start, clip_end, min_duration=0.7, max_duration=1.5):
    """Group words into kinetic phrases (1-4 words, ~0.8-1.5s)."""
    flat_words = []
    for segment in transcript.get('segments', []):
        flat_words.extend(segment.get('words', []))
    flat_words = merge_continuation_words(flat_words)

    words = []
    for word_info in flat_words:
        if word_info.get('end', 0) > clip_start and word_info.get('start', 0) < clip_end:
            cleaned = _normalize_subtitle_word(word_info.get('word', ''))
            if not cleaned:
                continue
            words.append({
                'word': cleaned,
                'start': max(0, word_info['start'] - clip_start),
                'end': max(0, word_info['end'] - clip_start),
            })

    if not words:
        return []

    blocks = []
    current_words = []
    block_start = 0.0

    def clean_text(t):
        return re.sub(r'[^a-z0-9]', '', str(t or '').lower())

    for word in words:
        if not current_words:
            current_words = [word]
            block_start = word['start']
            continue

        duration = word['end'] - block_start
        prev_word = current_words[-1]
        speech_pause = word['start'] - prev_word['end']
        ends_with_punct = bool(re.search(r'[.!?]$', prev_word['word'].strip()))
        is_single_emphatic = (len(current_words) == 1 and
                              clean_text(prev_word['word']) in _KINETIC_HIGH_IMPACT_WORDS and
                              duration >= 0.6)

        should_close = (len(current_words) >= 4 or
                        duration >= max_duration or
                        speech_pause > 0.3 or
                        ends_with_punct or
                        is_single_emphatic or
                        (len(current_words) >= 2 and duration >= min_duration and
                         re.match(r'^(to|in|on|at|for|with|and|but|or|because|if|so|that)$', word['word'].strip(), re.I)))

        if should_close:
            top_text, punch_word = _detect_kinetic_punch_word(current_words)
            blocks.append({
                'words': list(current_words),
                'start': block_start,
                'end': current_words[-1]['end'],
                'top_text': top_text,
                'punch_word': punch_word,
            })
            current_words = [word]
            block_start = word['start']
        else:
            current_words.append(word)

    if current_words:
        top_text, punch_word = _detect_kinetic_punch_word(current_words)
        blocks.append({
            'words': list(current_words),
            'start': block_start,
            'end': current_words[-1]['end'],
            'top_text': top_text,
            'punch_word': punch_word,
        })

    # Extend gap under 150ms to next block to prevent flicker
    for i in range(len(blocks) - 1):
        next_start = blocks[i + 1]['start']
        if next_start > blocks[i]['end'] and (next_start - blocks[i]['end']) < 0.15:
            blocks[i]['end'] = next_start

    return blocks


def generate_kinetic_ass(transcript, clip_start, clip_end, output_path,
                         font_name="Montserrat ExtraBold", **kwargs):
    """
    Generates viral-style kinetic subtitles for vertical video (9:16, 1080x1920):
    - Horizontally centered (x = 540 = 50% width).
    - Vertically anchored at 56.5% height (y = 1085 in 1060-1110 range).
    - Pure white (#FFFFFF), no stroke, no background box, no highlight color.
    - Normal words ~4.7% height (90px), emphasis keyword ~11% height (210px).
    - Two-line stack: small words on top, BIG keyword below.
    - Fast scale pop: 100% to 104% over ~80ms.
    """
    blocks = _collect_kinetic_blocks(transcript, clip_start, clip_end)
    if not blocks:
        return False

    safe_font = _sanitize_font_name(font_name or "Montserrat ExtraBold")

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Kinetic,{safe_font},90,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,5,108,108,0,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events = []
    for b in blocks:
        ev_start = b['start']
        ev_end = b['end']
        if ev_end <= ev_start:
            continue

        punch_word = _escape_ass_text(b['punch_word'])
        top_text = _escape_ass_text(b['top_text'])

        # Dynamic font size clamp if keyword is long so it fits within 80% line width
        kw_len = len(punch_word)
        if kw_len > 7:
            punch_fs = max(90, min(210, int(860 / (kw_len * 0.58))))
        else:
            punch_fs = 210

        # Fast scale pop over 80ms: \fscx104\fscy104\t(0,80,\fscx100\fscy100)
        # Position anchored at vertical center ~56.5% height (y=1085, x=540)
        pop_prefix = r"{\an5\pos(540,1085)\fscx104\fscy104\t(0,80,\fscx100\fscy100)"

        if top_text:
            text = f"{pop_prefix}\\fs90}}{top_text}\\N{{\\fs{punch_fs}}}{punch_word}"
        else:
            text = f"{pop_prefix}\\fs{punch_fs}}}{punch_word}"

        events.append(
            f"Dialogue: 0,{_ass_time(ev_start)},{_ass_time(ev_end)},Kinetic,,0,0,0,,{text}"
        )

    if not events:
        return False

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(header + "\n".join(events) + "\n")

    return True


def generate_ass(transcript, clip_start, clip_end, output_path,
                 max_chars=16, max_duration=1.4, alignment='bottom',
                 fontsize=8, font_name="Barlow-ExtraLight", font_color="#FFFFFF",
                 border_color="#000000", border_width=1,
                 highlight_color="#FFE500", bg_color="#000000", bg_opacity=0.0,
                 effect="pop", base_opacity=1.0, uppercase=True,
                 margin_v=SAFE_MARGIN_V, split_ranges=None,
                 reveal=False, shadow=0, style="karaoke", **kwargs):
    """
    Generates a karaoke-style or kinetic ASS file:
    When style="kinetic" or effect="kinetic", delegates to generate_kinetic_ass.
    Otherwise generates modern karaoke ASS captions with per-word highlight.

    effect: "none" | "glow" (neon shine around the active word) |
            "pop" (active word scales up) | "box" (thick colored outline) |
            "highlight" (active word on a solid box in highlight_color, the
            CapCut / Submagic look).
    base_opacity: opacity of the non-active words (dimmed base text, e.g. 0.4).
    reveal: words not spoken yet are invisible, so the line builds up word
    by word (Hormozi style).
    shadow: drop shadow depth in PlayRes units (0 = none).
    max_chars=1 puts one word on screen at a time.
    """
    if style == "kinetic" or effect == "kinetic":
        return generate_kinetic_ass(
            transcript, clip_start, clip_end, output_path,
            font_name=font_name or "Montserrat ExtraBold", **kwargs
        )

    blocks = _collect_word_blocks(transcript, clip_start, clip_end, max_chars, max_duration)
    if not blocks:
        return False

    # Match the SRT burn path: PlayResY 288 keeps font sizes consistent.
    # For sizes <= 12 (e.g. 6, 7, 8), keep exact integer font size so text never cuts off.
    if fontsize <= 12:
        final_fontsize = int(round(_clamp_number(fontsize, 4, 12, 8)))
    else:
        final_fontsize = int(_clamp_number(fontsize, 4, 200, 8) * 0.85)
    if final_fontsize < 4:
        final_fontsize = 4

    align_map = {'top': 8, 'middle': 5, 'bottom': 2}
    ass_alignment = align_map.get(str(alignment).lower(), 2)

    # On a SPLIT scene the two speakers are stacked and the seam between the
    # halves (exactly mid-frame) is the one place the text covers nobody, so
    # every word event inside such a stretch is anchored there with an inline
    # \an5, per event rather than per style: a clip mixes stacked and single
    # shots, and the text moves with the cut. ``split_ranges`` is a list of
    # (start, end) in clip seconds (layout_ranges.split_ranges); the style's
    # own alignment still rules everywhere else. Only the ASS path can do
    # this: SRT burns carry one alignment for the whole file.
    seam_ranges = [(float(a), float(b)) for a, b in (split_ranges or [])]

    def seam_prefix(t):
        return "{\\an5}" if any(a <= t < b for a, b in seam_ranges) else ""

    safe_font = _sanitize_font_name(font_name)
    base_opacity = _clamp_number(base_opacity, 0.05, 1.0, 1.0)
    # Dim inactive words via a fully-opaque scaled color (NOT alpha — see
    # _dim_hex_color); the active word overrides the color inline.
    primary_colour = hex_to_ass_color(_dim_hex_color(font_color, base_opacity), 1.0)
    bg_opacity = _clamp_number(bg_opacity, 0.0, 1.0, 0.0)
    border_width = _clamp_number(border_width, 0, 10, 2)
    shadow = int(_clamp_number(shadow, 0, 6, 0))

    if bg_opacity > 0:
        border_style = 3
        outline_colour = hex_to_ass_color(bg_color, bg_opacity, fallback="000000")
        outline_width = 1
    else:
        border_style = 1
        outline_colour = hex_to_ass_color(border_color, 1.0, fallback="000000")
        # 0 is a real choice now (the shadow-only "clean" look); before the
        # slider's "None" still drew a 1px outline. Without a shadow keep the
        # old floor, or white text on a white wall disappears.
        outline_width = int(border_width) if shadow else max(1, int(border_width))

    back_colour = hex_to_ass_color("#000000", 0.55 if shadow else 0.0)
    highlight_inline = _hex_to_ass_inline_color(highlight_color, fallback="FFD700")

    # Inline override tags for the active word; {\r} after it resets to the
    # (dimmed) style so the rest of the block stays untouched.
    if effect == "glow":
        glow_bord = max(3, int(outline_width) + 2)
        active_prefix = (f"{{\\c&HFFFFFF&\\3c{highlight_inline}"
                         f"\\bord{glow_bord}\\blur4}}")
    elif effect == "box":
        box_bord = max(4, int(outline_width) + 3)
        active_prefix = (f"{{\\c&HFFFFFF&\\3c{highlight_inline}"
                         f"\\bord{box_bord}\\blur0}}")
    elif effect == "highlight":
        # A second style with BorderStyle 3 draws an opaque box around just
        # the active word (libass boxes each style run separately). The text
        # on it flips to black when the box is light, or yellow would carry
        # white text nobody can read.
        active_prefix = "{\\rActive}"
    elif effect == "pop":
        # Gentle pop. The old 75->112 range started the word so small that any
        # frame caught mid-animation read as a sizing bug rather than a beat.
        active_prefix = (f"{{\\c{highlight_inline}"
                         f"\\fscx90\\fscy90\\t(0,110,\\fscx108\\fscy108)}}")
    else:
        active_prefix = f"{{\\c{highlight_inline}}}"

    active_style = ""
    if effect == "highlight":
        box_colour = hex_to_ass_color(highlight_color, 1.0, fallback="FFD700")
        on_box = "#000000" if _luminance(highlight_color) > 0.6 else font_color
        # Padding scales with the text so the box keeps its shape at any size.
        pad = max(2, round(final_fontsize * 0.12))
        active_style = (
            f"Style: Active,{safe_font},{final_fontsize},"
            f"{hex_to_ass_color(on_box, 1.0)},{hex_to_ass_color(on_box, 1.0)},"
            f"{box_colour},{box_colour},1,0,0,0,100,100,0,0,3,{pad},0,"
            f"{ass_alignment},10,10,{int(_clamp_number(margin_v, 0, 200, SAFE_MARGIN_V))},1\n"
        )

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResY: 288\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{safe_font},{final_fontsize},{primary_colour},{primary_colour},"
        f"{outline_colour},{back_colour},1,0,0,0,100,100,0,0,{border_style},"
        f"{outline_width},{shadow},{ass_alignment},10,10,{int(_clamp_number(margin_v, 0, 200, SAFE_MARGIN_V))},1\n"
        f"{active_style}"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events = []
    for block in blocks:
        for i, word in enumerate(block):
            # Event runs until the next word starts (no flicker in gaps);
            # the last word holds until the block ends.
            ev_start = block[0]['start'] if i == 0 else word['start']
            ev_end = block[i + 1]['start'] if i < len(block) - 1 else block[-1]['end']
            if ev_end <= ev_start:
                continue

            parts = []
            for j, other in enumerate(block):
                text = _escape_ass_text(other['word'])
                if uppercase:
                    text = text.upper()
                if j == i:
                    parts.append(f"{active_prefix}{text}{{\\r}}")
                elif reveal and j > i:
                    parts.append(f"{{\\alpha&HFF&}}{text}{{\\r}}")
                else:
                    parts.append(text)

            events.append(
                f"Dialogue: 0,{_ass_time(ev_start)},{_ass_time(ev_end)},Default,,0,0,0,,"
                f"{seam_prefix(ev_start)}{' '.join(parts)}"
            )

    if not events:
        return False

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(header + "\n".join(events) + "\n")

    return True

def format_srt_block(index, start, end, text):
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
    return f"{index}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"

_HEX_COLOR_RE = re.compile(r'^[0-9A-Fa-f]{6}$')
_FONT_NAME_RE = re.compile(r'[^A-Za-z0-9 _-]')


def hex_to_ass_color(hex_color, opacity=1.0, fallback="FFFFFF"):
    """Convert #RRGGBB to ASS &HAABBGGRR format. opacity: 0.0=transparent, 1.0=opaque.

    Invalid hex (e.g. "#GGGGGG", None, wrong length) falls back to `fallback`
    instead of raising, so a bad color from the client can't 500 the request.
    """
    hex_digits = str(hex_color or "").lstrip('#')
    if not _HEX_COLOR_RE.match(hex_digits):
        hex_digits = fallback
    opacity = _clamp_number(opacity, 0.0, 1.0, 1.0)
    r = int(hex_digits[0:2], 16)
    g = int(hex_digits[2:4], 16)
    b = int(hex_digits[4:6], 16)
    alpha = round((1.0 - opacity) * 255)
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _luminance(hex_color):
    """Relative brightness 0-1 of #RRGGBB (Rec. 601 weights); 1.0 if invalid."""
    hex_digits = str(hex_color or "").lstrip('#')
    if not _HEX_COLOR_RE.match(hex_digits):
        return 1.0
    r, g, b = (int(hex_digits[i:i + 2], 16) for i in (0, 2, 4))
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def _clamp_number(value, lo, hi, default):
    """Coerce value to float and clamp to [lo, hi]; use default if not numeric."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = float(default)
    return max(lo, min(hi, num))


_FONT_ALIASES = {
    "barlow-extralight": "Barlow ExtraLight",
    "barlow_extralight": "Barlow ExtraLight",
    "barlow extralight": "Barlow ExtraLight",
    "barlow-extrabold": "Barlow ExtraBold",
    "barlow_extrabold": "Barlow ExtraBold",
    "barlow extrabold": "Barlow ExtraBold",
    "montserrat-extrabold": "Montserrat ExtraBold",
    "montserrat_extrabold": "Montserrat ExtraBold",
    "montserrat extrabold": "Montserrat ExtraBold",
    "inter-extrabold": "Inter ExtraBold",
    "inter_extrabold": "Inter ExtraBold",
    "inter extrabold": "Inter ExtraBold",
}


def _sanitize_font_name(name):
    """Strip anything but [A-Za-z0-9 _-] so the font name can't inject extra
    ASS override fields (commas/braces/backslashes) into force_style."""
    cleaned = _FONT_NAME_RE.sub('', str(name or '')).strip()
    if not cleaned:
        return "Verdana"
    return _FONT_ALIASES.get(cleaned.lower(), cleaned)


def subtitles_filter(srt_path, alignment=2, fontsize=8,
                     font_name="Barlow-ExtraLight", font_color="#FFFFFF",
                     border_color="#000000", border_width=1,
                     bg_color="#000000", bg_opacity=0.0):
    """The -vf string burn_subtitles uses (also fed to hooks.add_hook_to_video)."""
    # Position mapping
    ass_alignment = 2
    align_lower = str(alignment).lower()
    if align_lower == 'top':
        ass_alignment = 6
    elif align_lower == 'middle':
        ass_alignment = 10
    elif align_lower == 'bottom':
        ass_alignment = 2

    # Font size scaling for ASS virtual resolution (PlayResY=288 default)
    # For small sizes (<=12, e.g. 6, 7, 8), keep exact font size to stay properly inside screen.
    if fontsize <= 12:
        final_fontsize = int(round(_clamp_number(fontsize, 4, 12, 8)))
    else:
        final_fontsize = int(_clamp_number(fontsize, 4, 200, 8) * 0.85)
    if final_fontsize < 4:
        final_fontsize = 4

    safe_font_name = _sanitize_font_name(font_name)
    bg_opacity = _clamp_number(bg_opacity, 0.0, 1.0, 0.0)
    border_width = _clamp_number(border_width, 0, 10, 1)

    # Path handling for FFmpeg filter syntax
    safe_srt_path = _escape_ffmpeg_filter_value(srt_path)

    # Convert colors to ASS format and build style
    primary_colour = hex_to_ass_color(font_color, 1.0)

    if bg_opacity > 0:
        # Box mode: opaque background box
        border_style = 3
        outline_colour = hex_to_ass_color(bg_color, bg_opacity, fallback="000000")
        outline_width = 1
    else:
        # Outline mode: text border/outline
        border_style = 1
        outline_colour = hex_to_ass_color(border_color, 1.0, fallback="000000")
        outline_width = max(1, int(border_width))

    back_colour = hex_to_ass_color("#000000", 0.0)

    style_string = (
        f"Alignment={ass_alignment},"
        f"Fontname={safe_font_name},"
        f"Fontsize={final_fontsize},"
        f"PrimaryColour={primary_colour},"
        f"OutlineColour={outline_colour},"
        f"BackColour={back_colour},"
        f"BorderStyle={border_style},"
        f"Outline={outline_width},"
        f"Shadow=0,"
        f"MarginL=10,"
        f"MarginR=10,"
        f"MarginV={SAFE_MARGIN_V},"
        f"Bold=1"
    )

    # Let libass see the fonts bundled with the app (e.g. Anton for Impact)
    # even when the system fontconfig has no cache for them.
    fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    safe_fonts_dir = _escape_ffmpeg_filter_value(fonts_dir)

    # The first option is named explicitly (filename=) rather than positional:
    # ffmpeg 8's filtergraph parser rejects a quoted positional value that is
    # followed by more :name=value options ("No option name near ..."), while
    # the named form parses on every version back to 4.x.
    if str(srt_path).lower().endswith('.ass'):
        # ASS files (karaoke style) carry their own styles; force_style would
        # override the per-word color tags.
        vf = f"ass=filename='{safe_srt_path}':fontsdir='{safe_fonts_dir}'"
    else:
        vf = (f"subtitles=filename='{safe_srt_path}':fontsdir='{safe_fonts_dir}'"
              f":charenc=UTF-8:force_style='{style_string}'")

    return vf


def burn_subtitles(video_path, srt_path, output_path, alignment=2, fontsize=8,
                   font_name="Barlow-ExtraLight", font_color="#FFFFFF",
                   border_color="#000000", border_width=1,
                   bg_color="#000000", bg_opacity=0.0):
    """
    Burns subtitles into the video using FFmpeg.
    Supports two modes:
    - Outline mode (bg_opacity=0): Text with colored outline/border
    - Box mode (bg_opacity>0): Text with semi-transparent background box
    """
    vf = subtitles_filter(srt_path, alignment=alignment, fontsize=fontsize,
                          font_name=font_name, font_color=font_color,
                          border_color=border_color, border_width=border_width,
                          bg_color=bg_color, bg_opacity=bg_opacity)

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', vf,
        '-c:a', 'copy',
        *video_encode_args(QUALITY),
        *METADATA_SCRUB,
        '-movflags', '+faststart',
        output_path
    ]

    _log(f"🎬 Burning subtitles: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors='replace')
        _log(f"❌ FFmpeg Subtitle Error: {stderr_text}")
        raise Exception(f"FFmpeg failed: {stderr_text}")

    return True

