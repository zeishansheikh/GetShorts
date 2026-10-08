import type { CaptionWord } from "./types";

export interface CaptionBlock {
  words: CaptionWord[];
  startMs: number;
  endMs: number;
  text: string;
}

/**
 * Groups word-level captions into display blocks.
 * Same logic as OpenShorts' generate_srt: max chars per block, max duration per block.
 */
export function groupCaptionsIntoBlocks(
  captions: CaptionWord[],
  maxChars = 20,
  maxDurationMs = 2000
): CaptionBlock[] {
  const blocks: CaptionBlock[] = [];
  let currentWords: CaptionWord[] = [];
  let blockStartMs = 0;

  for (const word of captions) {
    if (currentWords.length === 0) {
      currentWords.push(word);
      blockStartMs = word.startMs;
      continue;
    }

    const currentTextLen = currentWords.reduce(
      (sum, w) => sum + w.text.length + 1,
      0
    );
    const duration = word.endMs - blockStartMs;

    if (
      currentTextLen + word.text.length > maxChars ||
      duration > maxDurationMs
    ) {
      // Finalize current block
      const lastWord = currentWords[currentWords.length - 1];
      blocks.push({
        words: [...currentWords],
        startMs: blockStartMs,
        endMs: lastWord.endMs,
        text: currentWords.map((w) => w.text).join(" "),
      });

      currentWords = [word];
      blockStartMs = word.startMs;
    } else {
      currentWords.push(word);
    }
  }

  // Final block
  if (currentWords.length > 0) {
    const lastWord = currentWords[currentWords.length - 1];
    blocks.push({
      words: [...currentWords],
      startMs: blockStartMs,
      endMs: lastWord.endMs,
      text: currentWords.map((w) => w.text).join(" "),
    });
  }

  return blocks;
}

export interface KineticCaptionBlock {
  words: CaptionWord[];
  startMs: number;
  endMs: number;
  topText: string;
  punchWord: string;
}

const STOP_WORDS = new Set([
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
]);

const HIGH_IMPACT_WORDS = new Set([
  "everything", "nothing", "never", "always", "skills", "money", "million", "billion",
  "dollar", "dollars", "secret", "truth", "viral", "huge", "insane", "crazy", "fast",
  "power", "powerful", "success", "life", "future", "world", "stop", "danger", "win",
  "winner", "lose", "free", "dead", "kill", "die", "best", "worst", "mistake", "fix",
  "rule", "rules", "game", "change", "changed", "business", "growth", "watch", "now",
  "everybody", "everyone", "nobody", "forever", "impossible", "simple", "easy", "hard"
]);

/**
 * Score and detect the punch keyword in a 1-4 word phrase.
 * Returns topText (supporting words) and punchWord (BIG uppercase keyword).
 */
export function detectPunchWord(words: CaptionWord[]): { topText: string; punchWord: string } {
  if (!words || words.length === 0) {
    return { topText: "", punchWord: "" };
  }

  const cleanWord = (t: string) => t.toLowerCase().replace(/[^a-z0-9]/g, "");

  if (words.length === 1) {
    const raw = words[0].text.trim().replace(/^[^a-zA-Z0-9$!%?]+|[^a-zA-Z0-9$!%?]+$/g, "");
    return {
      topText: "",
      punchWord: (raw || words[0].text.trim()).toUpperCase(),
    };
  }

  // Score each word to find the punch keyword
  let bestScore = -9999;
  let bestIdx = words.length - 1;

  words.forEach((w, idx) => {
    const clean = cleanWord(w.text);
    if (!clean) return;

    let score = clean.length * 1.5;

    if (HIGH_IMPACT_WORDS.has(clean)) {
      score += 15;
    }
    if (/\d/.test(clean)) {
      score += 12;
    }
    if (w.text === w.text.toUpperCase() && clean.length > 1) {
      score += 8;
    }
    if (STOP_WORDS.has(clean)) {
      score -= 10;
    }

    // End-focus bias: In English phrasing, punch keywords naturally conclude the phrase
    score += idx * 2.0;

    if (score > bestScore) {
      bestScore = score;
      bestIdx = idx;
    }
  });

  // If the selected word is at the very end, supporting words are on top line
  if (bestIdx === words.length - 1) {
    const topText = words.slice(0, bestIdx).map((w) => w.text.trim()).join(" ");
    const punchRaw = words[bestIdx].text.trim().replace(/^[^a-zA-Z0-9$!%?]+|[^a-zA-Z0-9$!%?]+$/g, "");
    return {
      topText,
      punchWord: (punchRaw || words[bestIdx].text.trim()).toUpperCase(),
    };
  }

  // If the selected punch word was in the middle or first:
  // Check if the last word is a non-stop content word; if so, make it the punch word
  const lastClean = cleanWord(words[words.length - 1].text);
  if (!STOP_WORDS.has(lastClean) && lastClean.length >= 3) {
    const topText = words.slice(0, -1).map((w) => w.text.trim()).join(" ");
    const punchRaw = words[words.length - 1].text.trim().replace(/^[^a-zA-Z0-9$!%?]+|[^a-zA-Z0-9$!%?]+$/g, "");
    return {
      topText,
      punchWord: (punchRaw || words[words.length - 1].text.trim()).toUpperCase(),
    };
  }

  // Otherwise, split at bestIdx: supporting words before bestIdx, punch word (+ trailing particles) on line 2
  const topWords = words.slice(0, bestIdx).map((w) => w.text.trim());
  const punchWords = words.slice(bestIdx).map((w) => w.text.trim());
  return {
    topText: topWords.join(" "),
    punchWord: punchWords.join(" ").toUpperCase(),
  };
}

/**
 * Groups word-level captions into kinetic phrase blocks (1-4 words per caption, ~0.8-1.5s).
 * Strictly enforces:
 * - 1-4 words per caption
 * - Duration ~0.8-1.5s
 * - Phrase-by-phrase sync to speech
 * - Two-line stack structure with punch keyword auto-detected
 */
export function groupCaptionsIntoKineticBlocks(
  captions: CaptionWord[],
  minDurationMs = 700,
  maxDurationMs = 1500
): KineticCaptionBlock[] {
  if (!captions || captions.length === 0) return [];

  const blocks: KineticCaptionBlock[] = [];
  let currentWords: CaptionWord[] = [];
  let blockStartMs = 0;

  const cleanWord = (t: string) => t.toLowerCase().replace(/[^a-z0-9]/g, "");

  for (let i = 0; i < captions.length; i++) {
    const word = captions[i];
    if (currentWords.length === 0) {
      currentWords.push(word);
      blockStartMs = word.startMs;
      continue;
    }

    const duration = word.endMs - blockStartMs;
    const prevWord = currentWords[currentWords.length - 1];
    const speechPause = word.startMs - prevWord.endMs;
    const endsWithTerminalPunctuation = /[.!?]$/.test(prevWord.text.trim());

    // Check if previous single word is an emphatic high-impact word that deserves its own block
    const isSingleEmphatic =
      currentWords.length === 1 &&
      HIGH_IMPACT_WORDS.has(cleanWord(prevWord.text)) &&
      duration >= 600;

    const shouldClose =
      currentWords.length >= 4 ||
      duration >= maxDurationMs ||
      speechPause > 300 ||
      endsWithTerminalPunctuation ||
      isSingleEmphatic ||
      // When at least 2 words and >= minDurationMs, avoid keeping a trailing phrase boundary
      (currentWords.length >= 2 &&
        duration >= minDurationMs &&
        /^(to|in|on|at|for|with|and|but|or|because|if|so|that)$/i.test(word.text.trim()));

    if (shouldClose) {
      const lastW = currentWords[currentWords.length - 1];
      const { topText, punchWord } = detectPunchWord(currentWords);
      blocks.push({
        words: [...currentWords],
        startMs: blockStartMs,
        endMs: lastW.endMs,
        topText,
        punchWord,
      });

      currentWords = [word];
      blockStartMs = word.startMs;
    } else {
      currentWords.push(word);
    }
  }

  if (currentWords.length > 0) {
    const lastW = currentWords[currentWords.length - 1];
    const { topText, punchWord } = detectPunchWord(currentWords);
    blocks.push({
      words: [...currentWords],
      startMs: blockStartMs,
      endMs: lastW.endMs,
      topText,
      punchWord,
    });
  }

  // Extend endMs to next block's startMs if gap is under 150ms to eliminate flicker
  for (let i = 0; i < blocks.length - 1; i++) {
    const nextStart = blocks[i + 1].startMs;
    if (nextStart > blocks[i].endMs && nextStart - blocks[i].endMs < 150) {
      blocks[i].endMs = nextStart;
    }
  }

  return blocks;
}

/**
 * Find the active word at a given time in milliseconds.
 */
export function getActiveWordIndex(
  words: CaptionWord[],
  timeMs: number
): number {
  for (let i = 0; i < words.length; i++) {
    if (timeMs >= words[i].startMs && timeMs < words[i].endMs) {
      return i;
    }
  }
  return -1;
}
