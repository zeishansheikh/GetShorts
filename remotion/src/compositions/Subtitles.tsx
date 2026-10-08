import React from "react";
import {
  AbsoluteFill,
  Sequence,
  useCurrentFrame,
  useVideoConfig,
  spring,
  interpolate,
} from "remotion";
import type { SubtitleConfig } from "../lib/types";
import {
  groupCaptionsIntoBlocks,
  groupCaptionsIntoKineticBlocks,
  getActiveWordIndex,
  type KineticCaptionBlock,
} from "../lib/captions";
import { getFontStack } from "../lib/fonts";

interface SubtitlesProps {
  config: SubtitleConfig;
}

// Hebrew, Arabic, Syriac, Thaana, NKo and their presentation forms. The words
// are laid out as flex items, which always run left to right, so a block in
// one of these scripts has to flip the row itself (github issue #89).
const RTL_CHARS = /[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]/;

const POSITION_MAP: Record<string, React.CSSProperties> = {
  top: { top: "12%", bottom: "auto" },
  middle: { top: "45%", bottom: "auto" },
  bottom: { bottom: "10%", top: "auto" },
};

export const Subtitles: React.FC<SubtitlesProps> = ({ config }) => {
  const { fps } = useVideoConfig();
  const isKinetic =
    config.style?.animation === "kinetic" ||
    (config.style as any)?.style === "kinetic" ||
    config.style?.isKinetic ||
    config.kinetic;

  if (isKinetic) {
    const kineticBlocks = groupCaptionsIntoKineticBlocks(config.captions);
    return (
      <AbsoluteFill>
        {kineticBlocks.map((block, i) => {
          const startFrame = Math.round((block.startMs / 1000) * fps);
          const durationFrames = Math.max(
            1,
            Math.round(((block.endMs - block.startMs) / 1000) * fps)
          );

          return (
            <Sequence
              key={`kinetic-${i}`}
              from={startFrame}
              durationInFrames={durationFrames}
              layout="none"
            >
              <KineticSubtitleBlock
                block={block}
                config={config}
              />
            </Sequence>
          );
        })}
      </AbsoluteFill>
    );
  }

  const blocks = groupCaptionsIntoBlocks(config.captions);

  return (
    <AbsoluteFill>
      {blocks.map((block, i) => {
        const startFrame = Math.round((block.startMs / 1000) * fps);
        const durationFrames = Math.max(
          1,
          Math.round(((block.endMs - block.startMs) / 1000) * fps)
        );

        return (
          <Sequence
            key={i}
            from={startFrame}
            durationInFrames={durationFrames}
            layout="none"
          >
            <SubtitleBlock
              block={block}
              config={config}
              blockStartMs={block.startMs}
            />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

interface KineticSubtitleBlockProps {
  block: KineticCaptionBlock;
  config: SubtitleConfig;
}

const KineticSubtitleBlock: React.FC<KineticSubtitleBlockProps> = ({
  block,
  config,
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  const normalFontSize = Math.round(height * 0.047);
  const baseEmphasisFontSize = Math.round(height * 0.11);
  const lineGap = Math.round(height * (15 / 1920));
  const maxLineWidth = Math.round(width * 0.8);

  const charWidthFactor = 0.58;
  const estimatedPunchWidth = block.punchWord.length * (baseEmphasisFontSize * charWidthFactor);
  const emphasisFontSize =
    estimatedPunchWidth > maxLineWidth
      ? Math.max(normalFontSize, Math.round(maxLineWidth / (block.punchWord.length * charWidthFactor)))
      : baseEmphasisFontSize;

  const popDurationFrames = Math.max(2, Math.round((80 / 1000) * fps));
  const scale =
    frame <= popDurationFrames
      ? interpolate(
          frame,
          [0, Math.floor(popDurationFrames / 2), popDurationFrames],
          [1.0, 1.04, 1.0],
          { extrapolateRight: "clamp" }
        )
      : 1.0;

  const requestedFont = config.style?.fontFamily || "Montserrat ExtraBold";
  const fontStack = getFontStack(requestedFont) || `'Montserrat-ExtraBold', 'Montserrat', sans-serif`;

  return (
    <div
      style={{
        position: "absolute",
        left: "50%",
        top: "56.5%",
        transform: `translate(-50%, -50%) scale(${scale})`,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        maxWidth: "80%",
        width: "max-content",
        boxSizing: "border-box",
        pointerEvents: "none",
        zIndex: 50,
      }}
    >
      {block.topText ? (
        <div
          style={{
            fontFamily: fontStack,
            fontSize: normalFontSize,
            fontWeight: 800,
            color: "#FFFFFF",
            lineHeight: 1.1,
            textAlign: "center",
            maxWidth: "100%",
            wordBreak: "break-word",
            overflowWrap: "break-word",
            marginBottom: lineGap,
            textShadow: "none",
          }}
        >
          {block.topText}
        </div>
      ) : null}
      <div
        style={{
          fontFamily: fontStack,
          fontSize: emphasisFontSize,
          fontWeight: 800,
          color: "#FFFFFF",
          textTransform: "uppercase",
          lineHeight: 0.95,
          letterSpacing: "0.01em",
          textAlign: "center",
          maxWidth: "100%",
          wordBreak: "break-word",
          overflowWrap: "break-word",
          textShadow: "none",
        }}
      >
        {block.punchWord}
      </div>
    </div>
  );
};

interface SubtitleBlockProps {
  block: ReturnType<typeof groupCaptionsIntoBlocks>[number];
  config: SubtitleConfig;
  blockStartMs: number;
}

const SubtitleBlock: React.FC<SubtitleBlockProps> = ({
  block,
  config,
  blockStartMs,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { style, position } = config;

  // Current time relative to composition start (sequence-relative frame)
  const currentTimeMs = blockStartMs + (frame / fps) * 1000;
  const activeIndex = getActiveWordIndex(block.words, currentTimeMs);

  const positionStyle = POSITION_MAP[position] ?? POSITION_MAP.bottom;
  const fontStack = getFontStack(style.fontFamily);

  // Background box style
  const hasBg = style.bgOpacity > 0;
  const bgStyle: React.CSSProperties = hasBg
    ? {
        backgroundColor: `${style.bgColor}${Math.round(style.bgOpacity * 255)
          .toString(16)
          .padStart(2, "0")}`,
        borderRadius: 8,
        padding: "8px 16px",
      }
    : {};

  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        display: "flex",
        justifyContent: "center",
        ...positionStyle,
      }}
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "center",
          direction: block.words.some((w) => RTL_CHARS.test(w.text)) ? "rtl" : "ltr",
          gap: "6px 8px",
          maxWidth: "80%",
          padding: "0 24px",
          boxSizing: "border-box",
          ...bgStyle,
        }}
      >
        {block.words.map((word, i) => (
          <WordSpan
            key={i}
            word={word.text}
            isActive={i === activeIndex}
            style={style}
            fontStack={fontStack}
            animation={style.animation}
            frame={frame}
            fps={fps}
            wordStartMs={word.startMs}
            blockStartMs={blockStartMs}
          />
        ))}
      </div>
    </div>
  );
};

interface WordSpanProps {
  word: string;
  isActive: boolean;
  style: SubtitleConfig["style"];
  fontStack: string;
  animation: SubtitleConfig["style"]["animation"];
  frame: number;
  fps: number;
  wordStartMs: number;
  blockStartMs: number;
}

const WordSpan: React.FC<WordSpanProps> = ({
  word,
  isActive,
  style,
  fontStack,
  animation,
  frame,
  fps,
  wordStartMs,
  blockStartMs,
}) => {
  const wordStartFrame = Math.round(
    ((wordStartMs - blockStartMs) / 1000) * fps
  );

  let transform = "";
  let color = style.fontColor;
  let extraStyle: React.CSSProperties = {};

  if (isActive) {
    color = style.highlightColor;

    switch (animation) {
      case "pop": {
        const scale = spring({
          frame: frame - wordStartFrame,
          fps,
          config: { mass: 0.5, stiffness: 300, damping: 12 },
          durationInFrames: 10,
        });
        const scaleValue = interpolate(scale, [0, 1], [1, 1.25]);
        transform = `scale(${scaleValue})`;
        break;
      }
      case "karaoke": {
        extraStyle = {
          backgroundColor: style.highlightColor,
          color: style.bgColor || "#000000",
          borderRadius: 4,
          padding: "2px 6px",
        };
        break;
      }
      case "word-highlight": {
        extraStyle = {
          textShadow: `0 0 12px ${style.highlightColor}, 0 0 24px ${style.highlightColor}40`,
        };
        break;
      }
      default:
        break;
    }
  }

  // Text stroke via textShadow (CSS paint-order not reliable in Remotion)
  const strokeShadow =
    style.borderWidth > 0
      ? [
          `${style.borderWidth}px 0 0 ${style.borderColor}`,
          `-${style.borderWidth}px 0 0 ${style.borderColor}`,
          `0 ${style.borderWidth}px 0 ${style.borderColor}`,
          `0 -${style.borderWidth}px 0 ${style.borderColor}`,
        ].join(", ")
      : "none";

  return (
    <span
      style={{
        fontFamily: fontStack,
        fontSize: style.fontSize,
        fontWeight: 700,
        color: animation === "karaoke" && isActive ? undefined : color,
        textShadow:
          animation !== "karaoke"
            ? [strokeShadow, extraStyle.textShadow].filter(Boolean).join(", ")
            : strokeShadow,
        transform,
        display: "inline-block",
        transition: "none",
        ...extraStyle,
      }}
    >
      {word}
    </span>
  );
};
