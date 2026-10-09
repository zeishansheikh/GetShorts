"""Core orchestrator for the 4-API-Key Consensus Pipeline.

Executes:
  Stage 1 (Key 1): Initial Clip Discovery
  Stage 2 (Key 2): Independent Quality Evaluation
  Stage 3 (Key 3): Independent Virality Evaluation
  Stage 4 (Key 4): Final Consensus Review & Filtering

Enforces:
  - True independent multi-key evaluation
  - Strict 3-stage consensus filtering by default
  - Temporal IoU and semantic similarity candidate matching
  - Transparent ranking and scoring
  - Millisecond precision word boundary snapping
  - Complete error traceability and reporting
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import clip_selection
import consensus_config
from consensus_client import (
    ConsensusAuthError,
    ConsensusQuotaError,
    ConsensusStageExecutionError,
    execute_consensus_call,
)
from consensus_matching import cluster_multi_stage_candidates
from consensus_models import (
    CandidateClip,
    ConsensusReport,
    MatchedCandidateGroup,
    Stage4Response,
    Stage4ReviewDecision,
    StageCandidatesResponse,
    StageExecutionMetadata,
)
from consensus_prompts import (
    STAGE1_PROMPT_TEMPLATE,
    STAGE2_PROMPT_TEMPLATE,
    STAGE3_PROMPT_TEMPLATE,
    STAGE4_PROMPT_TEMPLATE,
)
from consensus_ranking import rank_consensus_clips


def _format_transcript_for_prompt(transcript: Dict[str, Any], max_chars: int = 40000) -> str:
    """Format Whisper transcript segments into readable text with timestamps."""
    lines: List[str] = []
    total_len = 0
    segments = transcript.get("segments", [])

    for seg in segments:
        s = float(seg.get("start", 0))
        e = float(seg.get("end", 0))
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        line = f"[{s:.2f}s -> {e:.2f}s] {text}"
        lines.append(line)
        total_len += len(line) + 1
        if total_len > max_chars:
            lines.append("... [transcript truncated for context length]")
            break

    if not lines and transcript.get("text"):
        return str(transcript.get("text", "")).strip()

    return "\n".join(lines)


def _validate_and_sanitize_candidates(
    candidates: List[CandidateClip],
    video_duration: float,
    min_secs: float,
    max_secs: float,
    stage_prefix: str,
) -> List[CandidateClip]:
    """Sanitize and validate candidate timestamps against video bounds."""
    valid: List[CandidateClip] = []
    idx = 1

    for cand in candidates:
        s = max(0.0, float(cand.start))
        e = min(float(video_duration), float(cand.end))

        # Check chronology
        if e <= s or s >= video_duration:
            continue

        dur = e - s
        # Ensure within duration band with slight tolerance
        if dur < (min_secs - 3.0) or dur > (max_secs + 5.0):
            # Clamp or drop if degenerate
            if dur < 5.0:
                continue
            if dur > max_secs + 5.0:
                e = min(video_duration, s + max_secs)

        cand.start = round(s, 3)
        cand.end = round(e, 3)
        cand.duration = round(e - s, 3)
        if not cand.candidate_id or cand.candidate_id.strip() == "":
            cand.candidate_id = f"{stage_prefix}_{idx:03d}"
        idx += 1
        valid.append(cand)

    return valid


def _run_single_discovery_stage(
    stage: int,
    prompt_template: str,
    language: str,
    video_duration: float,
    min_secs: float,
    max_secs: float,
    transcript_text: str,
) -> Tuple[int, List[CandidateClip], StageExecutionMetadata]:
    """Execute a single independent discovery stage (Stage 1, 2, or 3)."""
    model_name = consensus_config.get_consensus_model(stage)
    prompt = prompt_template.format(
        language=language,
        video_duration=video_duration,
        min_secs=min_secs,
        max_secs=max_secs,
        transcript_content=transcript_text,
    )

    meta = StageExecutionMetadata(
        stage=stage,
        model=model_name,
        status="running",
    )

    try:
        response, telemetry = execute_consensus_call(
            stage=stage,
            prompt=prompt,
            response_schema=StageCandidatesResponse,
        )
        assert isinstance(response, StageCandidatesResponse)
        raw_candidates = response.candidates or []

        sanitized = _validate_and_sanitize_candidates(
            raw_candidates,
            video_duration=video_duration,
            min_secs=min_secs,
            max_secs=max_secs,
            stage_prefix=f"s{stage}_cand",
        )

        meta.status = "success"
        meta.latency_seconds = telemetry.get("latency_seconds", 0.0)
        meta.candidate_count = len(sanitized)
        meta.cost_usd = telemetry.get("cost_usd")
        tokens = telemetry.get("tokens", {})
        meta.input_tokens = tokens.get("input")
        meta.output_tokens = tokens.get("output")
        return stage, sanitized, meta

    except Exception as e:
        meta.status = "failed"
        meta.error = str(e)
        return stage, [], meta


def run_consensus_pipeline(
    transcript: Dict[str, Any],
    video_duration: float,
    words: Optional[List[Dict[str, Any]]] = None,
    strict_consensus: Optional[bool] = None,
) -> Dict[str, Any]:
    """Execute the full 4-stage AI consensus pipeline.

    Args:
        transcript: Whisper transcript dictionary.
        video_duration: Duration of the source video in seconds.
        words: Ground-truth word timestamps for boundary snapping.
        strict_consensus: Override for strict 3-stage consensus filter.

    Returns:
        Dictionary compatible with downstream GetShorts rendering:
        {
            "shorts": List[Dict[str, Any]],
            "consensus_report": Dict[str, Any],
            "cost_analysis": List[Dict[str, Any]],
        }
    """
    total_start = time.time()
    language = str(transcript.get("language") or "unknown")
    min_secs, max_secs = clip_selection.clip_duration_bounds()
    strict_mode = strict_consensus if strict_consensus is not None else consensus_config.is_strict_consensus()
    iou_thresh = consensus_config.get_iou_threshold()
    semantic_thresh = consensus_config.get_semantic_threshold()

    print(f"🤝 Running 4-Stage AI Consensus Pipeline (strict={strict_mode}, IoU threshold={iou_thresh})...")

    # Extract words list if not supplied
    if words is None:
        words = []
        for segment in transcript.get("segments", []):
            for word in segment.get("words", []):
                words.append({"w": word.get("word", ""), "s": word.get("start", 0), "e": word.get("end", 0)})

    transcript_text = _format_transcript_for_prompt(transcript)

    # Initialize Consensus Report
    report = ConsensusReport(
        strict_consensus_mode=strict_mode,
        consensus_achieved=False,
    )

    stage_definitions = [
        (1, STAGE1_PROMPT_TEMPLATE),
        (2, STAGE2_PROMPT_TEMPLATE),
        (3, STAGE3_PROMPT_TEMPLATE),
    ]

    candidates_by_stage: Dict[int, List[CandidateClip]] = {1: [], 2: [], 3: []}
    stage_errors: Dict[int, str] = {}

    # Run Stages 1, 2, 3 concurrently
    print("   [Stages 1-3] Launching independent discovery across Keys 1, 2, and 3 in parallel...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                _run_single_discovery_stage,
                stg,
                tmpl,
                language,
                video_duration,
                min_secs,
                max_secs,
                transcript_text,
            ): stg
            for stg, tmpl in stage_definitions
        }

        for fut in as_completed(futures):
            stg, cands, meta = fut.result()
            report.stage_telemetry[f"stage_{stg}"] = meta
            if meta.status == "success":
                candidates_by_stage[stg] = cands
                print(f"   ✅ Stage {stg} ({meta.model}) completed in {meta.latency_seconds:.1f}s: {len(cands)} candidate(s)")
            else:
                stage_errors[stg] = meta.error or "Unknown failure"
                print(f"   ❌ Stage {stg} ({meta.model}) failed: {stage_errors[stg]}")

    report.total_candidates_stage1 = len(candidates_by_stage[1])
    report.total_candidates_stage2 = len(candidates_by_stage[2])
    report.total_candidates_stage3 = len(candidates_by_stage[3])

    # Strict failure checking: If any stage 1-3 failed, do NOT claim 3-stage consensus!
    if stage_errors:
        err_details = "; ".join(f"Stage {s}: {err}" for s, err in stage_errors.items())
        print(f"   ⚠️ Consensus pipeline incomplete due to stage failure(s): {err_details}")
        report.rejection_log.append({
            "stage": "discovery_1_to_3",
            "reason": f"One or more discovery stages failed ({err_details}). Cannot verify consensus.",
        })
        return {
            "shorts": [],
            "consensus_report": report.model_dump(),
            "cost_analysis": [],
            "error": f"Consensus pipeline discovery failed: {err_details}",
        }

    # Match and cluster candidates across stages
    matched_groups = cluster_multi_stage_candidates(
        candidates_by_stage,
        iou_threshold=iou_thresh,
        semantic_threshold=semantic_thresh,
    )
    report.matched_groups_total = len(matched_groups)
    three_stage_groups = [g for g in matched_groups if g.is_three_stage_consensus]
    report.three_stage_consensus_count = len(three_stage_groups)

    print(f"   [Matching] Formed {len(matched_groups)} candidate group(s); {len(three_stage_groups)} have 3-stage consensus.")

    # Apply strict consensus filter
    if strict_mode:
        groups_to_review = three_stage_groups
        # Log partial groups into rejection log
        for g in matched_groups:
            if not g.is_three_stage_consensus:
                report.rejection_log.append({
                    "group_id": g.group_id,
                    "reason": f"Lacks 3-stage consensus: only supported by stage(s) {g.stages_supported}",
                    "start": g.avg_start,
                    "end": g.avg_end,
                })
    else:
        groups_to_review = matched_groups

    report.consensus_groups = groups_to_review

    # Check if there are qualifying groups for Stage 4 review
    if not groups_to_review:
        msg = ("No candidate clips satisfied the 3-stage consensus requirement. "
               "The independent AI models did not find sufficient mutual agreement on any moments.")
        print(f"   ℹ️ {msg}")
        report.rejection_log.append({"stage": "matching", "reason": msg})
        return {
            "shorts": [],
            "consensus_report": report.model_dump(),
            "cost_analysis": [],
        }

    # Stage 4: Final Consensus Review
    print(f"   [Stage 4] Reviewing {len(groups_to_review)} consensus candidate group(s) with AI Key 4...")
    candidate_groups_json = json.dumps([
        {
            "group_id": g.group_id,
            "stages_supported": g.stages_supported,
            "avg_start": g.avg_start,
            "avg_end": g.avg_end,
            "avg_virality_score": g.avg_virality_score,
            "combined_summary": g.combined_summary,
            "combined_hook": g.combined_hook,
            "candidate_proposals": [
                {
                    "stage": s,
                    "start": c.start,
                    "end": c.end,
                    "summary": c.summary,
                    "hook": c.hook,
                    "score": c.virality_score,
                }
                for s, c in g.candidates_by_stage.items()
            ]
        }
        for g in groups_to_review
    ], indent=2)

    stage4_prompt = STAGE4_PROMPT_TEMPLATE.format(
        language=language,
        video_duration=video_duration,
        min_secs=min_secs,
        max_secs=max_secs,
        strict_mode="YES" if strict_mode else "NO",
        candidate_groups_json=candidate_groups_json,
    )

    stage4_meta = StageExecutionMetadata(
        stage=4,
        model=consensus_config.get_consensus_model(4),
        status="running",
    )

    try:
        stage4_resp, stage4_telemetry = execute_consensus_call(
            stage=4,
            prompt=stage4_prompt,
            response_schema=Stage4Response,
        )
        assert isinstance(stage4_resp, Stage4Response)
        stage4_meta.status = "success"
        stage4_meta.latency_seconds = stage4_telemetry.get("latency_seconds", 0.0)
        stage4_meta.candidate_count = len(stage4_resp.decisions)
        stage4_meta.cost_usd = stage4_telemetry.get("cost_usd")
        tokens = stage4_telemetry.get("tokens", {})
        stage4_meta.input_tokens = tokens.get("input")
        stage4_meta.output_tokens = tokens.get("output")
        report.stage_telemetry["stage_4"] = stage4_meta
        print(f"   ✅ Stage 4 completed in {stage4_meta.latency_seconds:.1f}s")
    except Exception as e:
        stage4_meta.status = "failed"
        stage4_meta.error = str(e)
        report.stage_telemetry["stage_4"] = stage4_meta
        print(f"   ❌ Stage 4 failed: {e}. Preserving Stage 1-3 results as incomplete approval.")
        return {
            "shorts": [],
            "consensus_report": report.model_dump(),
            "cost_analysis": [],
            "error": f"Stage 4 final consensus review failed: {e}",
        }

    # Match decisions back to groups
    groups_by_id = {g.group_id: g for g in groups_to_review}
    approved_pairs: List[Tuple[MatchedCandidateGroup, Stage4ReviewDecision]] = []

    for dec in stage4_resp.decisions:
        grp = groups_by_id.get(dec.group_id)
        if not grp:
            continue

        if not dec.approved:
            report.rejection_log.append({
                "group_id": dec.group_id,
                "reason": dec.rejection_reason or "Rejected by Stage 4 reviewer",
                "notes": dec.reviewer_notes,
            })
            continue

        # Strict consensus verification: ensure group has 3 stages if strict mode is on
        if strict_mode and not grp.is_three_stage_consensus:
            report.rejection_log.append({
                "group_id": dec.group_id,
                "reason": "Rejected: strict consensus required but moment lacks all 3 discovery stages.",
            })
            continue

        # Validate refined timestamps against video duration
        s = max(0.0, float(dec.refined_start))
        e = min(float(video_duration), float(dec.refined_end))
        if e <= s or s >= video_duration:
            s, e = grp.avg_start, grp.avg_end

        dur = e - s
        if dur < min_secs or dur > max_secs:
            e = min(video_duration, s + max(min_secs, min(max_secs, dur)))

        dec.refined_start = round(s, 3)
        dec.refined_end = round(e, 3)
        approved_pairs.append((grp, dec))

    report.final_approved_count = len(approved_pairs)
    report.consensus_achieved = len(approved_pairs) > 0

    # Rank and score approved clips
    ranked_items = rank_consensus_clips(approved_pairs)

    # Millisecond precision: Snap clip boundaries onto words and build final shorts dict
    final_shorts: List[Dict[str, Any]] = []

    for item in ranked_items:
        grp: MatchedCandidateGroup = item["group"]
        dec: Stage4ReviewDecision = item["decision"]
        raw_start, raw_end = dec.refined_start, dec.refined_end

        # Snap to real word boundaries + pauses
        snap_start, snap_end = clip_selection.snap_clip_to_words(
            raw_start,
            raw_end,
            words,
            video_duration,
            min_duration=min_secs,
            max_duration=max_secs,
        )

        title = dec.video_title_for_youtube_short or grp.combined_hook or "Viral Clip"
        hook_text = dec.viral_hook_text or grp.combined_hook

        short_dict = {
            "start": snap_start,
            "end": snap_end,
            "proposed": [raw_start, raw_end],
            "predicted_score": item["score"],
            "video_title_for_youtube_short": title,
            "viral_hook_text": hook_text,
            "video_description_for_tiktok": dec.video_description_for_tiktok,
            "video_description_for_instagram": dec.video_description_for_instagram,
            "why": dec.reviewer_notes or grp.combined_summary,
            "consensus_metadata": {
                "group_id": grp.group_id,
                "stages_supported": grp.stages_supported,
                "is_three_stage_consensus": grp.is_three_stage_consensus,
                "consensus_iou": grp.consensus_iou,
                "reviewer_score": dec.reviewer_score,
                "score_breakdown": item["breakdown"],
                "reviewer_notes": dec.reviewer_notes,
            },
        }
        final_shorts.append(short_dict)

    # Deduplicate overlapping final clips
    deduped_shorts = clip_selection.dedupe_overlapping(final_shorts)

    # Aggregate costs across telemetry
    costs: List[Dict[str, Any]] = []
    for meta in report.stage_telemetry.values():
        if meta.cost_usd is not None:
            costs.append({
                "stage": meta.stage,
                "model": meta.model,
                "total_cost": meta.cost_usd,
                "input_tokens": meta.input_tokens,
                "output_tokens": meta.output_tokens,
            })

    total_time = round(time.time() - total_start, 2)
    print(f"🎉 Consensus Pipeline finished in {total_time}s: Delivered {len(deduped_shorts)} verified short(s).")

    return {
        "shorts": deduped_shorts,
        "consensus_report": report.model_dump(),
        "cost_analysis": costs,
    }
