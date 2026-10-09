"""Comprehensive automated tests for the 4-API-Key Consensus Pipeline.

Covers:
  1. Each stage independently (Stage 1, 2, 3)
  2. Matching clips with identical timestamps (IoU = 1.0)
  3. Matching clips with slightly different timestamps (IoU >= 0.5)
  4. Semantic matches with small temporal differences
  5. Non-overlapping clips (IoU = 0)
  6. Duplicate candidate removal
  7. Clips found by only one or two stages (rejected under strict consensus)
  8. Strict three-stage consensus
  9. Final review by Stage 4
  10. Invalid AI responses and schema validation
  11. Missing API keys and startup validation
  12. Authentication errors and rate limits
  13. Timeouts and retry limits
  14. Empty candidate sets
  15. Source-video duration validation
  16. Regression testing of downstream clip exports & metadata compatibility
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

import consensus_config
from consensus_client import (
    ConsensusAuthError,
    ConsensusStageExecutionError,
    execute_consensus_call,
    is_permanent_error,
    is_transient_error,
)
from consensus_matching import (
    are_candidates_matching,
    calculate_semantic_similarity,
    calculate_temporal_iou,
    cluster_multi_stage_candidates,
    deduplicate_stage_candidates,
)
from consensus_models import (
    CandidateClip,
    MatchedCandidateGroup,
    Stage4Response,
    Stage4ReviewDecision,
    StageCandidatesResponse,
    StageExecutionMetadata,
)
from consensus_pipeline import (
    _validate_and_sanitize_candidates,
    run_consensus_pipeline,
)
from consensus_ranking import compute_clip_score, rank_consensus_clips


# ==============================================================================
# 1. Independent Stage Tests (Stages 1, 2, 3)
# ==============================================================================

def test_stage1_candidate_model_and_validation():
    """Verify Stage 1 CandidateClip schema creation and duration auto-computation."""
    clip = CandidateClip(
        candidate_id="c1_001",
        start=10.0,
        end=35.0,
        summary="A great discussion on AI startups",
        hook="Stop building wrappers",
        virality_score=85,
        confidence_score=0.9,
    )
    assert clip.candidate_id == "c1_001"
    assert clip.duration == 25.0
    assert clip.viral_hook_text == "Stop building wrappers"
    assert clip.virality_score == 85


def test_stage2_candidate_model_independent():
    """Verify Stage 2 CandidateClip handles narrative validation."""
    clip = CandidateClip(
        candidate_id="c2_001",
        start=15.0,
        end=45.0,
        summary="Complete narrative breakdown",
        hook="Why most products fail",
        virality_score=78,
        confidence_score=0.85,
        why="Self-contained story with no missing context",
    )
    assert clip.duration == 30.0
    assert clip.why == "Self-contained story with no missing context"


def test_stage3_candidate_model_independent():
    """Verify Stage 3 CandidateClip handles virality metrics."""
    clip = CandidateClip(
        candidate_id="c3_001",
        start=12.0,
        end=42.0,
        summary="Shocking statistic revealed",
        hook="97% of creators do this wrong",
        virality_score=92,
        confidence_score=0.95,
    )
    assert clip.virality_score == 92
    assert clip.confidence_score == 0.95


# ==============================================================================
# 2. Matching Clips with Identical Timestamps
# ==============================================================================

def test_matching_identical_timestamps():
    """Clips with identical start and end must have IoU = 1.0 and match."""
    iou = calculate_temporal_iou(10.0, 40.0, 10.0, 40.0)
    assert iou == 1.0

    c1 = CandidateClip(candidate_id="s1", start=10.0, end=40.0, summary="Identical text", hook="hook")
    c2 = CandidateClip(candidate_id="s2", start=10.0, end=40.0, summary="Identical text", hook="hook")

    is_match, match_iou, _ = are_candidates_matching(c1, c2, iou_threshold=0.5)
    assert is_match is True
    assert match_iou == 1.0


# ==============================================================================
# 3. Matching Clips with Slightly Different Timestamps
# ==============================================================================

def test_matching_slightly_different_timestamps():
    """Clips with minor boundary shifts (e.g. 10-40s vs 12-42s) must have high IoU and match."""
    # Overlap: 40 - 12 = 28; Union: 42 - 10 = 32; IoU = 28/32 = 0.875
    iou = calculate_temporal_iou(10.0, 40.0, 12.0, 42.0)
    assert iou == pytest.approx(0.875, 0.01)

    c1 = CandidateClip(candidate_id="s1", start=10.0, end=40.0, summary="Growth hacking", hook="Growth")
    c2 = CandidateClip(candidate_id="s2", start=12.0, end=42.0, summary="Growth tips", hook="Growth tips")

    is_match, match_iou, _ = are_candidates_matching(c1, c2, iou_threshold=0.5)
    assert is_match is True
    assert match_iou >= 0.5


# ==============================================================================
# 4. Semantic Matches with Small Temporal Differences
# ==============================================================================

def test_semantic_matches_with_small_temporal_differences():
    """Clips with moderate IoU (e.g. 0.35) but strong semantic overlap match via secondary signal."""
    # 10 to 40 vs 25 to 55 -> overlap = 15, union = 45 -> IoU = 0.333
    iou = calculate_temporal_iou(10.0, 40.0, 25.0, 55.0)
    assert 0.30 <= iou < 0.50

    c1 = CandidateClip(
        candidate_id="s1", start=10.0, end=40.0,
        summary="Secret formula for viral retention on TikTok and Reels",
        hook="The secret formula for TikTok retention"
    )
    c2 = CandidateClip(
        candidate_id="s2", start=25.0, end=55.0,
        summary="Viral TikTok and Reels retention secret formula explained",
        hook="How to master TikTok retention"
    )

    sim = calculate_semantic_similarity(c1.summary, c2.summary)
    assert sim >= 0.35

    is_match, _, _ = are_candidates_matching(c1, c2, iou_threshold=0.5, semantic_threshold=0.35)
    assert is_match is True


# ==============================================================================
# 5. Non-Overlapping Clips
# ==============================================================================

def test_non_overlapping_clips_do_not_match():
    """Clips with 0 temporal overlap must NEVER match, even if talking about similar topics."""
    iou = calculate_temporal_iou(10.0, 30.0, 40.0, 60.0)
    assert iou == 0.0

    c1 = CandidateClip(candidate_id="s1", start=10.0, end=30.0, summary="Bitcoin price cycle", hook="Crypto")
    c2 = CandidateClip(candidate_id="s2", start=40.0, end=60.0, summary="Bitcoin price cycle", hook="Crypto")

    is_match, match_iou, _ = are_candidates_matching(c1, c2, iou_threshold=0.5)
    assert is_match is False
    assert match_iou == 0.0


# ==============================================================================
# 6. Duplicate Candidate Removal
# ==============================================================================

def test_duplicate_candidate_removal():
    """Intra-stage deduplication removes heavily overlapping candidates, keeping the higher score."""
    c1 = CandidateClip(candidate_id="cand_a", start=10.0, end=40.0, virality_score=90, summary="A")
    c2 = CandidateClip(candidate_id="cand_b", start=12.0, end=38.0, virality_score=70, summary="B")
    c3 = CandidateClip(candidate_id="cand_c", start=60.0, end=90.0, virality_score=85, summary="C")

    deduped = deduplicate_stage_candidates([c1, c2, c3], overlap_threshold=0.5)
    assert len(deduped) == 2
    # c1 kept over c2 because c1 has score 90 vs 70
    assert any(c.candidate_id == "cand_a" for c in deduped)
    assert not any(c.candidate_id == "cand_b" for c in deduped)
    assert any(c.candidate_id == "cand_c" for c in deduped)


# ==============================================================================
# 7. Clips Found by Only One or Two Stages
# ==============================================================================

def test_clips_found_by_only_one_or_two_stages():
    """Clips supported by only 1 or 2 stages must NOT be marked as three-stage consensus."""
    candidates_by_stage = {
        1: [CandidateClip(candidate_id="s1_1", start=10.0, end=35.0, virality_score=80, summary="Moment 1")],
        2: [CandidateClip(candidate_id="s2_1", start=11.0, end=36.0, virality_score=82, summary="Moment 1")],
        # Stage 3 did NOT find Moment 1, found Moment 2 instead
        3: [CandidateClip(candidate_id="s3_2", start=80.0, end=110.0, virality_score=75, summary="Moment 2")],
    }

    groups = cluster_multi_stage_candidates(candidates_by_stage, iou_threshold=0.5)
    assert len(groups) == 2

    # Group 1 has stages [1, 2] -> is_three_stage_consensus must be False
    g1 = next(g for g in groups if 1 in g.stages_supported and 2 in g.stages_supported)
    assert g1.stages_supported == [1, 2]
    assert g1.is_three_stage_consensus is False

    # Group 2 has stage [3] -> is_three_stage_consensus must be False
    g2 = next(g for g in groups if g.stages_supported == [3])
    assert g2.is_three_stage_consensus is False


# ==============================================================================
# 8. Strict Three-Stage Consensus
# ==============================================================================

def test_strict_three_stage_consensus_filtering():
    """When all 3 stages independently find the moment, is_three_stage_consensus must be True."""
    candidates_by_stage = {
        1: [CandidateClip(candidate_id="s1_1", start=20.0, end=50.0, virality_score=85, summary="Viral Moment")],
        2: [CandidateClip(candidate_id="s2_1", start=21.0, end=49.0, virality_score=88, summary="Viral Moment")],
        3: [CandidateClip(candidate_id="s3_1", start=20.5, end=51.0, virality_score=82, summary="Viral Moment")],
    }

    groups = cluster_multi_stage_candidates(candidates_by_stage, iou_threshold=0.5)
    assert len(groups) == 1
    grp = groups[0]
    assert grp.stages_supported == [1, 2, 3]
    assert grp.is_three_stage_consensus is True
    assert grp.consensus_iou >= 0.8
    assert 20.0 <= grp.avg_start <= 21.0


# ==============================================================================
# 9. Final Review by Stage 4
# ==============================================================================

def test_stage4_final_review_decision_and_ranking():
    """Stage 4 reviews candidate group, refines timestamps, and ranks approved items."""
    grp = MatchedCandidateGroup(
        group_id="consensus_group_001",
        stages_supported=[1, 2, 3],
        is_three_stage_consensus=True,
        consensus_iou=0.9,
        avg_start=20.5,
        avg_end=50.0,
        avg_virality_score=85.0,
        avg_confidence=0.9,
        combined_summary="Mind-blowing AI agent architecture",
        combined_hook="Why multi-agent systems will replace single prompts",
    )

    decision = Stage4ReviewDecision(
        group_id="consensus_group_001",
        approved=True,
        consensus_level="3_stage_unanimous",
        refined_start=20.0,
        refined_end=49.5,
        reviewer_score=94,
        reviewer_notes="Unanimous 3-stage consensus. Refined start slightly to capture opening statement.",
        viral_hook_text="Stop writing single prompts",
        video_title_for_youtube_short="The Multi-Agent Revolution",
    )

    score, breakdown = compute_clip_score(grp, decision)
    assert 80 <= score <= 100
    assert breakdown["stages_count"] == 3
    assert breakdown["agreement_score"] == 100.0

    ranked = rank_consensus_clips([(grp, decision)])
    assert len(ranked) == 1
    assert ranked[0]["score"] == score
    assert ranked[0]["decision"].refined_start == 20.0


# ==============================================================================
# 10. Invalid AI Responses & Schema Validation
# ==============================================================================

def test_invalid_ai_responses_and_schema_validation(monkeypatch):
    """Malformed AI responses trigger retry and eventually raise ConsensusStageExecutionError."""
    monkeypatch.setenv("AI_API_KEY_1", "test_key_12345678")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = ValueError("Gemini returned an empty response body.")

    with patch("consensus_client.genai.Client", return_value=mock_client):
        with pytest.raises(ConsensusStageExecutionError) as exc_info:
            execute_consensus_call(
                stage=1,
                prompt="test prompt",
                response_schema=StageCandidatesResponse,
                max_retries=2,
            )
        assert "failed on attempt 2/2" in str(exc_info.value)


# ==============================================================================
# 11. Missing API Keys and Startup Validation
# ==============================================================================

def test_missing_api_keys_startup_validation(monkeypatch):
    """Missing required keys returns is_valid=False with clear descriptive messages."""
    for s in (1, 2, 3, 4):
        monkeypatch.delenv(f"AI_API_KEY_{s}", raising=False)
        monkeypatch.delenv(f"GEMINI_API_KEY_{s}", raising=False)

    is_valid, issues = consensus_config.validate_consensus_config()
    assert is_valid is False
    assert any("Missing required API key(s)" in issue for issue in issues)


def test_startup_validation_succeeds_when_all_four_keys_present(monkeypatch):
    """Startup validation passes when all four keys are configured."""
    for s in (1, 2, 3, 4):
        monkeypatch.setenv(f"AI_API_KEY_{s}", f"key_stage_{s}_secret123")

    is_valid, issues = consensus_config.validate_consensus_config()
    assert is_valid is True
    assert len(issues) == 0


def test_mask_key_never_leaks_secrets():
    """Ensure mask_key truncates secrets safely."""
    assert consensus_config.mask_key("AIzaSyD-1234567890abcdef") == "AIz...cdef"
    assert consensus_config.mask_key(None) == "<unset>"
    assert consensus_config.mask_key("short") == "***"


# ==============================================================================
# 12. Authentication Errors and Rate Limits
# ==============================================================================

def test_permanent_auth_error_not_retried(monkeypatch):
    """401/403/API_KEY_INVALID raises ConsensusAuthError immediately without wasting retries."""
    monkeypatch.setenv("AI_API_KEY_1", "invalid_key_12345678")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("401 API_KEY_INVALID: Key was revoked")

    with patch("consensus_client.genai.Client", return_value=mock_client):
        with pytest.raises(ConsensusAuthError) as exc_info:
            execute_consensus_call(
                stage=1,
                prompt="prompt",
                response_schema=StageCandidatesResponse,
                max_retries=3,
            )
        assert "Authentication / Permission error" in str(exc_info.value)
        # Verify it did not retry 3 times
        assert mock_client.models.generate_content.call_count == 1


def test_transient_error_detection():
    """Verify is_transient_error identifies 429, 503, RESOURCE_EXHAUSTED."""
    assert is_transient_error("429 RESOURCE_EXHAUSTED: rate limit exceeded") is True
    assert is_transient_error("503 Service Unavailable") is True
    assert is_transient_error("ReadTimeout: connection closed") is True
    assert is_transient_error("SyntaxError: bad syntax") is False


# ==============================================================================
# 13. Timeouts and Retry Limits
# ==============================================================================

def test_transient_retry_and_exhaustion(monkeypatch):
    """Transient errors are retried up to max_retries, then raise ConsensusStageExecutionError."""
    monkeypatch.setenv("AI_API_KEY_2", "valid_key_stage_2")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("503 UNAVAILABLE: Server overloaded")

    with patch("consensus_client.genai.Client", return_value=mock_client):
        with patch("time.sleep"):  # Mock sleep so tests run instantly
            with pytest.raises(ConsensusStageExecutionError) as exc_info:
                execute_consensus_call(
                    stage=2,
                    prompt="prompt",
                    response_schema=StageCandidatesResponse,
                    max_retries=2,
                )
            assert mock_client.models.generate_content.call_count == 2
            assert "failed on attempt 2/2" in str(exc_info.value)


# ==============================================================================
# 14. Empty Candidate Sets
# ==============================================================================

def test_empty_candidate_sets_produce_clean_report():
    """Empty candidate responses result in 0 clips with clear consensus report."""
    fake_transcript = {"language": "en", "segments": [{"start": 0, "end": 20, "text": "Hello world"}]}

    with patch("consensus_pipeline._run_single_discovery_stage") as mock_disc:
        meta = MagicMock(status="success", latency_seconds=1.0, candidate_count=0, cost_usd=0.01)
        mock_disc.return_value = (1, [], meta)

        result = run_consensus_pipeline(fake_transcript, video_duration=20.0, strict_consensus=True)
        assert result["shorts"] == []
        assert result["consensus_report"]["three_stage_consensus_count"] == 0
        assert result["consensus_report"]["consensus_achieved"] is False


# ==============================================================================
# 15. Source-Video Duration Validation
# ==============================================================================

def test_source_video_duration_validation():
    """Timestamps outside video bounds are sanitized or dropped."""
    cands = [
        # Valid clip
        CandidateClip(candidate_id="c1", start=5.0, end=35.0, summary="Good"),
        # Start past duration -> dropped
        CandidateClip(candidate_id="c2", start=120.0, end=140.0, summary="Past duration"),
        # Inverted timestamps (end < start) -> dropped
        CandidateClip(candidate_id="c3", start=50.0, end=30.0, summary="Inverted"),
        # End past duration -> clamped to 100.0
        CandidateClip(candidate_id="c4", start=80.0, end=130.0, summary="Over duration"),
    ]

    sanitized = _validate_and_sanitize_candidates(
        cands, video_duration=100.0, min_secs=15.0, max_secs=60.0, stage_prefix="test"
    )

    ids = [c.candidate_id for c in sanitized]
    assert "c1" in ids
    assert "c2" not in ids
    assert "c3" not in ids
    assert "c4" in ids
    # Check that c4 was clamped to video_duration
    c4 = next(c for c in sanitized if c.candidate_id == "c4")
    assert c4.end <= 100.0


# ==============================================================================
# 16. Full Pipeline Integration & Metadata Regression Test
# ==============================================================================

def test_full_pipeline_consensus_integration(monkeypatch):
    """End-to-end simulation of 4-stage consensus pipeline with downstream shape validation."""
    for s in (1, 2, 3, 4):
        monkeypatch.setenv(f"AI_API_KEY_{s}", f"mock_key_{s}_12345")
    monkeypatch.setenv("CONSENSUS_STRICT", "1")

    fake_transcript = {
        "language": "en",
        "segments": [
            {"start": 10.0, "end": 25.0, "text": "This is the big secret of productivity.", "words": [
                {"word": "This", "start": 10.0, "end": 10.3},
                {"word": "productivity.", "start": 24.5, "end": 25.0},
            ]},
            {"start": 25.0, "end": 45.0, "text": "Never check your email before noon.", "words": [
                {"word": "Never", "start": 25.0, "end": 25.4},
                {"word": "noon.", "start": 44.5, "end": 45.0},
            ]},
        ]
    }

    # Stage 1, 2, and 3 all discover the same moment (10s to 45s)
    s1_cand = CandidateClip(candidate_id="s1_01", start=10.0, end=45.0, virality_score=88, hook="Email rule", summary="Never email before noon")
    s2_cand = CandidateClip(candidate_id="s2_01", start=10.5, end=44.5, virality_score=84, hook="Email rule", summary="Never email before noon")
    s3_cand = CandidateClip(candidate_id="s3_01", start=10.0, end=45.0, virality_score=90, hook="Email rule", summary="Never email before noon")

    def mock_discovery_stage(stg, *args, **kwargs):
        meta = StageExecutionMetadata(stage=stg, model="gemini-2.5-flash", status="success", latency_seconds=0.5, candidate_count=1)
        cands = [s1_cand] if stg == 1 else ([s2_cand] if stg == 2 else [s3_cand])
        return stg, cands, meta

    # Stage 4 approves it
    s4_resp = Stage4Response(decisions=[
        Stage4ReviewDecision(
            group_id="consensus_group_001",
            approved=True,
            consensus_level="3_stage_unanimous",
            refined_start=10.0,
            refined_end=45.0,
            reviewer_score=95,
            viral_hook_text="Stop checking email at 9 AM",
            video_title_for_youtube_short="The 12 PM Email Rule",
            video_description_for_tiktok="Change your mornings forever #productivity",
            video_description_for_instagram="Morning routine hack #focus",
            reviewer_notes="Unanimous agreement across all 3 models. Clear opening statement.",
        )
    ])

    with patch("consensus_pipeline._run_single_discovery_stage", side_effect=mock_discovery_stage):
        with patch("consensus_pipeline.execute_consensus_call") as mock_s4:
            mock_s4.return_value = (s4_resp, {"latency_seconds": 0.8, "cost_usd": 0.002})

            result = run_consensus_pipeline(fake_transcript, video_duration=60.0)

            # Assert output format meets downstream expectations
            assert "shorts" in result
            assert "consensus_report" in result
            assert len(result["shorts"]) == 1

            short = result["shorts"][0]
            assert "start" in short
            assert "end" in short
            assert "predicted_score" in short
            assert "video_title_for_youtube_short" in short
            assert "viral_hook_text" in short
            assert "consensus_metadata" in short

            # Verify consensus metadata
            meta = short["consensus_metadata"]
            assert meta["is_three_stage_consensus"] is True
            assert meta["stages_supported"] == [1, 2, 3]
            assert meta["reviewer_score"] == 95

            # Verify report
            report = result["consensus_report"]
            assert report["consensus_achieved"] is True
            assert report["three_stage_consensus_count"] == 1
            assert report["final_approved_count"] == 1
