"""Pydantic data models for the 4-API-Key Consensus Pipeline.

Enforces strict schemas for:
  - Stage 1, 2, 3 candidate clips
  - Inter-stage matched candidate groups
  - Stage 4 final consensus review decisions
  - Comprehensive consensus execution reporting
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CandidateClip(BaseModel):
    """A single candidate clip discovered or evaluated by an AI stage."""
    candidate_id: str = Field(description="Stable identifier for the candidate clip, e.g. 'c1_001'")
    start: float = Field(description="Start timestamp in absolute seconds from video beginning")
    end: float = Field(description="End timestamp in absolute seconds from video beginning")
    duration: float = Field(default=0.0, description="Duration in seconds (end - start)")
    summary: str = Field(default="", description="Brief content summary of the clip moment")
    hook: str = Field(default="", description="Description of the opening hook")
    virality_score: int = Field(default=50, ge=0, le=100, description="Estimated virality score (0-100)")
    confidence_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Model confidence (0.0-1.0)")
    why: str = Field(default="", description="One sentence explaining why this moment is engaging")
    viral_hook_text: str = Field(default="", description="On-screen punchy hook text (max 10 words)")
    video_title_for_youtube_short: str = Field(default="", description="YouTube Short title (max 100 chars)")
    video_description_for_tiktok: str = Field(default="", description="TikTok description + hashtags")
    video_description_for_instagram: str = Field(default="", description="Instagram description + hashtags")

    def model_post_init(self, __context) -> None:
        if self.duration <= 0.0 and self.end > self.start:
            self.duration = round(self.end - self.start, 3)
        if not self.viral_hook_text and self.hook:
            self.viral_hook_text = self.hook


class StageCandidatesResponse(BaseModel):
    """Output schema enforced on Stage 1, 2, and 3 discovery models."""
    candidates: List[CandidateClip] = Field(default_factory=list)


class MatchedCandidateGroup(BaseModel):
    """A group of candidates from different stages that match the same video moment."""
    group_id: str = Field(description="Unique identifier for the matched candidate group")
    stages_supported: List[int] = Field(default_factory=list, description="Stages that independently found this moment, e.g. [1, 2, 3]")
    is_three_stage_consensus: bool = Field(default=False, description="True if supported by all 3 independent stages")
    consensus_iou: float = Field(default=0.0, description="Average temporal IoU across matching candidate pairs")
    avg_start: float = Field(default=0.0, description="Consensus start timestamp")
    avg_end: float = Field(default=0.0, description="Consensus end timestamp")
    avg_virality_score: float = Field(default=0.0, description="Average virality score across supporting stages")
    avg_confidence: float = Field(default=0.0, description="Average confidence score across supporting stages")
    candidates_by_stage: Dict[str, CandidateClip] = Field(default_factory=dict, description="Keyed by 'stage_1', 'stage_2', 'stage_3'")
    combined_summary: str = Field(default="")
    combined_hook: str = Field(default="")


class Stage4ReviewDecision(BaseModel):
    """Final decision produced by AI Key 4 for a candidate group."""
    group_id: str = Field(description="Group ID being reviewed")
    approved: bool = Field(description="Whether this clip is approved for final publication")
    consensus_level: str = Field(default="unanimous", description="'3_stage_unanimous', 'partial', or 'rejected'")
    refined_start: float = Field(description="Refined start timestamp in absolute seconds")
    refined_end: float = Field(description="Refined end timestamp in absolute seconds")
    reviewer_score: int = Field(default=50, ge=0, le=100, description="Stage 4 reviewer quality score (0-100)")
    rejection_reason: str = Field(default="", description="Reason for rejection if approved=False")
    reviewer_notes: str = Field(default="", description="Reviewer feedback on hook, flow, and standalone value")
    viral_hook_text: str = Field(default="", description="Polished viral hook text for the video overlay")
    video_title_for_youtube_short: str = Field(default="", description="Polished YouTube Short title")
    video_description_for_tiktok: str = Field(default="", description="Polished TikTok description")
    video_description_for_instagram: str = Field(default="", description="Polished Instagram description")


class Stage4Response(BaseModel):
    """Output schema enforced on Stage 4 final consensus reviewer."""
    decisions: List[Stage4ReviewDecision] = Field(default_factory=list)


class StageExecutionMetadata(BaseModel):
    """Telemetry and timing for a single stage execution."""
    stage: int
    model: str
    status: str  # "success", "failed", "skipped"
    latency_seconds: float = 0.0
    candidate_count: int = 0
    error: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None


class ConsensusReport(BaseModel):
    """Comprehensive final report on the 4-stage consensus execution."""
    consensus_pipeline_version: str = "1.0.0"
    strict_consensus_mode: bool = True
    consensus_achieved: bool = False
    total_candidates_stage1: int = 0
    total_candidates_stage2: int = 0
    total_candidates_stage3: int = 0
    matched_groups_total: int = 0
    three_stage_consensus_count: int = 0
    final_approved_count: int = 0
    stage_telemetry: Dict[str, StageExecutionMetadata] = Field(default_factory=dict)
    rejection_log: List[Dict[str, Any]] = Field(default_factory=list)
    consensus_groups: List[MatchedCandidateGroup] = Field(default_factory=list)
