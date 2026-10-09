"""Transparent scoring, ranking, and redundancy evaluation for consensus clips.

Formula:
  Final Score =
      w_agreement * Agreement_Score +
      w_discovery * Avg_Discovery_Score +
      w_reviewer  * Stage4_Reviewer_Score +
      w_hook      * Hook_Strength_Score -
      Redundancy_Penalty

Strict Consensus Rule:
  A high ranking score NEVER bypasses the 3-stage consensus requirement.
  Under strict mode, clips missing 3-stage support are disqualified before ranking.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import consensus_config
from consensus_models import MatchedCandidateGroup, Stage4ReviewDecision


def evaluate_hook_strength(hook_text: str) -> float:
    """Evaluate hook strength based on curiosity cues, brevity, and impactful keywords."""
    if not hook_text:
        return 50.0

    text = hook_text.strip().lower()
    score = 65.0

    # Length check: viral hooks are best between 3 and 10 words
    word_count = len(text.split())
    if 3 <= word_count <= 8:
        score += 15.0
    elif word_count > 14:
        score -= 10.0

    # Impact signals
    viral_triggers = ("why", "how", "stop", "never", "secret", "mistake", "always",
                      "don't", "truth", "every", "nobody", "hack", "proven", "best")
    if any(t in text for t in viral_triggers):
        score += 10.0

    if "?" in text or "!" in text:
        score += 5.0

    return min(100.0, max(0.0, score))


def compute_clip_score(
    group: MatchedCandidateGroup,
    decision: Stage4ReviewDecision,
    weights: Optional[Dict[str, float]] = None,
    redundancy_penalty: float = 0.0,
) -> Tuple[int, Dict[str, Any]]:
    """Compute transparent final ranking score for an approved consensus clip.

    Returns:
        (clamped_integer_score_0_100, score_breakdown_dict)
    """
    w = weights or consensus_config.get_ranking_weights()

    # 1. Agreement score (100 for 3 stages, 60 for 2 stages, 20 for 1 stage)
    num_stages = len(group.stages_supported)
    if num_stages >= 3:
        agreement_score = 100.0
    elif num_stages == 2:
        agreement_score = 60.0
    else:
        agreement_score = 20.0

    # 2. Average discovery score from Stage 1, 2, 3
    discovery_score = float(group.avg_virality_score)

    # 3. Stage 4 reviewer score
    reviewer_score = float(decision.reviewer_score)

    # 4. Hook strength score
    hook_score = evaluate_hook_strength(decision.viral_hook_text or group.combined_hook)

    # Weighted calculation
    w_agree = w.get("weight_agreement", 0.35)
    w_disc = w.get("weight_discovery", 0.25)
    w_rev = w.get("weight_reviewer", 0.25)
    w_hook = w.get("weight_hook", 0.15)

    weighted_total = (
        (w_agree * agreement_score) +
        (w_disc * discovery_score) +
        (w_rev * reviewer_score) +
        (w_hook * hook_score)
    )

    final_score = max(0.0, min(100.0, weighted_total - redundancy_penalty))
    int_score = int(round(final_score))

    breakdown = {
        "final_score": int_score,
        "agreement_score": round(agreement_score, 1),
        "discovery_score": round(discovery_score, 1),
        "reviewer_score": round(reviewer_score, 1),
        "hook_score": round(hook_score, 1),
        "redundancy_penalty": round(redundancy_penalty, 1),
        "stages_count": num_stages,
        "weights": {
            "w_agreement": w_agree,
            "w_discovery": w_disc,
            "w_reviewer": w_rev,
            "w_hook": w_hook,
        },
    }

    return int_score, breakdown


def rank_consensus_clips(
    approved_items: List[Tuple[MatchedCandidateGroup, Stage4ReviewDecision]],
    redundancy_overlap_threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    """Rank approved consensus clips by score, applying redundancy penalties if overlapping.

    Returns:
        List of ranked clip dictionaries ready for downstream metadata and rendering.
    """
    if not approved_items:
        return []

    weights = consensus_config.get_ranking_weights()
    redundancy_penalty_val = weights.get("redundancy_penalty", 15.0)

    # Initial scoring without redundancy
    initial_scored = []
    for grp, dec in approved_items:
        score, breakdown = compute_clip_score(grp, dec, weights=weights, redundancy_penalty=0.0)
        initial_scored.append({
            "group": grp,
            "decision": dec,
            "score": score,
            "breakdown": breakdown,
            "start": dec.refined_start,
            "end": dec.refined_end,
        })

    # Sort descending by preliminary score
    initial_scored.sort(key=lambda item: item["score"], reverse=True)

    # Second pass: apply redundancy penalty if temporal overlap with an already selected clip
    final_ranked: List[Dict[str, Any]] = []

    for item in initial_scored:
        penalty = 0.0
        # Check against previously kept higher-ranking clips
        for selected in final_ranked:
            s1, e1 = item["start"], item["end"]
            s2, e2 = selected["start"], selected["end"]
            overlap = max(0.0, min(e1, e2) - max(s1, s2))
            shorter = max(1e-5, min(e1 - s1, e2 - s2))
            if overlap / shorter >= redundancy_overlap_threshold:
                penalty = redundancy_penalty_val
                break

        if penalty > 0:
            new_score, new_breakdown = compute_clip_score(
                item["group"], item["decision"], weights=weights, redundancy_penalty=penalty
            )
            item["score"] = new_score
            item["breakdown"] = new_breakdown

        final_ranked.append(item)

    # Re-sort by final score
    final_ranked.sort(key=lambda item: item["score"], reverse=True)
    return final_ranked
