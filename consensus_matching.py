"""Reusable matching module for multi-stage candidate clips.

Matches candidates across Stage 1, Stage 2, and Stage 3 using:
  1. Temporal Intersection-over-Union (IoU)
  2. Semantic similarity across summaries and hooks (as secondary signal)

Avoids merging temporally separate moments simply because they have similar topics.
Prevents duplicate candidates from the same stage in a single cluster.
Tracks which stages supported each candidate group.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from consensus_models import CandidateClip, MatchedCandidateGroup

# Common English and general stop words to ignore in semantic similarity
STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "is", "it", "this", "that", "these", "those", "we", "you", "they", "i",
    "he", "she", "what", "how", "why", "when", "where", "who", "be", "have", "do"
}


def calculate_temporal_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """Calculate temporal Intersection over Union (IoU) between two time spans.

    Formula:
        overlap = max(0, min(end1, end2) - max(start1, start2))
        union = max(end1, end2) - min(start1, start2)
        IoU = overlap / union
    """
    s1, e1 = float(start1), float(end1)
    s2, e2 = float(start2), float(end2)

    overlap = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)

    if union <= 0.0 or overlap <= 0.0:
        return 0.0

    return round(overlap / union, 4)


def _tokenize(text: str) -> Set[str]:
    """Tokenize text into lowercase alphanumeric words, filtering stopwords."""
    if not text:
        return set()
    words = re.findall(r"\b\w{2,}\b", text.lower())
    return {w for w in words if w not in STOP_WORDS}


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate token-based Jaccard similarity between two text snippets.

    Returns float in range [0.0, 1.0].
    """
    t1 = _tokenize(text1)
    t2 = _tokenize(text2)

    if not t1 or not t2:
        return 0.0

    intersection = len(t1.intersection(t2))
    union = len(t1.union(t2))

    if union == 0:
        return 0.0

    return round(intersection / union, 4)


def are_candidates_matching(
    c1: CandidateClip,
    c2: CandidateClip,
    iou_threshold: float = 0.5,
    semantic_threshold: float = 0.35,
) -> Tuple[bool, float, float]:
    """Determine whether two candidate clips represent the same video moment.

    Returns:
        (is_match, iou_score, semantic_score)
    """
    iou = calculate_temporal_iou(c1.start, c1.end, c2.start, c2.end)

    # Hard requirement: must have positive temporal overlap.
    # We never merge separate moments that have 0 temporal overlap regardless of topic.
    if iou <= 0.0:
        return False, 0.0, 0.0

    text1 = f"{c1.summary} {c1.hook} {c1.why}"
    text2 = f"{c2.summary} {c2.hook} {c2.why}"
    sim = calculate_semantic_similarity(text1, text2)

    # Condition A: Strong temporal overlap (>= iou_threshold)
    if iou >= iou_threshold:
        return True, iou, sim

    # Condition B: Moderate temporal overlap (>= 0.30) with supportive semantic similarity
    if iou >= 0.30 and sim >= semantic_threshold:
        return True, iou, sim

    return False, iou, sim


def deduplicate_stage_candidates(candidates: List[CandidateClip], overlap_threshold: float = 0.5) -> List[CandidateClip]:
    """Remove duplicate/heavily overlapping candidates produced within the same stage.

    Favors the candidate with the higher virality_score.
    """
    if not candidates:
        return []

    # Sort descending by virality_score
    sorted_cands = sorted(candidates, key=lambda c: (c.virality_score, c.confidence_score), reverse=True)
    kept: List[CandidateClip] = []

    for cand in sorted_cands:
        is_dup = False
        for existing in kept:
            iou = calculate_temporal_iou(cand.start, cand.end, existing.start, existing.end)
            if iou >= overlap_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)

    # Restore chronological order
    return sorted(kept, key=lambda c: c.start)


def cluster_multi_stage_candidates(
    candidates_by_stage: Dict[int, List[CandidateClip]],
    iou_threshold: float = 0.5,
    semantic_threshold: float = 0.35,
) -> List[MatchedCandidateGroup]:
    """Cluster candidate clips across stages 1, 2, and 3 into unified candidate groups.

    Args:
        candidates_by_stage: Map of stage number (1, 2, 3) to list of CandidateClip.
        iou_threshold: Overlap threshold for IoU matching.
        semantic_threshold: Secondary text similarity threshold.

    Returns:
        List of MatchedCandidateGroup sorted chronologically.
    """
    # 1. Deduplicate within each stage first
    clean_by_stage: Dict[int, List[CandidateClip]] = {
        stage: deduplicate_stage_candidates(cands)
        for stage, cands in candidates_by_stage.items()
    }

    # 2. Iteratively cluster into groups
    # Each group is a dict: {"members": {stage_num: CandidateClip}, "ious": []}
    groups: List[Dict[str, Any]] = []

    # Process stages in order 1 -> 2 -> 3
    for stage_num in (1, 2, 3):
        stage_cands = clean_by_stage.get(stage_num, [])
        for cand in stage_cands:
            best_group = None
            best_iou = 0.0

            for grp in groups:
                # Do not put two candidates from the same stage in the same group
                if stage_num in grp["members"]:
                    continue

                # Check match against all existing members in this group
                matches = []
                for existing_stage, existing_cand in grp["members"].items():
                    is_match, iou, sim = are_candidates_matching(
                        cand, existing_cand, iou_threshold, semantic_threshold
                    )
                    if is_match:
                        matches.append(iou)

                if matches:
                    avg_match_iou = sum(matches) / len(matches)
                    if avg_match_iou > best_iou:
                        best_iou = avg_match_iou
                        best_group = grp

            if best_group is not None:
                best_group["members"][stage_num] = cand
                best_group["ious"].append(best_iou)
            else:
                # Start a new candidate group
                groups.append({
                    "members": {stage_num: cand},
                    "ious": [1.0],
                })

    # 3. Construct MatchedCandidateGroup objects
    result_groups: List[MatchedCandidateGroup] = []
    group_idx = 1

    for grp in groups:
        members: Dict[int, CandidateClip] = grp["members"]
        stages_supported = sorted(list(members.keys()))
        cands_list = list(members.values())

        # Average start & end across supporting stages
        avg_start = round(sum(c.start for c in cands_list) / len(cands_list), 3)
        avg_end = round(sum(c.end for c in cands_list) / len(cands_list), 3)
        avg_score = round(sum(c.virality_score for c in cands_list) / len(cands_list), 1)
        avg_conf = round(sum(c.confidence_score for c in cands_list) / len(cands_list), 2)
        avg_iou = round(sum(grp["ious"]) / len(grp["ious"]), 3)

        # Pick best candidate's summary and hook
        best_cand = max(cands_list, key=lambda c: c.virality_score)
        combined_summary = best_cand.summary
        combined_hook = best_cand.viral_hook_text or best_cand.hook

        candidates_map = {f"stage_{s}": members[s] for s in stages_supported}
        is_three_way = (set(stages_supported) == {1, 2, 3})

        matched_group = MatchedCandidateGroup(
            group_id=f"consensus_group_{group_idx:03d}",
            stages_supported=stages_supported,
            is_three_stage_consensus=is_three_way,
            consensus_iou=avg_iou,
            avg_start=avg_start,
            avg_end=avg_end,
            avg_virality_score=avg_score,
            avg_confidence=avg_conf,
            candidates_by_stage=candidates_map,
            combined_summary=combined_summary,
            combined_hook=combined_hook,
        )
        result_groups.append(matched_group)
        group_idx += 1

    # Sort groups chronologically by start timestamp
    return sorted(result_groups, key=lambda g: g.avg_start)
