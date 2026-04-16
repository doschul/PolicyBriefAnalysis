"""
Broad-content policy-frame detector.

Sends broad document content to the LLM for frame assessment, replacing
the previous keyword-prefilter + per-frame LLM call approach.
Uses frame definitions from frames.yaml as conceptual guidance.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import LLMClient
from .models import (
    Evidence,
    FrameAssessment,
    FrameDecision,
    FrameExtractionResponse,
    PageText,
    SingleFrameResult,
)
from .recommendation_extractor import DocumentContent

logger = logging.getLogger(__name__)

# ── Prompt template ───────────────────────────────────────────────────────

FRAME_SYSTEM_PROMPT = """You are an expert policy-frame analyst. You will be given:
1. Definitions of policy-instrument frames (categories from the smart-regulation literature).
2. The full text of a policy document (or a large section).

Your task: For EACH frame, decide whether it is PRESENT, ABSENT, or has INSUFFICIENT_EVIDENCE in this document.

Classification criteria:
- PRESENT: The document explicitly advocates, describes, or proposes the policy instrument with concrete details. The frame's mechanism must be clearly discussed — not merely mentioned in passing.
- ABSENT: The document does not meaningfully engage with this instrument type.
- INSUFFICIENT_EVIDENCE: The document mentions related concepts but lacks clear evidence that the specific instrument type is being discussed as a policy mechanism.

Critical distinctions:
- self_regulation vs voluntarism: Self-regulation involves COLLECTIVE industry governance (sector-wide standards, certification schemes, codes of practice). Voluntarism involves UNILATERAL individual-actor commitments (company pledges, CSR). The key test: collective sector norm vs single-actor commitment.
- information_strategies vs generic citation: Information strategies must be presented as a GOVERNANCE MECHANISM (monitoring systems, transparency requirements, traceability tools used to influence behaviour) — not merely citing scientific studies or presenting data as background.
- command_and_control requires LEGAL COMPULSION: binding rules, sanctions, permits, compliance obligations. General mentions of "governance" or "policy" are not sufficient.
- economic_instruments require a specific FINANCIAL MECHANISM: PES, subsidies, credits, taxes. General statements about economic value are not economic instruments.

Rules:
- Assess ALL frames independently. Multiple frames CAN co-occur in one document.
- Evidence quotes MUST be verbatim text from the provided document.
- Evidence quotes must be 10-500 characters.
- Confidence: 0.0 to 1.0. Use 0.9+ only when evidence is unambiguous.
- Provide a concise rationale for each frame explaining your decision.
- If unsure, return INSUFFICIENT_EVIDENCE rather than PRESENT.
- Return a result for EVERY frame listed, even if absent."""


class FrameDetector:
    """Detect policy frames in document text via broad-content LLM assessment."""

    def __init__(
        self,
        llm_client: LLMClient,
        frames_config: List[Dict[str, Any]],
        min_confidence: float = 0.7,
        min_evidence_quotes: int = 1,
        max_evidence_quotes: int = 3,
        max_chars_per_chunk: int = 100000,
        # Legacy params accepted but ignored
        max_spans_per_frame: int = 5,
        context_window: int = 500,
    ):
        self.llm = llm_client
        self.frames = frames_config
        self.min_confidence = min_confidence
        self.min_evidence = min_evidence_quotes
        self.max_evidence = max_evidence_quotes
        self.max_chars_per_chunk = max_chars_per_chunk

    # ── Public API ────────────────────────────────────────────────────

    def detect_frames(
        self,
        pages: List[PageText],
        excluded_pages: Optional[set] = None,
    ) -> List[FrameAssessment]:
        """Run frame detection on a document's pages."""
        excluded = excluded_pages or set()
        active_pages = [p for p in pages if p.page_num not in excluded]
        if not active_pages:
            return [self._absent_assessment(f, "No pages available") for f in self.frames]

        content = DocumentContent(active_pages)
        full_text = "\n".join(p.text for p in active_pages)

        # Run LLM over chunks and aggregate
        chunks = content.page_chunks(max_chars=self.max_chars_per_chunk)
        all_frame_results: Dict[str, List[SingleFrameResult]] = {
            f["id"]: [] for f in self.frames
        }

        for chunk_idx, (chunk_pages, chunk_text) in enumerate(chunks):
            try:
                results = self._assess_chunk(chunk_text)
                for fr in results:
                    if fr.frame_id in all_frame_results:
                        all_frame_results[fr.frame_id].append(fr)
            except Exception as exc:
                logger.warning(f"Frame chunk {chunk_idx + 1} failed: {exc}")

        # Aggregate results per frame
        assessments: List[FrameAssessment] = []
        for frame in self.frames:
            frame_id = frame["id"]
            chunk_results = all_frame_results.get(frame_id, [])
            assessment = self._aggregate_frame(frame, chunk_results, full_text)
            assessments.append(assessment)

        return assessments

    def detect_policy_mix(self, assessments: List[FrameAssessment]) -> bool:
        """Return True if 2+ distinct frames are present."""
        present = [
            a for a in assessments
            if a.decision == FrameDecision.PRESENT
            and a.confidence >= self.min_confidence
        ]
        return len(present) >= 2

    # ── LLM assessment ────────────────────────────────────────────────

    def _assess_chunk(self, chunk_text: str) -> List[SingleFrameResult]:
        """Assess all frames for a single chunk via one LLM call."""
        user_content = self._build_user_prompt(chunk_text)
        messages = [
            {"role": "system", "content": FRAME_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        result: FrameExtractionResponse = self.llm.structured_completion(
            messages, FrameExtractionResponse
        )
        return result.frames

    def _build_user_prompt(self, chunk_text: str) -> str:
        """Build the user prompt with frame definitions and document text."""
        parts = ["## Frame Definitions\n"]
        for frame in self.frames:
            parts.append(f"### {frame['id']}: {frame['label']}")
            parts.append(f"**Definition:** {frame['short_definition']}")
            if frame.get("analytical_notes"):
                parts.append(f"**Analytical notes:** {frame['analytical_notes']}")
            if frame.get("false_positive_notes"):
                parts.append(f"**False-positive guidance:** {frame['false_positive_notes']}")
            parts.append("")

        parts.append("## Document Text\n")
        parts.append(chunk_text)
        parts.append(
            "\n\nReturn a JSON object with a 'frames' array containing one result "
            "per frame defined above. Each must have: frame_id, decision, confidence, "
            "evidence (list of {page, quote}), rationale."
        )
        return "\n".join(parts)

    # ── Aggregation across chunks ─────────────────────────────────────

    def _aggregate_frame(
        self,
        frame: Dict[str, Any],
        chunk_results: List[SingleFrameResult],
        full_text: str,
    ) -> FrameAssessment:
        """Aggregate multiple chunk results into a single FrameAssessment."""
        frame_id = frame["id"]
        frame_label = frame["label"]

        if not chunk_results:
            return self._absent_assessment(frame, "No LLM results")

        # Take the strongest result
        best = max(chunk_results, key=lambda r: r.confidence)
        decision = best.decision
        confidence = best.confidence

        # Merge and validate all evidence across chunks
        all_evidence: List[Evidence] = []
        for cr in chunk_results:
            all_evidence.extend(cr.evidence)
        validated_evidence = self._validate_quotes(all_evidence, full_text)

        # If present but no validated evidence, downgrade
        if decision == FrameDecision.PRESENT:
            if confidence < self.min_confidence:
                decision = FrameDecision.INSUFFICIENT_EVIDENCE
            elif len(validated_evidence) < self.min_evidence:
                decision = FrameDecision.INSUFFICIENT_EVIDENCE

        # Use best rationale
        rationale = best.rationale

        return FrameAssessment(
            frame_id=frame_id,
            frame_label=frame_label,
            decision=decision,
            confidence=confidence,
            evidence=validated_evidence[: self.max_evidence],
            counterevidence=[],
            rationale=rationale,
        )

    # ── Quote validation ──────────────────────────────────────────────

    def _validate_quotes(
        self,
        evidence: List[Evidence],
        source_text: str,
    ) -> List[Evidence]:
        """Keep only evidence whose quote appears in the source text."""
        normalised = re.sub(r"\s+", " ", source_text.lower())
        validated: List[Evidence] = []
        for ev in evidence:
            q = re.sub(r"\s+", " ", ev.quote.lower())
            if q in normalised:
                validated.append(ev)
            else:
                prefix = q[:40]
                if prefix in normalised:
                    validated.append(ev)
                else:
                    logger.debug(f"Quote not found in source: {ev.quote[:60]}...")
        return validated

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _absent_assessment(frame: Dict[str, Any], reason: str) -> FrameAssessment:
        return FrameAssessment(
            frame_id=frame["id"],
            frame_label=frame["label"],
            decision=FrameDecision.ABSENT,
            confidence=0.0,
            evidence=[],
            counterevidence=[],
            rationale=reason,
        )
