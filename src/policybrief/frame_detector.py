"""
Two-stage policy-frame detector.

Stage 1: keyword candidate selection (deterministic).
Stage 2: LLM assessment with structured output (per-frame).

Policy-mix detection when >=2 frames are present.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from .models import (
    Evidence,
    FrameAssessment,
    FrameDecision,
    FrameDetectionOutput,
    PageText,
)

logger = logging.getLogger(__name__)

# ── System prompt (moved from llm_client) ─────────────────────────────────

_FRAME_SYSTEM_PROMPT = """You are an expert policy-frame analyst. You will be given:
1. A frame definition (a particular type of policy instrument).
2. Text spans from a policy document.

Your task: Decide whether the frame is PRESENT, ABSENT, or has INSUFFICIENT_EVIDENCE in this document.

Rules:
- PRESENT: The document explicitly advocates, describes, or proposes the policy instrument with concrete details.
- ABSENT: The document does not mention or engage with this instrument type at all.
- INSUFFICIENT_EVIDENCE: The document mentions related concepts but lacks clear evidence that the specific instrument type is being discussed.
- Evidence quotes MUST be verbatim text from the provided spans.
- Evidence quotes must be 10-500 characters.
- Confidence: 0.0 to 1.0 (0.9+ only when evidence is unambiguous).
- Provide a concise rationale explaining your decision.
"""


class FrameDetector:
    """Detect policy frames in document text via keyword pre-filter + LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        frames_config: List[Dict[str, Any]],
        min_confidence: float = 0.7,
        max_spans_per_frame: int = 5,
        context_window: int = 500,
        min_evidence_quotes: int = 1,
        max_evidence_quotes: int = 3,
    ):
        self.llm = llm_client
        self.frames = frames_config
        self.min_confidence = min_confidence
        self.max_spans = max_spans_per_frame
        self.ctx_window = context_window
        self.min_evidence = min_evidence_quotes
        self.max_evidence = max_evidence_quotes

        # Pre-compile keyword patterns per frame
        self._patterns: Dict[str, List[re.Pattern]] = {}
        for frame in self.frames:
            cues = frame.get("inclusion_cues", [])
            self._patterns[frame["id"]] = [
                re.compile(re.escape(c), re.IGNORECASE) for c in cues
            ]

    # ── Public API ────────────────────────────────────────────────────

    def detect_frames(
        self,
        pages: List[PageText],
        excluded_pages: Optional[set] = None,
    ) -> List[FrameAssessment]:
        """Run frame detection on a document's pages."""
        excluded = excluded_pages or set()
        full_text_by_page = {
            p.page_num: p.text for p in pages if p.page_num not in excluded
        }
        full_text = "\n\n".join(full_text_by_page.values())

        assessments: List[FrameAssessment] = []
        for frame in self.frames:
            try:
                assessment = self._assess_frame(frame, full_text, full_text_by_page)
                assessments.append(assessment)
            except Exception as exc:
                logger.warning(f"Frame {frame['id']} failed: {exc}")
                assessments.append(self._absent_assessment(frame, str(exc)))

        return assessments

    def detect_policy_mix(self, assessments: List[FrameAssessment]) -> bool:
        """Return True if 2+ distinct frames are present."""
        present = [
            a for a in assessments
            if a.decision == FrameDecision.PRESENT
            and a.confidence >= self.min_confidence
        ]
        return len(present) >= 2

    # ── Stage 1: Keyword candidate selection ──────────────────────────

    def _find_keyword_spans(
        self,
        frame_id: str,
        text: str,
        text_by_page: Dict[int, str],
    ) -> List[Dict[str, Any]]:
        """Find context windows around keyword matches."""
        patterns = self._patterns.get(frame_id, [])
        spans: List[Dict[str, Any]] = []
        seen_positions: set = set()

        for pat in patterns:
            for match in pat.finditer(text):
                start = max(0, match.start() - self.ctx_window)
                end = min(len(text), match.end() + self.ctx_window)
                # Deduplicate overlapping spans
                pos_key = (start // 200, end // 200)
                if pos_key in seen_positions:
                    continue
                seen_positions.add(pos_key)

                snippet = text[start:end].strip()
                page = self._locate_page(match.start(), text_by_page)
                spans.append({
                    "text": snippet,
                    "page": page,
                    "keyword": match.group(),
                })

                if len(spans) >= self.max_spans:
                    return spans
        return spans

    @staticmethod
    def _locate_page(char_offset: int, text_by_page: Dict[int, str]) -> int:
        """Map a character offset in the concatenated text to a page number."""
        running = 0
        for page_num, page_text in sorted(text_by_page.items()):
            running += len(page_text) + 2  # 2 for \n\n separator
            if char_offset < running:
                return page_num
        return max(text_by_page.keys()) if text_by_page else 1

    # ── Stage 2: LLM assessment ──────────────────────────────────────

    def _assess_frame(
        self,
        frame: Dict[str, Any],
        full_text: str,
        text_by_page: Dict[int, str],
    ) -> FrameAssessment:
        """Assess a single frame via keyword selection + LLM."""
        frame_id = frame["id"]
        frame_label = frame["label"]
        spans = self._find_keyword_spans(frame_id, full_text, text_by_page)

        if not spans:
            return self._absent_assessment(frame, "No keyword matches found")

        # Build LLM prompt
        user_content = self._build_user_prompt(frame, spans)
        messages = [
            {"role": "system", "content": _FRAME_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        result: FrameDetectionOutput = self.llm.structured_completion(
            messages, FrameDetectionOutput
        )

        # Validate evidence quotes against source text
        validated_evidence = self._validate_quotes(result.evidence, full_text)
        if not validated_evidence and result.decision == FrameDecision.PRESENT:
            if result.confidence < self.min_confidence:
                return self._absent_assessment(
                    frame, "Evidence quotes not found in source"
                )

        # Apply confidence threshold
        decision = result.decision
        if decision == FrameDecision.PRESENT:
            if result.confidence < self.min_confidence:
                decision = FrameDecision.INSUFFICIENT_EVIDENCE
            elif len(validated_evidence) < self.min_evidence:
                decision = FrameDecision.INSUFFICIENT_EVIDENCE

        return FrameAssessment(
            frame_id=frame_id,
            frame_label=frame_label,
            decision=decision,
            confidence=result.confidence,
            evidence=validated_evidence[: self.max_evidence],
            counterevidence=[],
            rationale=result.rationale,
        )

    def _build_user_prompt(
        self,
        frame: Dict[str, Any],
        spans: List[Dict[str, Any]],
    ) -> str:
        parts = [
            f"## Frame: {frame['label']}",
            f"**Definition:** {frame['short_definition']}",
        ]
        if frame.get("analytical_notes"):
            parts.append(f"**Analytical notes:** {frame['analytical_notes']}")
        if frame.get("false_positive_notes"):
            parts.append(f"**False-positive guidance:** {frame['false_positive_notes']}")

        parts.append("\n## Document Spans\n")
        for i, span in enumerate(spans, 1):
            parts.append(f"### Span {i} (page {span['page']}, keyword: '{span['keyword']}')")
            parts.append(span["text"])
            parts.append("")

        parts.append(
            f"\nReturn JSON with frame_id='{frame['id']}'. "
            "Provide evidence quotes that are verbatim from the spans above."
        )
        return "\n".join(parts)

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
                # Try fuzzy: first 40 chars
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
