"""
Policy-instrument frame detection with two-stage strategy.

Stage 1: Keyword-based candidate span selection (instrument-aware cues)
Stage 2: LLM-based assessment of selected spans

The typology follows smart regulation / regulatory pluralism
(Gunningham & Grabosky 1998; Howlett 2011) with five instrument
categories: command-and-control, economic instruments, self-regulation,
voluntarism, and information strategies.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import LLMClient
from .models import FrameAssessment, FrameDetectionInput, PageText, Evidence


logger = logging.getLogger(__name__)

# Minimum number of frames marked "present" to consider policy-mix
# annotation.  The document must explicitly discuss complementarity or
# instrument combination; mere co-occurrence is annotated but at lower
# confidence.
_POLICY_MIX_MIN_FRAMES = 2

# Regex cues that indicate explicit discussion of policy mixes /
# instrument complementarity.  Used for the policy_mix_present flag.
_POLICY_MIX_CUES: List[re.Pattern] = [
    re.compile(r"policy\s+mix", re.IGNORECASE),
    re.compile(r"instrument\s+mix", re.IGNORECASE),
    re.compile(r"mix\s+of\s+instruments", re.IGNORECASE),
    re.compile(r"combin(?:ation|ing|ed)\s+(?:of\s+)?(?:instruments?|policies|measures|tools)", re.IGNORECASE),
    re.compile(r"complement(?:ary|arity)", re.IGNORECASE),
    re.compile(r"regulat(?:ory|ion)\s+pluralism", re.IGNORECASE),
    re.compile(r"smart\s+regulation", re.IGNORECASE),
    re.compile(r"hybrid\s+(?:governance|approach|instrument)", re.IGNORECASE),
    re.compile(r"integrated\s+(?:policy|governance)\s+(?:approach|framework)", re.IGNORECASE),
    re.compile(r"multi-?instrument", re.IGNORECASE),
]


class FrameDetector:
    """Two-stage policy-instrument frame detection system.

    Stage 1 selects candidate text spans via keyword matching against
    instrument-specific cues defined in ``frames.yaml``.  Stage 2 sends
    the top candidates to the LLM for structured assessment.

    After all frames are assessed, :meth:`detect_policy_mix` checks
    whether the document explicitly discusses instrument complementarity
    or policy mixes.
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        frames_config: List[Dict[str, Any]],
        min_confidence: float = 0.7,
        max_spans_per_frame: int = 5,
        context_window: int = 500,
        min_evidence_quotes: int = 1,
        max_evidence_quotes: int = 3
    ):
        """
        Initialize frame detector.
        
        Args:
            llm_client: LLM client for frame assessment
            frames_config: Frame definitions from config
            min_confidence: Minimum confidence for 'present' classification
            max_spans_per_frame: Max text spans to send to LLM per frame
            context_window: Context chars around keyword matches
            min_evidence_quotes: Min evidence required for 'present'
            max_evidence_quotes: Max evidence to extract per frame
        """
        self.llm_client = llm_client
        self.frames = {f["id"]: f for f in frames_config}
        self.min_confidence = min_confidence
        self.max_spans_per_frame = max_spans_per_frame
        self.context_window = context_window
        self.min_evidence_quotes = min_evidence_quotes
        self.max_evidence_quotes = max_evidence_quotes
        
        logger.info(f"Initialized frame detector with {len(self.frames)} frames")
    
    def detect_frames(self, pages: List[PageText]) -> List[FrameAssessment]:
        """
        Detect all configured frames in document pages.
        
        Args:
            pages: Document pages with text
            
        Returns:
            List of frame assessments
        """
        logger.info(f"Detecting frames in document with {len(pages)} pages")
        
        assessments = []
        
        for frame_id, frame_config in self.frames.items():
            try:
                assessment = self._detect_single_frame(pages, frame_id, frame_config)
                assessments.append(assessment)
                
                logger.debug(f"Frame {frame_id}: {assessment.decision} (confidence: {assessment.confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Failed to detect frame {frame_id}: {e}")
                # Create fallback assessment
                assessment = FrameAssessment(
                    frame_id=frame_id,
                    frame_label=frame_config.get("label", frame_id),
                    decision="insufficient_evidence",
                    confidence=0.0,
                    evidence=[],
                    counterevidence=[],
                    rationale=f"Detection failed due to error: {str(e)}"
                )
                assessments.append(assessment)
        
        return assessments
    
    def _detect_single_frame(
        self, 
        pages: List[PageText], 
        frame_id: str, 
        frame_config: Dict[str, Any]
    ) -> FrameAssessment:
        """Detect a single theoretical frame."""
        logger.debug(f"Detecting frame: {frame_id}")
        
        # Stage 1: Candidate span selection
        candidate_spans = self._select_candidate_spans(pages, frame_config)
        
        if not candidate_spans:
            # No relevant spans found
            return FrameAssessment(
                frame_id=frame_id,
                frame_label=frame_config.get("label", frame_id),
                decision="absent",
                confidence=0.9,  # High confidence in absence
                evidence=[],
                counterevidence=[],
                rationale="No relevant text spans found matching frame indicators"
            )
        
        # Stage 2: LLM assessment
        frame_input = FrameDetectionInput(
            frame_id=frame_id,
            frame_definition=self._format_frame_definition(frame_config),
            text_spans=candidate_spans[:self.max_spans_per_frame]
        )
        
        llm_output = self.llm_client.detect_frame(
            frame_input,
            self._format_frame_definition(frame_config)
        )
        
        # Validate and process LLM output
        validated_assessment = self._validate_assessment(
            llm_output, 
            frame_config,
            pages
        )
        
        return validated_assessment
    
    def _select_candidate_spans(
        self, 
        pages: List[PageText], 
        frame_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Stage 1: Select candidate text spans using keyword matching."""
        inclusion_cues = frame_config.get("inclusion_cues", [])
        exclusion_cues = frame_config.get("exclusion_cues", [])
        must_have = frame_config.get("must_have", [])
        
        if not inclusion_cues:
            logger.warning(f"No inclusion cues defined for frame")
            return []
        
        candidate_spans = []
        
        for page in pages:
            page_text = page.text
            if not page_text:
                continue
            
            # Find inclusion cue matches
            matches = []
            for cue in inclusion_cues:
                pattern = re.escape(cue.lower())
                for match in re.finditer(pattern, page_text.lower()):
                    matches.append({
                        "start": match.start(),
                        "end": match.end(),
                        "cue": cue,
                        "score": 1.0
                    })
            
            if not matches:
                continue
            
            # Process matches into spans
            for match in matches:
                # Extract context window around match
                start_pos = max(0, match["start"] - self.context_window)
                end_pos = min(len(page_text), match["end"] + self.context_window)
                
                span_text = page_text[start_pos:end_pos]
                
                # Check exclusion cues
                has_exclusion = any(
                    exc.lower() in span_text.lower() 
                    for exc in exclusion_cues
                )
                
                if has_exclusion:
                    match["score"] *= 0.3  # Reduce score but don't eliminate
                
                # Check must-have requirements
                if must_have:
                    for must_group in must_have:
                        if isinstance(must_group, list):
                            has_must = any(
                                req.lower() in span_text.lower() 
                                for req in must_group
                            )
                        else:
                            has_must = must_group.lower() in span_text.lower()
                        
                        if has_must:
                            match["score"] *= 2.0  # Boost score
                
                # Extract section heading if available
                section_heading = self._extract_section_heading(
                    page_text, match["start"]
                )
                
                candidate_spans.append({
                    "page": page.page_num,
                    "text": span_text.strip(),
                    "section_heading": section_heading,
                    "score": match["score"],
                    "matched_cue": match["cue"],
                    "start_char": start_pos,
                    "end_char": end_pos
                })
        
        # Sort by score and return top candidates
        candidate_spans.sort(key=lambda x: x["score"], reverse=True)
        
        logger.debug(f"Selected {len(candidate_spans)} candidate spans")
        return candidate_spans
    
    def _extract_section_heading(self, text: str, position: int) -> Optional[str]:
        """Extract section heading before the given position."""
        # Look backwards for potential headings
        lines_before = text[:position].split('\n')
        
        for line in reversed(lines_before[-10:]):  # Check last 10 lines
            line = line.strip()
            
            # Simple heading detection
            if (line and 
                len(line) < 100 and
                len(line.split()) >= 2 and
                not line.endswith('.') and
                (line.isupper() or line.istitle() or 
                 re.match(r'^\d+\.?\s+[A-Z]', line))):
                return line
        
        return None
    
    def _format_frame_definition(self, frame_config: Dict[str, Any]) -> str:
        """Format frame definition for LLM consumption.

        Includes analytical notes, positive examples, and false-positive
        guidance when available in the config.
        """
        parts: List[str] = []
        parts.append(f"Frame: {frame_config.get('label', 'Unknown')}")
        parts.append(f"Definition: {frame_config.get('short_definition', 'No definition')}")

        if frame_config.get("analytical_notes"):
            parts.append(f"Analytical guidance: {frame_config['analytical_notes']}")

        if frame_config.get("inclusion_cues"):
            parts.append(f"Keywords/Indicators: {', '.join(frame_config['inclusion_cues'])}")

        if frame_config.get("exclusion_cues"):
            parts.append(f"Exclusions: {', '.join(frame_config['exclusion_cues'])}")

        if frame_config.get("must_have"):
            parts.append(f"Required elements: {frame_config['must_have']}")

        if frame_config.get("positive_examples"):
            examples = "\n  - ".join(frame_config["positive_examples"])
            parts.append(f"Positive examples:\n  - {examples}")

        if frame_config.get("false_positive_notes"):
            parts.append(f"False-positive guidance: {frame_config['false_positive_notes']}")

        return "\n".join(parts)
    
    def _validate_assessment(
        self,
        llm_output: Any,
        frame_config: Dict[str, Any],
        pages: List[PageText]
    ) -> FrameAssessment:
        """Validate and process LLM assessment output."""
        # Apply confidence threshold
        if (llm_output.decision == "present" and 
            llm_output.confidence < self.min_confidence):
            llm_output.decision = "insufficient_evidence"
            llm_output.rationale += f" (Confidence {llm_output.confidence:.2f} below threshold {self.min_confidence})"
        
        # Validate evidence quotes
        validated_evidence = []
        for evidence in llm_output.evidence:
            if self._validate_quote(evidence, pages):
                validated_evidence.append(evidence)
            else:
                logger.warning(f"Invalid evidence quote for frame {llm_output.frame_id}: {evidence.quote[:50]}...")
        
        # Check minimum evidence requirement
        if (llm_output.decision == "present" and 
            len(validated_evidence) < self.min_evidence_quotes):
            llm_output.decision = "insufficient_evidence"
            llm_output.rationale += f" (Insufficient valid evidence quotes: {len(validated_evidence)} < {self.min_evidence_quotes})"
        
        # Limit evidence quotes
        if len(validated_evidence) > self.max_evidence_quotes:
            validated_evidence = validated_evidence[:self.max_evidence_quotes]
        
        return FrameAssessment(
            frame_id=llm_output.frame_id,
            frame_label=frame_config.get("label", llm_output.frame_id),
            decision=llm_output.decision,
            confidence=llm_output.confidence,
            evidence=validated_evidence,
            counterevidence=[],  # No counterevidence in simplified model
            rationale=llm_output.rationale
        )
    
    def _validate_quote(self, evidence: Evidence, pages: List[PageText]) -> bool:
        """Validate that quote exists verbatim in the document pages."""
        try:
            # Find the page
            target_page = None
            for page in pages:
                if page.page_num == evidence.page:
                    target_page = page
                    break
            
            if not target_page:
                logger.warning(f"Page {evidence.page} not found for quote validation")
                return False
            
            # Check if quote exists in page text
            quote_clean = evidence.quote.strip()
            page_text = target_page.text
            
            # Try exact match first
            if quote_clean in page_text:
                return True
            
            # Try normalized whitespace match
            quote_normalized = re.sub(r'\s+', ' ', quote_clean)
            page_normalized = re.sub(r'\s+', ' ', page_text)
            
            if quote_normalized in page_normalized:
                return True
            
            # Try case-insensitive match
            if quote_clean.lower() in page_text.lower():
                return True
            
            logger.warning(f"Quote not found in page {evidence.page}: '{quote_clean[:50]}...'")
            return False
            
        except Exception as e:
            logger.error(f"Quote validation failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Policy-mix / instrument-combination annotation
    # ------------------------------------------------------------------

    def detect_policy_mix(
        self,
        pages: List[PageText],
        assessments: List[FrameAssessment],
    ) -> bool:
        """Detect whether the document explicitly discusses policy mixes.

        Returns ``True`` when:
          - at least ``_POLICY_MIX_MIN_FRAMES`` frames are present AND
          - the document text contains explicit policy-mix / complementarity
            language matching ``_POLICY_MIX_CUES``.

        This is a conservative heuristic: mere co-occurrence of two
        instrument categories is necessary but not sufficient.
        """
        present_count = sum(
            1 for a in assessments if a.decision == "present"
        )
        if present_count < _POLICY_MIX_MIN_FRAMES:
            return False

        full_text = " ".join(p.text for p in pages if p.text)
        for pattern in _POLICY_MIX_CUES:
            if pattern.search(full_text):
                logger.info(
                    "Policy-mix language detected with %d instrument "
                    "categories present",
                    present_count,
                )
                return True

        logger.debug(
            "%d instrument categories present but no explicit "
            "policy-mix language found",
            present_count,
        )
        return False