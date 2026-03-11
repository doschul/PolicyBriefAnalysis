"""
Theoretical frame detection with two-stage strategy to optimize token usage.

Stage 1: Keyword-based candidate span selection
Stage 2: LLM-based assessment of selected spans
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import LLMClient
from .models import FrameAssessment, FrameDetectionInput, PageText, Evidence


logger = logging.getLogger(__name__)


class FrameDetector:
    """Two-stage theoretical frame detection system."""
    
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
        """Format frame definition for LLM consumption."""
        definition = f"Frame: {frame_config.get('label', 'Unknown')}\n"
        definition += f"Definition: {frame_config.get('short_definition', 'No definition')}\n"
        
        if frame_config.get('inclusion_cues'):
            definition += f"Keywords/Indicators: {', '.join(frame_config['inclusion_cues'])}\n"
        
        if frame_config.get('exclusion_cues'):
            definition += f"Exclusions: {', '.join(frame_config['exclusion_cues'])}\n"
        
        if frame_config.get('must_have'):
            definition += f"Required elements: {frame_config['must_have']}"
        
        return definition
    
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