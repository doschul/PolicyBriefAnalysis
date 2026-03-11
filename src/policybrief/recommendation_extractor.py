"""
Policy recommendation extraction with structured validation.

Extracts actionable policy recommendations with evidence and classification.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from .models import PolicyRecommendation, PageText, Evidence


logger = logging.getLogger(__name__)


class RecommendationExtractor:
    """Extract and structure policy recommendations from documents."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        enums_config: Dict[str, List[str]],
        min_confidence: float = 0.6,
        max_recommendations: int = 10,
        recommendation_signals: Optional[List[str]] = None,
        target_sections: Optional[List[str]] = None
    ):
        """
        Initialize recommendation extractor.
        
        Args:
            llm_client: LLM client for extraction
            enums_config: Enum configurations from config
            min_confidence: Minimum confidence for extraction
            max_recommendations: Maximum recommendations to extract
            recommendation_signals: Keywords that signal recommendations
            target_sections: Section names likely to contain recommendations
        """
        self.llm_client = llm_client
        self.enums_config = enums_config
        self.min_confidence = min_confidence
        self.max_recommendations = max_recommendations
        
        self.recommendation_signals = recommendation_signals or [
            "recommend", "should", "must", "need to", "ought to",
            "propose", "suggest", "call for", "urge", "require",
            "encourage", "promote", "establish", "implement",
            "develop", "create", "ensure", "strengthen"
        ]
        
        self.target_sections = target_sections or [
            "recommendation", "conclusion", "summary", 
            "policy implication", "next step", "action",
            "way forward", "future work", "implementation"
        ]
        
        logger.info(f"Initialized recommendation extractor")
    
    def extract_recommendations(self, pages: List[PageText]) -> List[PolicyRecommendation]:
        """
        Extract policy recommendations from document pages.
        
        Args:
            pages: Document pages with text
            
        Returns:
            List of validated policy recommendations
        """
        logger.info(f"Extracting recommendations from {len(pages)} pages")
        
        # Pre-filter pages likely to contain recommendations
        relevant_pages = self._filter_relevant_pages(pages)
        
        if not relevant_pages:
            logger.info("No pages with recommendation signals found")
            return []
        
        logger.debug(f"Found {len(relevant_pages)} relevant pages")
        
        # Prepare page info for LLM
        page_info = []
        full_text = ""
        
        for page in relevant_pages:
            page_info.append({
                "page_num": page.page_num,
                "text": page.text,
                "char_count": page.char_count
            })
            full_text += f"\n--- PAGE {page.page_num} ---\n{page.text}"
        
        # Extract recommendations using LLM
        try:
            extraction_output = self.llm_client.extract_recommendations(
                document_text=full_text,
                page_info=page_info,
                max_recommendations=self.max_recommendations
            )
            
            # Validate and process recommendations
            validated_recommendations = []
            
            for rec in extraction_output.recommendations:
                try:
                    validated_rec = self._validate_recommendation(rec, pages)
                    if validated_rec:
                        validated_recommendations.append(validated_rec)
                        
                except Exception as e:
                    logger.warning(f"Failed to validate recommendation: {e}")
                    continue
            
            logger.info(f"Extracted {len(validated_recommendations)} validated recommendations")
            return validated_recommendations
            
        except Exception as e:
            logger.error(f"Recommendation extraction failed: {e}")
            return []
    
    def _filter_relevant_pages(self, pages: List[PageText]) -> List[PageText]:
        """Filter pages likely to contain recommendations."""
        relevant_pages = []
        
        for page in pages:
            if not page.text:
                continue
            
            page_text_lower = page.text.lower()
            
            # Check for recommendation signals
            signal_count = sum(
                len(re.findall(rf'\b{re.escape(signal.lower())}\b', page_text_lower))
                for signal in self.recommendation_signals
            )
            
            # Check for target sections
            section_matches = sum(
                1 for section in self.target_sections
                if section.lower() in page_text_lower
            )
            
            # Score page relevance
            relevance_score = signal_count + (section_matches * 2)
            
            # Include page if it has sufficient signals
            if relevance_score >= 2:  # Threshold for relevance
                relevant_pages.append(page)
                logger.debug(f"Page {page.page_num}: {signal_count} signals, {section_matches} sections, score={relevance_score}")
        
        return relevant_pages
    
    def _validate_recommendation(
        self, 
        recommendation: PolicyRecommendation, 
        pages: List[PageText]
    ) -> Optional[PolicyRecommendation]:
        """
        Validate and clean up a recommendation.
        
        Args:
            recommendation: Raw recommendation from LLM
            pages: Document pages for quote validation
            
        Returns:
            Validated recommendation or None if invalid
        """
        # Validate required fields
        if not recommendation.action or not recommendation.target:
            logger.warning("Recommendation missing required action or target")
            return None
        
        # Validate evidence quotes
        validated_evidence = []
        for evidence in recommendation.evidence:
            if self._validate_quote(evidence, pages):
                validated_evidence.append(evidence)
            else:
                logger.warning(f"Invalid evidence quote: {evidence.quote[:50]}...")
        
        if not validated_evidence:
            logger.warning("No valid evidence quotes for recommendation")
            return None
        
        # Validate enum values
        validated_rec = self._validate_enum_fields(recommendation)
        if not validated_rec:
            return None
        
        # Update with validated evidence
        validated_rec.evidence = validated_evidence
        
        return validated_rec
    
    def _validate_enum_fields(self, recommendation: PolicyRecommendation) -> Optional[PolicyRecommendation]:
        """Validate enum field values against configuration."""
        try:
            # Check instrument_type
            valid_instruments = self.enums_config.get("instrument_types", [])
            if recommendation.instrument_type not in valid_instruments:
                logger.warning(f"Invalid instrument type: {recommendation.instrument_type}")
                recommendation.instrument_type = "other"
            
            # Check geographic_scope
            valid_scopes = self.enums_config.get("geographic_scopes", [])
            if recommendation.geographic_scope not in valid_scopes:
                logger.warning(f"Invalid geographic scope: {recommendation.geographic_scope}")
                recommendation.geographic_scope = "unspecified" 
            
            # Check timeframe
            valid_timeframes = self.enums_config.get("timeframes", [])
            if recommendation.timeframe not in valid_timeframes:
                logger.warning(f"Invalid timeframe: {recommendation.timeframe}")
                recommendation.timeframe = "unspecified"
            
            # Check strength
            valid_strengths = self.enums_config.get("strengths", [])
            if recommendation.strength not in valid_strengths:
                logger.warning(f"Invalid strength: {recommendation.strength}")
                recommendation.strength = "unspecified"
            
            # Check actor_type
            valid_actors = self.enums_config.get("actor_types", [])
            if recommendation.actor not in valid_actors:
                logger.warning(f"Invalid actor type: {recommendation.actor}")
                recommendation.actor = "unspecified"
            
            # Validate policy_domain (allow any string, but check against config if present)
            valid_domains = self.enums_config.get("policy_domains", [])
            if valid_domains and recommendation.policy_domain not in valid_domains:
                logger.warning(f"Unknown policy domain: {recommendation.policy_domain}")
                # Keep the original value but log the warning
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Enum validation failed: {e}")
            return None
    
    def _validate_quote(self, evidence: Evidence, pages: List[PageText]) -> bool:
        """Validate that quote exists verbatim in the document pages."""
        try:
            # Find the target page
            target_page = None
            for page in pages:
                if page.page_num == evidence.page:
                    target_page = page
                    break
            
            if not target_page:
                logger.warning(f"Page {evidence.page} not found for quote validation")
                return False
            
            quote_clean = evidence.quote.strip()
            page_text = target_page.text
            
            if not quote_clean or len(quote_clean) < 10:
                logger.warning("Quote too short")
                return False
            
            # Try exact match
            if quote_clean in page_text:
                return True
            
            # Try normalized whitespace match
            quote_normalized = re.sub(r'\s+', ' ', quote_clean)
            page_normalized = re.sub(r'\s+', ' ', page_text)
            
            if quote_normalized in page_normalized:
                return True
            
            # Try case-insensitive match as last resort
            if quote_clean.lower() in page_text.lower():
                logger.warning(f"Quote matched case-insensitively only: {quote_clean[:30]}...")
                return True
            
            logger.warning(f"Quote not found: '{quote_clean[:50]}...'")
            return False
            
        except Exception as e:
            logger.error(f"Quote validation error: {e}")
            return False
    
    def _generate_recommendation_id(self, recommendation: PolicyRecommendation, doc_id: str, index: int) -> str:
        """Generate stable recommendation ID."""
        # Create ID based on content hash for stability
        content = f"{recommendation.actor}_{recommendation.action}_{recommendation.target}"
        content_hash = abs(hash(content)) % 10000
        return f"{doc_id}_rec_{index:02d}_{content_hash:04d}"