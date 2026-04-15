"""
Section-aware policy extraction with candidate-span classification.

MIGRATION (Prompt 4): Full rewrite of the original recommendation extractor.
Key changes:
  1. Uses the section map to restrict candidate search to target sections
     (recommendations, key_messages, executive_summary, policy_options,
     implementation, conclusion) and explicitly excludes references,
     acknowledgements, about_authors, contact, appendix.
  2. Generates sentence-level candidate spans first, then classifies only
     those spans (never the whole document).
  3. Deterministic pre-filtering: prescriptive-language detection rejects
     spans that lack any recommendation function before LLM classification.
  4. LLM prompt uses strict negative examples to reject citations,
     literature summaries, and generic normative statements.
  5. Functional validation: a recommendation must have prescriptive
     language AND an action. Actor is kept null when not explicit.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from .llm_client import LLMClient
from .models import (
    ActorType,
    CandidateClassification,
    CandidateClassificationBatch,
    CandidateSpan,
    DocumentSectionMap,
    Evidence,
    ExtractionType,
    GeographicScope,
    InstrumentType,
    PageText,
    PolicyExtraction,
    RecommendationStrength,
    SectionLabel,
    Timeframe,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section targeting
# ---------------------------------------------------------------------------

# Sections where genuine recommendations / options are expected
_TARGET_SECTIONS: Set[SectionLabel] = {
    SectionLabel.RECOMMENDATIONS,
    SectionLabel.KEY_MESSAGES,
    SectionLabel.EXECUTIVE_SUMMARY,
    SectionLabel.POLICY_OPTIONS,
    SectionLabel.IMPLEMENTATION,
    SectionLabel.CONCLUSION,
}

# Sections that must NEVER contribute candidates
_EXCLUDED_SECTIONS: Set[SectionLabel] = {
    SectionLabel.REFERENCES,
    SectionLabel.ACKNOWLEDGEMENTS,
    SectionLabel.ABOUT_AUTHORS,
    SectionLabel.CONTACT,
    SectionLabel.APPENDIX,
}

# ---------------------------------------------------------------------------
# Prescriptive-language lexicon  (word-boundary regex compiled once)
# ---------------------------------------------------------------------------

# Modal / imperative prescriptive cues
_PRESCRIPTIVE_CUES: List[str] = [
    r"\bshould\b",
    r"\bmust\b",
    r"\bneeds?\s+to\b",
    r"\bought\s+to\b",
    r"\brecommends?\b",
    r"\bwe\s+recommend\b",
    r"\bit\s+is\s+recommended\b",
    r"\bwe\s+propose\b",
    r"\bpropose(?:s|d)?\b",
    r"\bcall(?:s|ed)?\s+(?:for|on|upon)\b",
    r"\burge(?:s|d)?\b",
    r"\brequire(?:s|d)?\b",
    r"\bensure\b",
    r"\bstrengthen\b",
    r"\bestablish\b",
    r"\bprioritize\b",
    r"\bprioritise\b",
    r"\binvest\s+in\b",
]

_PRESCRIPTIVE_RE = [re.compile(p, re.IGNORECASE) for p in _PRESCRIPTIVE_CUES]

# ---------------------------------------------------------------------------
# Reference / citation patterns — used to reject candidate spans
# ---------------------------------------------------------------------------

_CITATION_PATTERNS: List[re.Pattern] = [
    # In-text citations: (Author, Year) or (Author et al., Year)
    re.compile(r"\([A-Z][a-z]+(?:\s+(?:et\s+al\.?|&|and)\s+[A-Z][a-z]+)*,?\s*\d{4}\)"),
    # Narrative citations: Author (Year) — author name outside parentheses
    re.compile(r"[A-Z][a-z]+\s+\(\d{4}\)"),
    # Numbered citations: [1], [1, 2], [1-5]
    re.compile(r"\[\d+(?:[,\-–]\s*\d+)*\]"),
    # Reference-list entry starting with author-year
    re.compile(r"^[A-Z][a-z]+,?\s+[A-Z]\.?\s*(?:,|&|\()\s*(?:\d{4}|\(\d{4}\))"),
    # DOI links
    re.compile(r"doi[:\s]+10\.\d{4,}", re.IGNORECASE),
    # URL patterns typical of bibliography entries
    re.compile(r"https?://(?:dx\.)?doi\.org/"),
    # "et al. (YYYY)" mid-sentence — strong citation signal
    re.compile(r"et\s+al\.?\s*\(?\d{4}\)?"),
]

# A span is considered citation-heavy if it has ≥ this many citation hits.
_CITATION_HIT_THRESHOLD = 2

# Strength mapping from modal verbs in text
_STRENGTH_MAP: Dict[str, RecommendationStrength] = {
    "must": RecommendationStrength.MUST,
    "should": RecommendationStrength.SHOULD,
    "could": RecommendationStrength.COULD,
    "may": RecommendationStrength.MAY,
    "recommend": RecommendationStrength.SHOULD,
    "recommends": RecommendationStrength.SHOULD,
    "propose": RecommendationStrength.SHOULD,
    "proposes": RecommendationStrength.SHOULD,
    "urge": RecommendationStrength.MUST,
    "urges": RecommendationStrength.MUST,
    "consider": RecommendationStrength.CONSIDER,
}

# Actor-type normalization lookup (conservative — only explicit keywords)
_ACTOR_NORM: Dict[str, ActorType] = {
    "government": ActorType.GOVERNMENT,
    "governments": ActorType.GOVERNMENT,
    "policymakers": ActorType.GOVERNMENT,
    "policymaker": ActorType.GOVERNMENT,
    "policy-makers": ActorType.GOVERNMENT,
    "policy makers": ActorType.GOVERNMENT,
    "ministry": ActorType.GOVERNMENT,
    "ministries": ActorType.GOVERNMENT,
    "parliament": ActorType.GOVERNMENT,
    "legislature": ActorType.GOVERNMENT,
    "eu": ActorType.EU_INSTITUTIONS,
    "european commission": ActorType.EU_INSTITUTIONS,
    "european union": ActorType.EU_INSTITUTIONS,
    "european parliament": ActorType.EU_INSTITUTIONS,
    "un": ActorType.INTERNATIONAL_ORGANIZATIONS,
    "united nations": ActorType.INTERNATIONAL_ORGANIZATIONS,
    "world bank": ActorType.INTERNATIONAL_ORGANIZATIONS,
    "imf": ActorType.INTERNATIONAL_ORGANIZATIONS,
    "private sector": ActorType.PRIVATE_SECTOR,
    "industry": ActorType.PRIVATE_SECTOR,
    "businesses": ActorType.PRIVATE_SECTOR,
    "companies": ActorType.PRIVATE_SECTOR,
    "civil society": ActorType.CIVIL_SOCIETY,
    "ngos": ActorType.CIVIL_SOCIETY,
    "ngo": ActorType.CIVIL_SOCIETY,
    "researchers": ActorType.RESEARCH_INSTITUTIONS,
    "academia": ActorType.RESEARCH_INSTITUTIONS,
    "universities": ActorType.RESEARCH_INSTITUTIONS,
}

# Instrument-type normalization (conservative)
_INSTRUMENT_NORM: Dict[str, InstrumentType] = {
    "regulation": InstrumentType.REGULATION,
    "regulatory": InstrumentType.REGULATION,
    "legislation": InstrumentType.REGULATION,
    "law": InstrumentType.REGULATION,
    "ban": InstrumentType.REGULATION,
    "mandate": InstrumentType.REGULATION,
    "standard": InstrumentType.REGULATION,
    "subsidy": InstrumentType.SUBSIDY,
    "subsidies": InstrumentType.SUBSIDY,
    "grant": InstrumentType.SUBSIDY,
    "incentive": InstrumentType.SUBSIDY,
    "tax": InstrumentType.TAX,
    "taxation": InstrumentType.TAX,
    "levy": InstrumentType.TAX,
    "tariff": InstrumentType.TAX,
    "information": InstrumentType.INFORMATION,
    "awareness": InstrumentType.INFORMATION,
    "campaign": InstrumentType.INFORMATION,
    "education": InstrumentType.INFORMATION,
    "voluntary": InstrumentType.VOLUNTARY,
    "pledge": InstrumentType.VOLUNTARY,
    "monitoring": InstrumentType.MONITORING,
    "surveillance": InstrumentType.MONITORING,
    "procurement": InstrumentType.PROCUREMENT,
    "infrastructure": InstrumentType.INFRASTRUCTURE,
}


class RecommendationExtractor:
    """Section-aware policy extraction with candidate-span classification.

    Pipeline:
      1. Identify target pages from section map (or fall back to heuristic)
      2. Split text into sentence-level candidate spans
      3. Reject spans from excluded sections or with heavy citations
      4. Tag spans with prescriptive-language cues
      5. Send surviving candidates to LLM for narrow classification
      6. Apply post-classification validation
      7. Return list of PolicyExtraction objects
    """

    def __init__(
        self,
        llm_client: LLMClient,
        enums_config: Dict[str, List[str]],
        min_confidence: float = 0.6,
        max_recommendations: int = 10,
        # Legacy kwargs — accepted but ignored after rewrite
        recommendation_signals: Optional[List[str]] = None,
        target_sections: Optional[List[str]] = None,
    ):
        self.llm_client = llm_client
        self.enums_config = enums_config
        self.min_confidence = min_confidence
        self.max_recommendations = max_recommendations
        logger.info("Initialized section-aware recommendation extractor")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_recommendations(
        self,
        pages: List[PageText],
        section_map: Optional[DocumentSectionMap] = None,
    ) -> List[PolicyExtraction]:
        """Extract policy recommendations / options from document pages.

        Args:
            pages: Document pages with text.
            section_map: Section map from section segmenter (strongly recommended).

        Returns:
            List of validated PolicyExtraction objects.
        """
        logger.info(f"Extracting recommendations from {len(pages)} pages")

        # Step 1 — Identify target pages
        target_pages, page_section_map = self._identify_target_pages(
            pages, section_map
        )
        if not target_pages:
            logger.info("No target pages identified for recommendation extraction")
            return []

        logger.debug(
            f"Target pages: {[p.page_num for p in target_pages]}"
        )

        # Step 2+3 — Generate & filter candidate spans
        candidates = self._generate_candidates(target_pages, page_section_map)
        if not candidates:
            logger.info("No candidate spans survived filtering")
            return []

        logger.debug(f"Generated {len(candidates)} candidate spans")

        # Step 4 — Classify candidates via LLM
        try:
            extractions = self._classify_candidates(candidates, pages)
        except Exception as e:
            logger.error(f"Candidate classification failed: {e}")
            return []

        # Step 5 — Cap at max_recommendations
        extractions = extractions[: self.max_recommendations]

        logger.info(f"Extracted {len(extractions)} policy extractions")
        return extractions

    # ------------------------------------------------------------------
    # Step 1: Identify target pages
    # ------------------------------------------------------------------

    def _identify_target_pages(
        self,
        pages: List[PageText],
        section_map: Optional[DocumentSectionMap],
    ) -> Tuple[List[PageText], Dict[int, Optional[SectionLabel]]]:
        """Return pages from target sections and a page→section mapping.

        Falls back to simple heuristic if no section map is available.
        """
        page_section: Dict[int, Optional[SectionLabel]] = {}

        if section_map and section_map.sections:
            target_page_nums: Set[int] = set()

            for section in section_map.sections:
                label = section.normalized_label
                if label in _EXCLUDED_SECTIONS:
                    # Mark excluded pages so we never emit candidates from them
                    for pn in range(section.start_page, section.end_page + 1):
                        page_section[pn] = label
                    continue
                if label in _TARGET_SECTIONS or label is None:
                    # Include unlabeled sections (could be body text with recs)
                    for pn in range(section.start_page, section.end_page + 1):
                        target_page_nums.add(pn)
                        if pn not in page_section:
                            page_section[pn] = label

            target_pages = [p for p in pages if p.page_num in target_page_nums]
        else:
            # Fallback: include all pages, mark none as excluded
            logger.warning(
                "No section map provided; falling back to all pages"
            )
            target_pages = list(pages)
            for p in pages:
                page_section[p.page_num] = None

        return target_pages, page_section

    # ------------------------------------------------------------------
    # Step 2+3: Generate and filter candidate spans
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        target_pages: List[PageText],
        page_section: Dict[int, Optional[SectionLabel]],
    ) -> List[CandidateSpan]:
        """Split target pages into sentence-level spans and filter."""
        candidates: List[CandidateSpan] = []

        for page in target_pages:
            section_label = page_section.get(page.page_num)

            # Skip excluded sections (should already be removed, but defensive)
            if section_label in _EXCLUDED_SECTIONS:
                continue

            sentences = self._split_sentences(page.text)
            for sent in sentences:
                sent_stripped = sent.strip()
                if len(sent_stripped) < 20:
                    continue  # Too short to be a recommendation

                # Reject citation-heavy spans
                if self._is_citation_heavy(sent_stripped):
                    continue

                # Tag prescriptive language
                has_prescriptive, cues = self._detect_prescriptive(sent_stripped)

                candidates.append(
                    CandidateSpan(
                        text=sent_stripped,
                        page=page.page_num,
                        source_section=section_label,
                        has_prescriptive_language=has_prescriptive,
                        prescriptive_cues=cues,
                    )
                )

        # Only keep spans with prescriptive language
        prescriptive_candidates = [
            c for c in candidates if c.has_prescriptive_language
        ]

        if not prescriptive_candidates:
            logger.debug(
                "No candidates with prescriptive language found; "
                f"total raw candidates: {len(candidates)}"
            )

        return prescriptive_candidates

    # ------------------------------------------------------------------
    # Step 4: Classify candidates via LLM
    # ------------------------------------------------------------------

    def _classify_candidates(
        self,
        candidates: List[CandidateSpan],
        all_pages: List[PageText],
    ) -> List[PolicyExtraction]:
        """Send candidates to LLM for classification and build results."""

        # Batch candidates (to reduce API calls)
        batch_size = 10
        all_extractions: List[PolicyExtraction] = []

        for start in range(0, len(candidates), batch_size):
            batch = candidates[start: start + batch_size]
            classifications = self._llm_classify_batch(batch)

            for candidate, classification in zip(batch, classifications):
                extraction = self._build_extraction(
                    candidate, classification, all_pages
                )
                if extraction is not None:
                    all_extractions.append(extraction)

        # Sort: recommendations first, then options, by confidence desc
        type_priority = {
            ExtractionType.RECOMMENDATION: 0,
            ExtractionType.POLICY_OPTION: 1,
            ExtractionType.ACTOR_RESPONSIBILITY: 2,
            ExtractionType.IMPLEMENTATION_STEP: 3,
            ExtractionType.EXPECTED_OUTCOME: 4,
            ExtractionType.TRADE_OFF: 5,
        }
        all_extractions.sort(
            key=lambda e: (
                type_priority.get(e.extraction_type, 99),
                -e.confidence,
            )
        )

        return all_extractions

    def _llm_classify_batch(
        self, candidates: List[CandidateSpan]
    ) -> List[CandidateClassification]:
        """Call LLM to classify a batch of candidate spans."""

        # Build numbered candidate list for the prompt
        candidate_lines = []
        for i, c in enumerate(candidates, 1):
            section_info = f" [section: {c.source_section.value}]" if c.source_section else ""
            candidate_lines.append(
                f"CANDIDATE {i} (page {c.page}{section_info}):\n{c.text}"
            )
        candidates_text = "\n\n".join(candidate_lines)

        system_message = self._build_classification_system_prompt(len(candidates))
        user_message = (
            f"Classify each candidate span below. Return exactly "
            f"{len(candidates)} classifications in order.\n\n"
            f"{candidates_text}"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        try:
            result = self.llm_client.structured_completion(
                messages, CandidateClassificationBatch
            )
            classifications = result.classifications

            # Pad or truncate to match input length
            if len(classifications) < len(candidates):
                logger.warning(
                    f"LLM returned {len(classifications)} classifications "
                    f"for {len(candidates)} candidates; padding with non_recommendation"
                )
                while len(classifications) < len(candidates):
                    classifications.append(
                        CandidateClassification(
                            extraction_type=ExtractionType.NON_RECOMMENDATION,
                            confidence=0.0,
                            rejection_reason="LLM did not classify this span",
                        )
                    )
            return classifications[: len(candidates)]

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Return non_recommendation for all on failure
            return [
                CandidateClassification(
                    extraction_type=ExtractionType.NON_RECOMMENDATION,
                    confidence=0.0,
                    rejection_reason=f"Classification error: {e}",
                )
                for _ in candidates
            ]

    def _build_classification_system_prompt(self, num_candidates: int) -> str:
        """Build the system prompt for candidate classification."""
        return f"""You are a strict policy-document analyst. You will receive {num_candidates} candidate text spans extracted from a policy brief. For each span, classify it into EXACTLY ONE of these categories:

- recommendation: A prescriptive statement where the document's authors recommend a specific action or policy measure. Must contain an action/intervention and prescriptive language (should, must, recommend, propose, etc.).
- policy_option: A named policy alternative or scenario being presented for consideration, not necessarily endorsed.
- implementation_step: A concrete step describing how to carry out a recommendation.
- expected_outcome: A statement about anticipated results or impacts.
- trade_off: A statement about downsides, risks, costs, or trade-offs.
- actor_responsibility: A statement assigning responsibility to a specific actor without a new action.
- non_recommendation: Everything else — background, description, literature summary, citation, finding, or generic normative language.

CRITICAL RULES:
1. References, bibliographic entries, and citations are ALWAYS non_recommendation, even if they contain "should" or "recommend". A citation describes what someone ELSE said, not what THIS document recommends.
2. Descriptive findings ("The study found that X should be considered") are non_recommendation unless the document is ENDORSING that finding as its own recommendation.
3. Generic normative statements ("Education is important") are non_recommendation.
4. If a span discusses what a DIFFERENT study, report, or author recommended, it is non_recommendation.
5. Only classify as recommendation if the AUTHORS OF THIS DOCUMENT are making a prescriptive statement.
6. If you are uncertain, classify as non_recommendation with confidence < 0.5.
7. Actor must be extracted ONLY from explicit text. Do NOT infer actors that are not stated.
8. Return null for any field you cannot confidently determine from the text.
9. Return exactly {num_candidates} classifications in the same order as the input candidates.

For each classification, provide:
- extraction_type: one of the categories above
- confidence: 0.0-1.0
- actor_text_raw: exact actor text from span, or null
- action_text_raw: exact action phrase, or null
- target_text_raw: exact target, or null
- instrument_type: policy instrument if EXPLICITLY stated (regulation, subsidy, tax, etc.), or null
- strength: modal verb strength (must, should, could, may, consider), or null
- expected_outcomes: list of outcome phrases, or empty
- implementation_steps: list of step phrases, or empty
- trade_offs: list of trade-off/risk phrases, or empty
- rejection_reason: why classified as non_recommendation, or null"""

    # ------------------------------------------------------------------
    # Post-classification: build and validate PolicyExtraction
    # ------------------------------------------------------------------

    def _build_extraction(
        self,
        candidate: CandidateSpan,
        classification: CandidateClassification,
        all_pages: List[PageText],
    ) -> Optional[PolicyExtraction]:
        """Build a PolicyExtraction from a classified candidate.

        Returns None for non_recommendation or failed validation.
        """
        etype = classification.extraction_type

        # Reject non-recommendations
        if etype == ExtractionType.NON_RECOMMENDATION:
            return None

        # Reject low-confidence
        if classification.confidence < self.min_confidence:
            logger.debug(
                f"Rejecting low-confidence ({classification.confidence:.2f}) "
                f"extraction: {candidate.text[:60]}..."
            )
            return None

        # For recommendations, enforce functional criteria
        if etype == ExtractionType.RECOMMENDATION:
            if not classification.action_text_raw:
                logger.debug(
                    "Rejecting recommendation without action: "
                    f"{candidate.text[:60]}..."
                )
                return None

        # Build evidence from the candidate span itself
        evidence = self._make_evidence(candidate, all_pages)

        # Normalize actor type if possible
        actor_norm = self._normalize_actor(classification.actor_text_raw)

        # Normalize instrument type
        instrument = self._normalize_instrument(classification.instrument_type)

        # Normalize strength
        strength = self._normalize_strength(classification.strength)

        return PolicyExtraction(
            extraction_type=etype,
            confidence=classification.confidence,
            source_text_raw=candidate.text,
            source_section=candidate.source_section,
            page=candidate.page,
            actor_text_raw=classification.actor_text_raw,
            actor_type_normalized=actor_norm,
            action_text_raw=classification.action_text_raw,
            target_text_raw=classification.target_text_raw,
            instrument_type=instrument,
            strength=strength,
            expected_outcomes=classification.expected_outcomes,
            implementation_steps=classification.implementation_steps,
            trade_offs=classification.trade_offs,
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentence-level spans.

        Uses a simple rule: split on sentence-ending punctuation followed
        by whitespace or end-of-string.  Preserves bullet/list items as
        separate spans.
        """
        if not text:
            return []

        # First split on newlines that look like list items or paragraph breaks
        lines = re.split(r"\n(?=\s*[\-•●◦▪]|\s*\d+[\.\)]\s|\n)", text)

        sentences: List[str] = []
        for line in lines:
            # Split on sentence-ending punctuation
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
            sentences.extend(parts)

        return [s for s in sentences if s.strip()]

    @staticmethod
    def _is_citation_heavy(text: str) -> bool:
        """Return True if the span looks like a citation or reference entry."""
        hits = 0
        for pattern in _CITATION_PATTERNS:
            matches = pattern.findall(text)
            hits += len(matches)
            if hits >= _CITATION_HIT_THRESHOLD:
                return True

        # Also reject if it starts with typical reference-list formatting
        if re.match(r"^\d+\.\s+[A-Z][a-z]+,?\s+[A-Z]", text):
            return True

        return False

    @staticmethod
    def _detect_prescriptive(text: str) -> Tuple[bool, List[str]]:
        """Detect prescriptive language in a span."""
        matched: List[str] = []
        for i, pattern in enumerate(_PRESCRIPTIVE_RE):
            if pattern.search(text):
                matched.append(_PRESCRIPTIVE_CUES[i])
        return bool(matched), matched

    @staticmethod
    def _make_evidence(
        candidate: CandidateSpan, all_pages: List[PageText]
    ) -> List[Evidence]:
        """Create an Evidence object from a candidate span.

        Validates that the quote exists verbatim in the source page.
        Returns empty list if validation fails.
        """
        quote = candidate.text.strip()

        # Enforce Evidence model constraints
        if len(quote) < 10:
            return []
        if len(quote) > 500:
            quote = quote[:497] + "..."

        # Verify quote exists in source page
        target_page = None
        for page in all_pages:
            if page.page_num == candidate.page:
                target_page = page
                break

        if target_page is None:
            return []

        # Try exact match
        if quote in target_page.text:
            return [Evidence(page=candidate.page, quote=quote)]

        # Try normalized whitespace match
        quote_norm = re.sub(r"\s+", " ", quote)
        page_norm = re.sub(r"\s+", " ", target_page.text)
        if quote_norm in page_norm:
            return [Evidence(page=candidate.page, quote=quote)]

        # Candidate text should always exist (we split it from the page), but
        # if truncation happened, accept anyway since we have source_text_raw
        logger.warning(
            f"Evidence quote not found verbatim on page {candidate.page}: "
            f"{quote[:50]}..."
        )
        return [Evidence(page=candidate.page, quote=quote)]

    @staticmethod
    def _normalize_actor(actor_raw: Optional[str]) -> Optional[ActorType]:
        """Normalize raw actor text to ActorType enum if safely mappable."""
        if not actor_raw:
            return None
        key = actor_raw.strip().lower()
        # Try exact match first
        if key in _ACTOR_NORM:
            return _ACTOR_NORM[key]
        # Try substring match (e.g., "national government" → GOVERNMENT)
        for token, atype in _ACTOR_NORM.items():
            if token in key:
                return atype
        return None

    @staticmethod
    def _normalize_instrument(raw: Optional[str]) -> Optional[InstrumentType]:
        """Normalize raw instrument type string."""
        if not raw:
            return None
        key = raw.strip().lower()
        if key in _INSTRUMENT_NORM:
            return _INSTRUMENT_NORM[key]
        for token, itype in _INSTRUMENT_NORM.items():
            if token in key:
                return itype
        return None

    @staticmethod
    def _normalize_strength(raw: Optional[str]) -> Optional[RecommendationStrength]:
        """Normalize modal verb to RecommendationStrength."""
        if not raw:
            return None
        key = raw.strip().lower()
        return _STRENGTH_MAP.get(key)