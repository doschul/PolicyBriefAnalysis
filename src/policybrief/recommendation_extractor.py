"""
Recommendation extraction: references-page exclusion, prescriptive-language
pre-filter, citation rejection, batch LLM classification, post-validation.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .llm_client import LLMClient
from .models import (
    ActorType,
    CandidateClassification,
    CandidateClassificationBatch,
    Evidence,
    ExtractionType,
    InstrumentType,
    PageText,
    PolicyExtraction,
    RecommendationStrength,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_REFS_PATTERNS = [
    re.compile(r"^\s*references?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*bibliography\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*works?\s+cited\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*literature\s+cited\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*endnotes?\s*$", re.IGNORECASE | re.MULTILINE),
]

_PRESCRIPTIVE_CUES = [
    re.compile(r"\bshould\b", re.IGNORECASE),
    re.compile(r"\bmust\b", re.IGNORECASE),
    re.compile(r"\bneed\s+to\b", re.IGNORECASE),
    re.compile(r"\bought\s+to\b", re.IGNORECASE),
    re.compile(r"\brecommend", re.IGNORECASE),
    re.compile(r"\bpropose[sd]?\b", re.IGNORECASE),
    re.compile(r"\bsuggest", re.IGNORECASE),
    re.compile(r"\bcall[s]?\s+for\b", re.IGNORECASE),
    re.compile(r"\burge[sd]?\b", re.IGNORECASE),
    re.compile(r"\brequire[sd]?\b", re.IGNORECASE),
    re.compile(r"\badvise[sd]?\b", re.IGNORECASE),
    re.compile(r"\badvocate[sd]?\b", re.IGNORECASE),
    re.compile(r"\bprioritize\b", re.IGNORECASE),
    re.compile(r"\bimplement\b", re.IGNORECASE),
    re.compile(r"\bestablish\b", re.IGNORECASE),
    re.compile(r"\bstrengthen\b", re.IGNORECASE),
    re.compile(r"\bensure\b", re.IGNORECASE),
    re.compile(r"\bfoster\b", re.IGNORECASE),
    re.compile(r"\bpromote\b", re.IGNORECASE),
    re.compile(r"\bencourage\b", re.IGNORECASE),
]

_CITATION_PATTERNS = [
    re.compile(r"\(\s*[A-Z][a-z]+(?:\s+(?:et\s+al\.?|&|and)\s+[A-Z][a-z]+)*\s*,?\s*\d{4}\s*\)"),
    re.compile(r"\[\d+(?:[,;\s]+\d+)*\]"),
    re.compile(r"\b(?:ibid|op\.?\s*cit|loc\.?\s*cit)\b", re.IGNORECASE),
]

_STRENGTH_MAP = {
    "must": RecommendationStrength.MUST,
    "require": RecommendationStrength.MUST,
    "should": RecommendationStrength.SHOULD,
    "ought": RecommendationStrength.SHOULD,
    "recommend": RecommendationStrength.SHOULD,
    "could": RecommendationStrength.COULD,
    "suggest": RecommendationStrength.COULD,
    "may": RecommendationStrength.MAY,
    "consider": RecommendationStrength.CONSIDER,
    "might": RecommendationStrength.CONSIDER,
}

_ACTOR_MAP = {
    "government": ActorType.GOVERNMENT,
    "state": ActorType.GOVERNMENT,
    "national": ActorType.GOVERNMENT,
    "ministry": ActorType.GOVERNMENT,
    "eu": ActorType.EU_INSTITUTIONS,
    "european": ActorType.EU_INSTITUTIONS,
    "commission": ActorType.EU_INSTITUTIONS,
    "un": ActorType.INTERNATIONAL_ORGANIZATIONS,
    "united nations": ActorType.INTERNATIONAL_ORGANIZATIONS,
    "fao": ActorType.INTERNATIONAL_ORGANIZATIONS,
    "world bank": ActorType.INTERNATIONAL_ORGANIZATIONS,
    "private": ActorType.PRIVATE_SECTOR,
    "company": ActorType.PRIVATE_SECTOR,
    "industry": ActorType.PRIVATE_SECTOR,
    "ngo": ActorType.CIVIL_SOCIETY,
    "civil society": ActorType.CIVIL_SOCIETY,
    "community": ActorType.CIVIL_SOCIETY,
    "research": ActorType.RESEARCH_INSTITUTIONS,
    "university": ActorType.RESEARCH_INSTITUTIONS,
    "academic": ActorType.RESEARCH_INSTITUTIONS,
}

_INSTRUMENT_MAP = {
    "regulation": InstrumentType.REGULATION,
    "law": InstrumentType.REGULATION,
    "legislation": InstrumentType.REGULATION,
    "subsidy": InstrumentType.SUBSIDY,
    "payment": InstrumentType.SUBSIDY,
    "incentive": InstrumentType.SUBSIDY,
    "tax": InstrumentType.TAX,
    "levy": InstrumentType.TAX,
    "information": InstrumentType.INFORMATION,
    "monitoring": InstrumentType.MONITORING,
    "research": InstrumentType.RESEARCH,
    "certification": InstrumentType.VOLUNTARY,
    "voluntary": InstrumentType.VOLUNTARY,
    "planning": InstrumentType.PLANNING,
    "procurement": InstrumentType.PROCUREMENT,
    "infrastructure": InstrumentType.INFRASTRUCTURE,
    "institutional": InstrumentType.INSTITUTIONAL,
}

# ── LLM Prompt ────────────────────────────────────────────────────────────

_CLASSIFICATION_PROMPT = """You are an expert policy analyst. Classify each candidate text span.

Types:
- recommendation: A clear suggestion or directive for future action by a specific or implicit actor.
- policy_option: A policy alternative presented for consideration (not yet endorsed).
- implementation_step: A concrete action step for implementing a broader recommendation.
- expected_outcome: A predicted result of a recommendation or policy.
- trade_off: An acknowledged tension or cost of a policy choice.
- actor_responsibility: Assignment of responsibility to a specific actor.
- non_recommendation: The text does not contain any of the above.

Rules:
- Only classify as 'recommendation' if the text contains a clear call to action.
- Do NOT classify citations, references, bibliographic entries, or footnotes as recommendations.
- Do NOT classify general observations or descriptions of existing policy as recommendations.
- If the actor is not explicitly named, set actor_text_raw to null.
- If the action is vague or implied, set action_text_raw to null.
- Confidence: 0.0-1.0. Use 0.8+ only for unambiguous prescriptive statements.
- For non_recommendation, set rejection_reason explaining why.

Return one classification per candidate, in order."""


class RecommendationExtractor:
    """Extract policy recommendations from document text."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: Dict[str, Any],
    ):
        self.llm = llm_client
        self.min_confidence = config.get("min_confidence", 0.6)
        self.batch_size = config.get("batch_size", 10)

    # ── Public API ────────────────────────────────────────────────────

    def extract_recommendations(
        self,
        pages: List[PageText],
        doc_id: str,
    ) -> List[PolicyExtraction]:
        """Extract recommendations from document pages."""
        # Stage 1: Detect and exclude reference pages
        refs_start = detect_references_start_page(pages)
        if refs_start is not None:
            eligible_pages = [p for p in pages if p.page_num < refs_start]
            logger.info(
                f"[{doc_id}] References start at page {refs_start}, "
                f"using {len(eligible_pages)}/{len(pages)} pages"
            )
        else:
            eligible_pages = pages

        if not eligible_pages:
            return []

        # Stage 2: Generate candidates (prescriptive filter + citation reject)
        candidates = self._generate_candidates(eligible_pages)
        if not candidates:
            logger.info(f"[{doc_id}] No prescriptive candidates found")
            return []

        logger.info(f"[{doc_id}] {len(candidates)} prescriptive candidates")

        # Stage 3: Batch LLM classification
        classifications = self._classify_candidates(candidates)

        # Stage 4: Build validated extractions
        full_text = "\n".join(p.text for p in eligible_pages)
        extractions = self._build_extractions(
            candidates, classifications, full_text, doc_id
        )
        return extractions

    # ── Stage 2: Candidate generation ─────────────────────────────────

    def _generate_candidates(
        self,
        pages: List[PageText],
    ) -> List[Dict[str, Any]]:
        """Split into sentences, keep prescriptive, reject citation-heavy."""
        candidates: List[Dict[str, Any]] = []
        for page in pages:
            sentences = _split_sentences(page.text)
            for sent in sentences:
                if len(sent.split()) < 5:
                    continue
                if not _has_prescriptive_cue(sent):
                    continue
                if _is_citation_heavy(sent):
                    continue
                candidates.append({
                    "text": sent.strip(),
                    "page": page.page_num,
                })
        return candidates

    # ── Stage 3: LLM classification ──────────────────────────────────

    def _classify_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[CandidateClassification]:
        """Send candidate batches to LLM for classification."""
        all_classifications: List[CandidateClassification] = []

        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i : i + self.batch_size]
            batch_text = "\n".join(
                f"[{j+1}] (page {c['page']}): {c['text']}"
                for j, c in enumerate(batch)
            )

            messages = [
                {"role": "system", "content": _CLASSIFICATION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Classify these {len(batch)} candidate spans:\n\n{batch_text}"
                    ),
                },
            ]

            try:
                result: CandidateClassificationBatch = self.llm.structured_completion(
                    messages, CandidateClassificationBatch
                )
                # Pad or truncate to match batch size
                clfs = result.classifications
                while len(clfs) < len(batch):
                    clfs.append(CandidateClassification(
                        extraction_type=ExtractionType.NON_RECOMMENDATION,
                        confidence=0.0,
                        rejection_reason="No classification returned",
                    ))
                all_classifications.extend(clfs[: len(batch)])
            except Exception as exc:
                logger.warning(f"Batch classification failed: {exc}")
                all_classifications.extend(
                    CandidateClassification(
                        extraction_type=ExtractionType.NON_RECOMMENDATION,
                        confidence=0.0,
                        rejection_reason=f"LLM error: {exc}",
                    )
                    for _ in batch
                )

        return all_classifications

    # ── Stage 4: Post-validation and extraction building ──────────────

    def _build_extractions(
        self,
        candidates: List[Dict[str, Any]],
        classifications: List[CandidateClassification],
        full_text: str,
        doc_id: str,
    ) -> List[PolicyExtraction]:
        extractions: List[PolicyExtraction] = []
        rec_counter = 0

        for cand, clf in zip(candidates, classifications):
            if clf.extraction_type == ExtractionType.NON_RECOMMENDATION:
                continue
            if clf.confidence < self.min_confidence:
                continue

            # Build evidence (quote = candidate text itself)
            evidence = self._build_evidence(cand, full_text)

            # Skip recommendations that require evidence but have none
            if clf.extraction_type in (
                ExtractionType.RECOMMENDATION,
                ExtractionType.POLICY_OPTION,
            ) and not evidence:
                continue

            rec_counter += 1
            rec_id = f"{doc_id}_rec_{rec_counter:03d}"

            extractions.append(PolicyExtraction(
                rec_id=rec_id,
                extraction_type=clf.extraction_type,
                confidence=clf.confidence,
                source_text_raw=cand["text"],
                source_section=None,
                page=cand["page"],
                actor_text_raw=clf.actor_text_raw,
                actor_type_normalized=_normalize_actor(clf.actor_text_raw),
                action_text_raw=clf.action_text_raw,
                target_text_raw=clf.target_text_raw,
                instrument_type=_normalize_instrument(clf.instrument_type),
                strength=_normalize_strength(clf.strength),
                expected_outcomes=clf.expected_outcomes,
                implementation_steps=clf.implementation_steps,
                trade_offs=clf.trade_offs,
                evidence=evidence,
            ))

        return extractions

    @staticmethod
    def _build_evidence(cand: Dict[str, Any], full_text: str) -> List[Evidence]:
        """Create evidence from the candidate text if it exists in source."""
        quote = cand["text"][:500].strip()
        if len(quote) < 10:
            return []
        # Verify quote is in source text (normalised whitespace)
        norm_source = re.sub(r"\s+", " ", full_text)
        norm_quote = re.sub(r"\s+", " ", quote)
        if norm_quote not in norm_source:
            # Try shorter prefix
            short = norm_quote[:100]
            if short not in norm_source:
                return []
        return [Evidence(page=cand["page"], quote=quote)]


# ── Module-level helpers ──────────────────────────────────────────────────


def detect_references_start_page(pages: List[PageText]) -> Optional[int]:
    """Find the page where references/bibliography section begins.

    Searches only the last 40% of the document (from the back) to avoid
    false positives from citation lines or author-block "Reference" headings
    that appear near the front of the document.
    """
    if not pages:
        return None
    # Only search the back portion of the document
    cutoff = max(1, int(len(pages) * 0.6))
    back_pages = [p for p in pages if p.page_num > cutoff]
    if not back_pages:
        back_pages = pages[-2:]  # at least check the last 2 pages

    for page in reversed(back_pages):
        for pat in _REFS_PATTERNS:
            if pat.search(page.text):
                return page.page_num
    return None


def _split_sentences(text: str) -> List[str]:
    """Cheap regex sentence splitter."""
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if len(s.strip()) > 10]


def _has_prescriptive_cue(sentence: str) -> bool:
    return any(p.search(sentence) for p in _PRESCRIPTIVE_CUES)


def _is_citation_heavy(sentence: str, threshold: float = 0.4) -> bool:
    """Reject sentences where >40% of content is citation markup."""
    total_len = len(sentence)
    if total_len == 0:
        return False
    citation_chars = sum(
        len(m.group()) for p in _CITATION_PATTERNS for m in p.finditer(sentence)
    )
    return (citation_chars / total_len) > threshold


def _normalize_strength(raw: Optional[str]) -> Optional[RecommendationStrength]:
    if not raw:
        return None
    low = raw.lower().strip()
    # Try direct enum match
    try:
        return RecommendationStrength(low)
    except ValueError:
        pass
    # Try keyword map
    for key, val in _STRENGTH_MAP.items():
        if key in low:
            return val
    return RecommendationStrength.UNSPECIFIED


def _normalize_actor(raw: Optional[str]) -> Optional[ActorType]:
    if not raw:
        return None
    low = raw.lower()
    for key, val in _ACTOR_MAP.items():
        if key in low:
            return val
    return ActorType.UNSPECIFIED


def _normalize_instrument(raw: Optional[str]) -> Optional[InstrumentType]:
    if not raw:
        return None
    # Try direct enum match
    try:
        return InstrumentType(raw.lower().strip())
    except ValueError:
        pass
    low = raw.lower()
    for key, val in _INSTRUMENT_MAP.items():
        if key in low:
            return val
    return InstrumentType.OTHER
