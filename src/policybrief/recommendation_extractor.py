"""
Broad-content recommendation extraction via multi-pass LLM.

Replaces narrow sentence-level prescriptive-cue prefiltering with
broad page-window LLM passes. Deterministic rails: reference-boundary
exclusion, evidence verification, normalization, deduplication.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .llm_client import LLMClient
from .models import (
    ActorType,
    Evidence,
    ExtractionType,
    InstrumentType,
    PageText,
    PolicyExtraction,
    RecommendationExtractionResponse,
    RecommendationStrength,
)

logger = logging.getLogger(__name__)

# ── Reference boundary detection ──────────────────────────────────────────

_REFS_PATTERNS = [
    re.compile(r"^\s*references?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*bibliography\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*works?\s+cited\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*literature\s+cited\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*endnotes?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*sources?\s*$", re.IGNORECASE | re.MULTILINE),
]


def detect_references_start_page(pages: List[PageText]) -> Optional[int]:
    """Find the page where references/bibliography section begins.

    Searches only the last 40% of the document to avoid false positives
    from citation lines or author-block headings near the front.
    """
    if not pages:
        return None
    cutoff = max(1, int(len(pages) * 0.6))
    back_pages = [p for p in pages if p.page_num > cutoff]
    if not back_pages:
        back_pages = pages[-2:]

    for page in reversed(back_pages):
        for pat in _REFS_PATTERNS:
            if pat.search(page.text):
                return page.page_num
    return None


# ── Normalization helpers ─────────────────────────────────────────────────

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


def _normalize_strength(raw: Optional[str]) -> Optional[RecommendationStrength]:
    if not raw:
        return None
    low = raw.lower().strip()
    try:
        return RecommendationStrength(low)
    except ValueError:
        pass
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
    try:
        return InstrumentType(raw.lower().strip())
    except ValueError:
        pass
    low = raw.lower()
    for key, val in _INSTRUMENT_MAP.items():
        if key in low:
            return val
    return InstrumentType.OTHER


# ── Prompt template ───────────────────────────────────────────────────────

RECOMMENDATION_PROMPT = """You are an expert policy analyst extracting structured policy recommendations from a document.

TASK: Read the document text below and extract all genuine policy recommendations, policy options, implementation steps, expected outcomes, trade-offs, and actor responsibilities.

EXTRACTION TYPES:
- recommendation: A clear call to action proposed BY THIS DOCUMENT — the brief's own proposed course of action directed at a specific or implied actor. Must function as a genuine policy directive, not just normative language.
- policy_option: A distinct course of action presented for consideration, especially when the document discusses alternatives, comparisons, or competing approaches. Policy options are NOT merely multiple recommendations listed independently. Clues for policy options include: comparative language ("Option A vs Option B", "alternatively"), contrasted intervention approaches, scenario analysis, trade-off comparisons between different pathways, or explicit enumeration of possible policy responses without clear endorsement. A policy option can appear even if the document does not explicitly endorse one option over another. Do NOT classify literature summaries, framing language, or references as policy options.
- implementation_step: A concrete action step for implementing a broader recommendation.
- expected_outcome: A predicted result of a recommendation or policy.
- trade_off: An acknowledged tension or cost of a policy choice.
- actor_responsibility: Explicit assignment of responsibility to a specific actor.

CORE DEFINITION — What counts as a genuine recommendation:
A recommendation is a forward-looking policy proposal that THIS DOCUMENT advances for uptake, adoption, implementation, or support by a policymaker or other stakeholder. It must function as policy guidance — not merely as interpretation, framing, or implication.

A recommendation should connect analysis or evidence to a proposed action in a way that contributes to the document's knowledge-brokering or decision-guidance role.

CRITICAL DISTINCTION — What is NOT a recommendation:
- Background discussion, framing, or problem description that uses normative language
- Literature summaries ("studies show...", "research suggests...", "evidence indicates...")
- Analytical implications ("findings suggest that...", "this implies...")
- Observations about what others have done or proposed in the past
- References to other authors' recommendations ("Smith (2020) recommends...")
- General exhortations to "understand", "recognize", or "be aware of" something
- Descriptive statements about existing policies or regulations
- Content from the references/bibliography section

ADDITIONAL NEGATIVE EXAMPLES (do NOT extract these):
- "policymakers should understand/recognize/be aware…"
- "the findings suggest that…"
- "this implies that…"
- "future research should…"
- "it is important to consider…"
- "governments face the challenge of… and should note…"
- any advisory statement used only to frame the issue rather than propose action

A genuine recommendation must function as THIS DOCUMENT's own proposed action, not merely report or summarize normative language from other sources or background discussion.

POLICY OPTION GUIDANCE:
- Look for passages where the document lays out alternative courses of action, compares policy approaches, or presents scenarios for decision-makers.
- A policy_option should represent a genuine alternative pathway — not every bullet point or listed item is a separate option.
- If the document presents "Option 1 / Option 2" or "Approach A vs Approach B" or discusses what would happen under different policy choices, extract each distinct option as a policy_option.
- If the document discusses multiple independent recommendations without comparing or contrasting them, those are separate recommendations, NOT policy options.
- Do NOT extract background descriptions of what different countries or regions have done as policy options — those are descriptive, not prescriptive.

RULES:
- source_quote MUST be a verbatim quote (10-500 characters) copied exactly from the document text.
- page MUST be the page number where the quote appears (use the [Page N] markers).
- If the actor is not explicitly named, set actor_text_raw to null.
- If the action is vague or implied, set action_text_raw to null.
- Set instrument_type, strength, geographic_scope, timeframe, policy_domain to null when not clearly determinable.
- Only extract expected_outcome when it is clearly tied to a specific recommendation or policy option.
- Only extract trade_off when the text explicitly describes a tension, cost, drawback, or competing objective associated with a policy choice.
- Only extract policy_option when the document presents distinct alternative courses of action for comparison, not just multiple independent recommendations.
- Consider the surrounding context (e.g. paragraph or page) when judging whether a statement functions as a recommendation. Do not rely on isolated sentences alone.
- Confidence: 0.0-1.0. Use 0.8+ only for unambiguous, explicit recommendations.
- If NO genuine recommendations exist in this text, return an empty items list.
- Do NOT over-extract. Missing a borderline case is better than a false positive.
- Do NOT extract from bibliographic references, footnotes, or endnotes."""


# ── Document content helper ──────────────────────────────────────────────

class DocumentContent:
    """Broad-content representation with page markers and chunking."""

    def __init__(self, pages: List[PageText], refs_start_page: Optional[int] = None):
        if refs_start_page is not None:
            self.pages = [p for p in pages if p.page_num < refs_start_page]
        else:
            self.pages = list(pages)

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    def full_text_with_markers(self) -> str:
        """Full pre-reference text with [Page N] markers."""
        parts = []
        for p in self.pages:
            parts.append(f"[Page {p.page_num}]")
            parts.append(p.text)
        return "\n\n".join(parts)

    def page_chunks(
        self,
        max_chars: int = 30000,
        overlap_pages: int = 2,
    ) -> List[Tuple[List[PageText], str]]:
        """Split into overlapping page-window chunks with markers.

        Returns list of (pages_in_chunk, text_with_markers).
        """
        if not self.pages:
            return []

        # If fits in one chunk, return whole document
        if self.total_chars <= max_chars:
            return [(self.pages, self.full_text_with_markers())]

        chunks = []
        i = 0
        while i < len(self.pages):
            chunk_pages = []
            chunk_chars = 0
            j = i
            while j < len(self.pages) and chunk_chars + self.pages[j].char_count <= max_chars:
                chunk_pages.append(self.pages[j])
                chunk_chars += self.pages[j].char_count
                j += 1
            # Ensure at least one page per chunk
            if not chunk_pages:
                chunk_pages = [self.pages[j]]
                j += 1

            text = "\n\n".join(
                f"[Page {p.page_num}]\n{p.text}" for p in chunk_pages
            )
            chunks.append((chunk_pages, text))

            # Advance with overlap
            next_start = j - overlap_pages
            if next_start <= i:
                next_start = i + 1
            i = next_start

        return chunks


# ── Evidence verification ─────────────────────────────────────────────────

def verify_evidence(quote: str, source_text: str) -> bool:
    """Check that a quote exists in the source text (whitespace-normalized)."""
    if len(quote.strip()) < 10:
        return False
    norm_source = re.sub(r"\s+", " ", source_text.lower())
    norm_quote = re.sub(r"\s+", " ", quote.strip().lower())
    if norm_quote in norm_source:
        return True
    # Fuzzy: try first 40 chars
    prefix = norm_quote[:40]
    return prefix in norm_source


# ── Deduplication ─────────────────────────────────────────────────────────

def _deduplicate_extractions(
    items: List[PolicyExtraction],
) -> List[PolicyExtraction]:
    """Remove near-duplicate extractions based on quote overlap and page proximity."""
    if not items:
        return []
    kept: List[PolicyExtraction] = []
    for item in items:
        is_dup = False
        norm_new = re.sub(r"\s+", " ", item.source_text_raw.lower().strip())
        for existing in kept:
            norm_existing = re.sub(r"\s+", " ", existing.source_text_raw.lower().strip())
            # Check substring containment
            if norm_new in norm_existing or norm_existing in norm_new:
                is_dup = True
                break
            # Check if quotes share a long common prefix (>60 chars)
            min_len = min(len(norm_new), len(norm_existing))
            if min_len > 60 and norm_new[:60] == norm_existing[:60]:
                if abs(item.page - existing.page) <= 1:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(item)
    return kept


# ── Main extractor ────────────────────────────────────────────────────────

class RecommendationExtractor:
    """Extract policy recommendations via broad-content LLM passes."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: Dict[str, Any],
    ):
        self.llm = llm_client
        self.min_confidence = config.get("min_confidence", 0.6)
        self.max_chars_per_chunk = config.get("max_chars_per_chunk", 100000)

    def extract_recommendations(
        self,
        pages: List[PageText],
        doc_id: str,
    ) -> List[PolicyExtraction]:
        """Extract recommendations from document pages via broad-content LLM."""
        # 1. Detect and exclude reference pages
        refs_start = detect_references_start_page(pages)
        if refs_start is not None:
            logger.info(
                f"[{doc_id}] References start at page {refs_start}, "
                f"using pages before that"
            )

        # 2. Build broad-content representation
        content = DocumentContent(pages, refs_start)
        if not content.pages:
            return []

        # 3. Run LLM extraction over chunks
        chunks = content.page_chunks(max_chars=self.max_chars_per_chunk)
        logger.info(f"[{doc_id}] {len(chunks)} chunk(s) for recommendation extraction")

        all_extractions: List[PolicyExtraction] = []
        full_text = "\n".join(p.text for p in content.pages)

        for chunk_idx, (chunk_pages, chunk_text) in enumerate(chunks):
            try:
                raw_items = self._extract_from_chunk(chunk_text)
                extractions = self._validate_and_build(
                    raw_items, full_text, doc_id, len(all_extractions)
                )
                all_extractions.extend(extractions)
            except Exception as exc:
                logger.warning(
                    f"[{doc_id}] Chunk {chunk_idx + 1} extraction failed: {exc}"
                )

        # 4. Deduplicate across chunks
        deduped = _deduplicate_extractions(all_extractions)
        logger.info(
            f"[{doc_id}] {len(all_extractions)} raw -> {len(deduped)} after dedup"
        )

        # 5. Re-number
        for i, ext in enumerate(deduped, 1):
            ext.rec_id = f"{doc_id}_rec_{i:03d}"

        return deduped

    def _extract_from_chunk(self, chunk_text: str) -> List[Any]:
        """Run the LLM on one chunk and return raw RecommendationItems."""
        messages = [
            {"role": "system", "content": RECOMMENDATION_PROMPT},
            {
                "role": "user",
                "content": f"Extract policy recommendations from this document text:\n\n{chunk_text}",
            },
        ]
        result: RecommendationExtractionResponse = self.llm.structured_completion(
            messages, RecommendationExtractionResponse
        )
        return result.items

    def _validate_and_build(
        self,
        raw_items: List[Any],
        full_text: str,
        doc_id: str,
        start_counter: int,
    ) -> List[PolicyExtraction]:
        """Validate evidence, normalize, build PolicyExtraction objects."""
        extractions: List[PolicyExtraction] = []
        counter = start_counter

        for item in raw_items:
            # Skip non-recommendations returned by model
            if item.extraction_type == ExtractionType.NON_RECOMMENDATION:
                continue
            # Apply confidence threshold
            if item.confidence < self.min_confidence:
                continue

            # Evidence verification
            quote = item.source_quote.strip()[:500]
            if not verify_evidence(quote, full_text):
                logger.debug(f"[{doc_id}] Quote not verified: {quote[:60]}...")
                continue

            # Build evidence list
            evidence = []
            if len(quote) >= 10:
                evidence.append(Evidence(page=item.page, quote=quote))

            # Require evidence for recommendations and policy options
            if item.extraction_type in (
                ExtractionType.RECOMMENDATION,
                ExtractionType.POLICY_OPTION,
            ) and not evidence:
                continue

            counter += 1
            extractions.append(PolicyExtraction(
                rec_id=f"{doc_id}_rec_{counter:03d}",
                extraction_type=item.extraction_type,
                confidence=item.confidence,
                source_text_raw=quote,
                source_section=None,
                page=item.page,
                actor_text_raw=item.actor_text_raw,
                actor_type_normalized=_normalize_actor(item.actor_text_raw),
                action_text_raw=item.action_text_raw,
                target_text_raw=item.target_text_raw,
                instrument_type=_normalize_instrument(item.instrument_type),
                strength=_normalize_strength(item.strength),
                expected_outcomes=item.expected_outcomes,
                implementation_steps=item.implementation_steps,
                trade_offs=item.trade_offs,
                evidence=evidence,
            ))

        return extractions
