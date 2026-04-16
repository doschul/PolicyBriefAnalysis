"""Tests for broad-content recommendation extraction: DocumentContent, evidence
verification, deduplication, normalization, references exclusion."""

import pytest

from src.policybrief.models import (
    ActorType,
    Evidence,
    ExtractionType,
    InstrumentType,
    PageText,
    PolicyExtraction,
    RecommendationExtractionResponse,
    RecommendationItem,
    RecommendationStrength,
)
from src.policybrief.recommendation_extractor import (
    DocumentContent,
    RecommendationExtractor,
    _deduplicate_extractions,
    _normalize_actor,
    _normalize_instrument,
    _normalize_strength,
    detect_references_start_page,
    verify_evidence,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _pages(texts):
    return [
        PageText(page_num=i + 1, text=t, char_count=len(t), word_count=len(t.split()))
        for i, t in enumerate(texts)
    ]


# ── References page detection ─────────────────────────────────────────────

class TestReferencesDetection:
    def test_finds_references_heading(self):
        pages = _pages([
            "Introduction text here.",
            "Some policy content.",
            "References\nSmith (2020). Title. Journal.",
        ])
        assert detect_references_start_page(pages) == 3

    def test_finds_bibliography(self):
        pages = _pages([
            "Body text.",
            "Bibliography\nSome ref.",
        ])
        assert detect_references_start_page(pages) == 2

    def test_no_references(self):
        pages = _pages(["Some text.", "More text."])
        assert detect_references_start_page(pages) is None

    def test_case_insensitive(self):
        pages = _pages(["Body.", "REFERENCES\nRef list."])
        assert detect_references_start_page(pages) == 2

    def test_works_cited(self):
        pages = _pages(["Body.", "Works Cited\nRef list."])
        assert detect_references_start_page(pages) == 2

    def test_references_early_in_doc_ignored(self):
        """References heading in the front of the document is ignored."""
        pages = _pages([
            "References\nEarly refs.",
            "Body content here.",
            "More body.",
        ])
        assert detect_references_start_page(pages) is None


# ── DocumentContent ───────────────────────────────────────────────────────

class TestDocumentContent:
    def test_full_text_with_markers(self):
        pages = _pages(["Page one content.", "Page two content."])
        dc = DocumentContent(pages)
        text = dc.full_text_with_markers()
        assert "[Page 1]" in text
        assert "[Page 2]" in text
        assert "Page one content." in text

    def test_total_chars(self):
        pages = _pages(["Hello", "World"])
        dc = DocumentContent(pages)
        assert dc.total_chars == 10

    def test_refs_excluded(self):
        pages = _pages(["Body text.", "References\nSmith 2020."])
        dc = DocumentContent(pages, refs_start_page=2)
        assert len(dc.pages) == 1
        assert dc.pages[0].page_num == 1

    def test_single_chunk_for_small_doc(self):
        pages = _pages(["Short doc content."])
        dc = DocumentContent(pages)
        chunks = dc.page_chunks(max_chars=5000)
        assert len(chunks) == 1

    def test_multiple_chunks_for_large_doc(self):
        # Create pages that exceed max_chars
        big_text = "x" * 5000
        pages = _pages([big_text] * 10)  # 50k total chars
        dc = DocumentContent(pages)
        chunks = dc.page_chunks(max_chars=15000)
        assert len(chunks) >= 2

    def test_empty_pages(self):
        dc = DocumentContent([])
        assert dc.total_chars == 0
        assert dc.page_chunks() == []

    def test_chunk_overlap(self):
        """Chunks should have overlapping pages."""
        pages = _pages(["x" * 4000] * 10)
        dc = DocumentContent(pages)
        chunks = dc.page_chunks(max_chars=12000, overlap_pages=2)
        if len(chunks) >= 2:
            # Get page numbers from each chunk
            chunk1_pages = {p.page_num for p, _ in [(p, None) for p in chunks[0][0]]}
            chunk2_pages = {p.page_num for p, _ in [(p, None) for p in chunks[1][0]]}
            overlap = chunk1_pages & chunk2_pages
            assert len(overlap) >= 1  # At least some overlap


# ── Evidence verification ─────────────────────────────────────────────────

class TestEvidenceVerification:
    def test_exact_match(self):
        assert verify_evidence(
            "Governments should strengthen monitoring",
            "Governments should strengthen monitoring systems."
        )

    def test_whitespace_normalized(self):
        assert verify_evidence(
            "Governments  should   strengthen",
            "Governments should strengthen monitoring."
        )

    def test_case_insensitive(self):
        assert verify_evidence(
            "GOVERNMENTS should STRENGTHEN",
            "governments should strengthen monitoring."
        )

    def test_prefix_fallback(self):
        assert verify_evidence(
            "Governments should strengthen monitoring and improve transparency in forest governance",
            "Governments should strengthen monitoring systems."
        )

    def test_no_match(self):
        assert not verify_evidence(
            "A completely unrelated quote from a different document",
            "Forests cover 30% of the land area."
        )

    def test_short_quote_rejected(self):
        assert not verify_evidence("short", "short text here.")


# ── Deduplication ─────────────────────────────────────────────────────────

class TestDeduplication:
    def _make_extraction(self, text, page=1, rec_id="r1"):
        return PolicyExtraction(
            rec_id=rec_id,
            extraction_type=ExtractionType.TRADE_OFF,
            confidence=0.8,
            source_text_raw=text,
            page=page,
            evidence=[],
        )

    def test_exact_duplicates_removed(self):
        items = [
            self._make_extraction("Governments should strengthen monitoring systems"),
            self._make_extraction("Governments should strengthen monitoring systems"),
        ]
        result = _deduplicate_extractions(items)
        assert len(result) == 1

    def test_substring_containment(self):
        items = [
            self._make_extraction("Governments should strengthen monitoring systems for all forests"),
            self._make_extraction("Governments should strengthen monitoring systems"),
        ]
        result = _deduplicate_extractions(items)
        assert len(result) == 1

    def test_distinct_kept(self):
        items = [
            self._make_extraction("Governments should strengthen monitoring"),
            self._make_extraction("Companies must reduce emissions significantly"),
        ]
        result = _deduplicate_extractions(items)
        assert len(result) == 2

    def test_empty_list(self):
        assert _deduplicate_extractions([]) == []

    def test_prefix_match_same_page(self):
        # Long quotes with same 60-char prefix on adjacent pages → dedup
        base = "Governments should strengthen forest monitoring systems and " + "x" * 50
        items = [
            self._make_extraction(base + " part one", page=1),
            self._make_extraction(base + " part two", page=1),
        ]
        result = _deduplicate_extractions(items)
        assert len(result) == 1


# ── Normalisation helpers ─────────────────────────────────────────────────

class TestNormalization:
    def test_strength_must(self):
        assert _normalize_strength("must") == RecommendationStrength.MUST

    def test_strength_should(self):
        assert _normalize_strength("should") == RecommendationStrength.SHOULD

    def test_strength_consider(self):
        assert _normalize_strength("consider") == RecommendationStrength.CONSIDER

    def test_strength_none(self):
        assert _normalize_strength(None) is None

    def test_strength_unknown(self):
        assert _normalize_strength("xyzzy") == RecommendationStrength.UNSPECIFIED

    def test_actor_government(self):
        assert _normalize_actor("national government") == ActorType.GOVERNMENT

    def test_actor_eu(self):
        assert _normalize_actor("European Commission") == ActorType.EU_INSTITUTIONS

    def test_actor_none(self):
        assert _normalize_actor(None) is None

    def test_instrument_regulation(self):
        assert _normalize_instrument("regulation") == InstrumentType.REGULATION

    def test_instrument_subsidy(self):
        assert _normalize_instrument("subsidy") == InstrumentType.SUBSIDY

    def test_instrument_none(self):
        assert _normalize_instrument(None) is None

    def test_instrument_unknown(self):
        assert _normalize_instrument("xyzzy") == InstrumentType.OTHER


# ── Extractor with mock LLM ─────────────────────────────────────────────

class FakeLLMEmpty:
    """Mock LLM that returns no recommendations."""
    def structured_completion(self, messages, response_model):
        return RecommendationExtractionResponse(items=[])


class FakeLLMWithRecs:
    """Mock LLM that returns a recommendation using text from the document."""
    def __init__(self, quote="Governments should strengthen forest monitoring systems"):
        self.quote = quote

    def structured_completion(self, messages, response_model):
        return RecommendationExtractionResponse(items=[
            RecommendationItem(
                extraction_type=ExtractionType.RECOMMENDATION,
                confidence=0.85,
                source_quote=self.quote,
                page=1,
                actor_text_raw="Governments",
                action_text_raw="strengthen forest monitoring systems",
                strength="should",
            ),
        ])


class TestRecommendationExtractor:
    def test_empty_document(self):
        extractor = RecommendationExtractor(
            llm_client=FakeLLMEmpty(), config={"min_confidence": 0.6},
        )
        result = extractor.extract_recommendations([], "test_doc")
        assert result == []

    def test_extracts_with_valid_evidence(self):
        quote = "Governments should strengthen forest monitoring systems"
        pages = _pages([f"{quote} for all tropical regions."])
        extractor = RecommendationExtractor(
            llm_client=FakeLLMWithRecs(quote=quote),
            config={"min_confidence": 0.6},
        )
        result = extractor.extract_recommendations(pages, "test_doc")
        assert len(result) >= 1
        assert result[0].extraction_type == ExtractionType.RECOMMENDATION
        assert result[0].actor_type_normalized == ActorType.GOVERNMENT
        assert result[0].strength == RecommendationStrength.SHOULD

    def test_evidence_not_in_text_rejected(self):
        """Recommendations with quotes not found in the document are filtered out."""
        pages = _pages(["Forests cover 30% of the total land area globally."])
        extractor = RecommendationExtractor(
            llm_client=FakeLLMWithRecs(
                quote="A completely different quote not in the document"
            ),
            config={"min_confidence": 0.6},
        )
        result = extractor.extract_recommendations(pages, "test_doc")
        assert len(result) == 0

    def test_low_confidence_filtered(self):
        """Recommendations below min_confidence are filtered out."""
        class LowConfLLM:
            def structured_completion(self, messages, response_model):
                return RecommendationExtractionResponse(items=[
                    RecommendationItem(
                        extraction_type=ExtractionType.RECOMMENDATION,
                        confidence=0.3,
                        source_quote="Governments should strengthen forest monitoring",
                        page=1,
                    ),
                ])

        pages = _pages(["Governments should strengthen forest monitoring systems."])
        extractor = RecommendationExtractor(
            llm_client=LowConfLLM(),
            config={"min_confidence": 0.6},
        )
        result = extractor.extract_recommendations(pages, "test_doc")
        assert len(result) == 0

    def test_policy_option_true_positive(self):
        """A genuine policy option with verified evidence is extracted."""
        option_text = "Option A proposes a carbon tax while Option B proposes cap-and-trade"

        class PolicyOptionLLM:
            def structured_completion(self, messages, response_model):
                return RecommendationExtractionResponse(items=[
                    RecommendationItem(
                        extraction_type=ExtractionType.POLICY_OPTION,
                        confidence=0.8,
                        source_quote=option_text,
                        page=1,
                        action_text_raw="carbon tax or cap-and-trade",
                    ),
                ])

        pages = _pages([f"The brief considers two pathways: {option_text} for emission reduction."])
        extractor = RecommendationExtractor(
            llm_client=PolicyOptionLLM(), config={"min_confidence": 0.6},
        )
        result = extractor.extract_recommendations(pages, "test_doc")
        assert len(result) == 1
        assert result[0].extraction_type == ExtractionType.POLICY_OPTION

    def test_policy_option_false_positive_excluded(self):
        """A literature summary should NOT be extracted as a policy option."""
        lit_quote = "Smith (2020) recommended increasing protected areas"

        class FalsePositiveLLM:
            def structured_completion(self, messages, response_model):
                return RecommendationExtractionResponse(items=[
                    RecommendationItem(
                        extraction_type=ExtractionType.POLICY_OPTION,
                        confidence=0.45,  # below threshold
                        source_quote=lit_quote,
                        page=1,
                    ),
                ])

        pages = _pages([f"In the literature, {lit_quote} as part of a review."])
        extractor = RecommendationExtractor(
            llm_client=FalsePositiveLLM(), config={"min_confidence": 0.6},
        )
        result = extractor.extract_recommendations(pages, "test_doc")
        assert len(result) == 0  # filtered by confidence threshold

    def test_non_recommendation_type_skipped(self):
        """ExtractionType.NON_RECOMMENDATION items are skipped."""
        class NonRecLLM:
            def structured_completion(self, messages, response_model):
                return RecommendationExtractionResponse(items=[
                    RecommendationItem(
                        extraction_type=ExtractionType.NON_RECOMMENDATION,
                        confidence=0.9,
                        source_quote="This is just background information about forests",
                        page=1,
                    ),
                ])

        pages = _pages(["This is just background information about forests and ecosystems."])
        extractor = RecommendationExtractor(
            llm_client=NonRecLLM(),
            config={"min_confidence": 0.6},
        )
        result = extractor.extract_recommendations(pages, "test_doc")
        assert len(result) == 0

    def test_references_pages_excluded(self):
        """Pages after references heading are not sent to LLM."""
        pages = _pages([
            "Governments should strengthen forest monitoring systems urgently.",
            "More policy content here for the analysis pipeline.",
            "More policy content here for the analysis pipeline.",
            "More policy content here for the analysis pipeline.",
            "References\nSmith (2020). Title. Journal. Governments should act.",
        ])
        refs_start = detect_references_start_page(pages)
        assert refs_start == 5
        dc = DocumentContent(pages, refs_start)
        assert all(p.page_num < 5 for p in dc.pages)

    def test_rec_ids_renumbered(self):
        """After dedup, rec_ids should be sequential."""
        quote = "Governments should strengthen forest monitoring systems"
        pages = _pages([f"{quote} for all regions across the globe."])
        extractor = RecommendationExtractor(
            llm_client=FakeLLMWithRecs(quote=quote),
            config={"min_confidence": 0.6},
        )
        result = extractor.extract_recommendations(pages, "doc1")
        if result:
            assert result[0].rec_id == "doc1_rec_001"
