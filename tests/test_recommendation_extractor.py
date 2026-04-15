"""
Tests for the rewritten section-aware recommendation extractor.

Covers:
  - candidate generation and filtering
  - section-map targeting and exclusion
  - citation rejection
  - prescriptive-language detection
  - LLM classification integration (mocked)
  - functional validation (actor/action/target)
  - normalization helpers
  - end-to-end pipeline integration
  - edge cases (sparse/empty documents)
"""

import pytest
from unittest.mock import MagicMock, patch

from src.policybrief.models import (
    ActorType,
    CandidateClassification,
    CandidateClassificationBatch,
    CandidateSpan,
    DocumentSection,
    DocumentSectionMap,
    Evidence,
    ExtractionType,
    InstrumentType,
    PageText,
    PolicyExtraction,
    RecommendationStrength,
    SectionLabel,
)
from src.policybrief.recommendation_extractor import (
    RecommendationExtractor,
    _EXCLUDED_SECTIONS,
    _TARGET_SECTIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_client():
    """LLM client mock that returns controllable classifications."""
    return MagicMock()


@pytest.fixture
def enums_config():
    return {
        "instrument_types": ["regulation", "subsidy", "tax", "other"],
        "geographic_scopes": ["local", "national", "international", "unspecified"],
        "timeframes": ["immediate", "short_term", "medium_term", "long_term", "unspecified"],
        "strengths": ["must", "should", "could", "may", "unspecified"],
        "actor_types": ["government", "private_sector", "civil_society", "unspecified"],
        "policy_domains": ["climate_change", "environment", "other"],
    }


@pytest.fixture
def extractor(mock_llm_client, enums_config):
    return RecommendationExtractor(
        llm_client=mock_llm_client,
        enums_config=enums_config,
        min_confidence=0.6,
        max_recommendations=10,
    )


def _make_page(page_num: int, text: str) -> PageText:
    return PageText(
        page_num=page_num,
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
    )


def _make_section_map(*sections) -> DocumentSectionMap:
    return DocumentSectionMap(
        sections=list(sections),
        detection_method="text_heuristic",
    )


def _make_section(
    label, start_page, end_page, raw_title=None, confidence=0.8
) -> DocumentSection:
    return DocumentSection(
        raw_title=raw_title or (label.value if label else "Unknown"),
        normalized_label=label,
        start_page=start_page,
        end_page=end_page,
        confidence=confidence,
        rule_source="text_heuristic",
    )


# ---------------------------------------------------------------------------
# Test: Section targeting
# ---------------------------------------------------------------------------

class TestSectionTargeting:
    """Verify that section map is used to include/exclude pages."""

    def test_excludes_references_section(self, extractor):
        pages = [
            _make_page(1, "Introduction text here with some content about policy."),
            _make_page(2, "Governments should implement carbon pricing mechanisms to reduce emissions."),
            _make_page(3, "Smith (2020) recommends that governments should invest in renewables. Jones et al. (2019) proposed similar measures."),
        ]
        section_map = _make_section_map(
            _make_section(SectionLabel.INTRODUCTION, 1, 1),
            _make_section(SectionLabel.RECOMMENDATIONS, 2, 2),
            _make_section(SectionLabel.REFERENCES, 3, 3),
        )

        target_pages, page_section = extractor._identify_target_pages(pages, section_map)

        # Page 3 (References) should not be in target pages
        target_nums = [p.page_num for p in target_pages]
        assert 3 not in target_nums
        assert 2 in target_nums
        # Page 3 should be marked as REFERENCES in the mapping
        assert page_section.get(3) == SectionLabel.REFERENCES

    def test_includes_all_target_sections(self, extractor):
        pages = [_make_page(i, f"Content on page {i}.") for i in range(1, 8)]
        section_map = _make_section_map(
            _make_section(SectionLabel.EXECUTIVE_SUMMARY, 1, 1),
            _make_section(SectionLabel.KEY_MESSAGES, 2, 2),
            _make_section(SectionLabel.POLICY_OPTIONS, 3, 3),
            _make_section(SectionLabel.RECOMMENDATIONS, 4, 4),
            _make_section(SectionLabel.IMPLEMENTATION, 5, 5),
            _make_section(SectionLabel.CONCLUSION, 6, 6),
            _make_section(SectionLabel.REFERENCES, 7, 7),
        )

        target_pages, _ = extractor._identify_target_pages(pages, section_map)
        target_nums = {p.page_num for p in target_pages}

        # Pages 1-6 should be included, page 7 excluded
        assert target_nums == {1, 2, 3, 4, 5, 6}

    def test_fallback_without_section_map(self, extractor):
        pages = [_make_page(i, f"Content {i}.") for i in range(1, 4)]
        target_pages, _ = extractor._identify_target_pages(pages, None)
        assert len(target_pages) == 3

    def test_unlabeled_sections_included(self, extractor):
        """Sections without a normalized label are included (could be body text)."""
        pages = [_make_page(1, "Content."), _make_page(2, "More content.")]
        section_map = _make_section_map(
            _make_section(None, 1, 1, raw_title="Discussion"),
            _make_section(SectionLabel.REFERENCES, 2, 2),
        )
        target_pages, _ = extractor._identify_target_pages(pages, section_map)
        assert len(target_pages) == 1
        assert target_pages[0].page_num == 1


# ---------------------------------------------------------------------------
# Test: Candidate generation and filtering
# ---------------------------------------------------------------------------

class TestCandidateGeneration:
    """Test sentence splitting, citation rejection, prescriptive tagging."""

    def test_prescriptive_language_detected(self, extractor):
        pages = [
            _make_page(
                1,
                "Governments should invest in renewable energy. "
                "The weather was nice today."
            )
        ]
        page_section = {1: SectionLabel.RECOMMENDATIONS}

        candidates = extractor._generate_candidates(pages, page_section)

        # Only the prescriptive sentence should survive
        assert len(candidates) == 1
        assert "should invest" in candidates[0].text
        assert candidates[0].has_prescriptive_language is True

    def test_citation_heavy_spans_rejected(self, extractor):
        text = (
            "According to Smith et al. (2019), governments should invest in "
            "renewable energy (Jones, 2020). This finding was confirmed by "
            "further research [3, 4]."
        )
        pages = [_make_page(1, text)]
        page_section = {1: SectionLabel.RECOMMENDATIONS}

        candidates = extractor._generate_candidates(pages, page_section)

        # Citation-heavy spans should be filtered out
        for c in candidates:
            # None of the surviving candidates should be citation-heavy
            assert not extractor._is_citation_heavy(c.text)

    def test_short_spans_rejected(self, extractor):
        pages = [_make_page(1, "Do this. Also that. Very short.")]
        page_section = {1: SectionLabel.RECOMMENDATIONS}

        candidates = extractor._generate_candidates(pages, page_section)
        # All spans are < 20 chars → no candidates
        assert len(candidates) == 0

    def test_excluded_section_pages_skipped(self, extractor):
        pages = [
            _make_page(1, "Governments should implement carbon pricing."),
        ]
        # Mark page 1 as REFERENCES → excluded
        page_section = {1: SectionLabel.REFERENCES}

        candidates = extractor._generate_candidates(pages, page_section)
        assert len(candidates) == 0


# ---------------------------------------------------------------------------
# Test: Citation detection
# ---------------------------------------------------------------------------

class TestCitationDetection:

    def test_author_year_citation(self, extractor):
        assert extractor._is_citation_heavy(
            "According to Smith (2020), we should act. Jones et al. (2019) agree."
        )

    def test_numbered_citation(self, extractor):
        assert extractor._is_citation_heavy(
            "Previous work [1, 2] suggests that governments should act [3]."
        )

    def test_reference_list_entry(self, extractor):
        assert extractor._is_citation_heavy(
            "1. Smith, J. (2020). Policy recommendations for climate action."
        )

    def test_clean_sentence_not_rejected(self, extractor):
        assert not extractor._is_citation_heavy(
            "Governments should implement carbon pricing mechanisms to reduce emissions."
        )

    def test_single_citation_not_rejected(self, extractor):
        # One citation hit is below threshold (needs ≥2)
        assert not extractor._is_citation_heavy(
            "As noted by Smith (2020), action is needed."
        )


# ---------------------------------------------------------------------------
# Test: Prescriptive language detection
# ---------------------------------------------------------------------------

class TestPrescriptiveDetection:

    def test_should(self, extractor):
        has, cues = extractor._detect_prescriptive("Governments should act now.")
        assert has is True
        assert any("should" in c for c in cues)

    def test_must(self, extractor):
        has, _ = extractor._detect_prescriptive("We must reform the system.")
        assert has is True

    def test_recommend(self, extractor):
        has, _ = extractor._detect_prescriptive("We recommend investing in infrastructure.")
        assert has is True

    def test_no_prescriptive(self, extractor):
        has, cues = extractor._detect_prescriptive(
            "The study found that carbon levels are rising."
        )
        assert has is False
        assert cues == []

    def test_call_for(self, extractor):
        has, _ = extractor._detect_prescriptive(
            "This brief calls for immediate regulatory reform."
        )
        assert has is True


# ---------------------------------------------------------------------------
# Test: LLM classification integration (mocked)
# ---------------------------------------------------------------------------

class TestClassificationIntegration:
    """Test end-to-end extraction with mocked LLM responses."""

    def test_recommendation_in_executive_summary(self, extractor, mock_llm_client):
        """Recommendation in ExecutiveSummary section is extracted."""
        pages = [
            _make_page(
                1,
                "Governments should implement a carbon tax to reduce greenhouse gas emissions effectively."
            ),
        ]
        section_map = _make_section_map(
            _make_section(SectionLabel.EXECUTIVE_SUMMARY, 1, 1),
        )

        mock_llm_client.structured_completion.return_value = CandidateClassificationBatch(
            classifications=[
                CandidateClassification(
                    extraction_type=ExtractionType.RECOMMENDATION,
                    confidence=0.9,
                    actor_text_raw="Governments",
                    action_text_raw="implement a carbon tax",
                    target_text_raw="greenhouse gas emissions",
                    instrument_type="tax",
                    strength="should",
                )
            ]
        )

        results = extractor.extract_recommendations(pages, section_map)

        assert len(results) == 1
        r = results[0]
        assert r.extraction_type == ExtractionType.RECOMMENDATION
        assert r.confidence == 0.9
        assert r.source_section == SectionLabel.EXECUTIVE_SUMMARY
        assert r.actor_text_raw == "Governments"
        assert r.actor_type_normalized == ActorType.GOVERNMENT
        assert r.action_text_raw == "implement a carbon tax"
        assert r.instrument_type == InstrumentType.TAX
        assert r.strength == RecommendationStrength.SHOULD
        assert len(r.evidence) == 1

    def test_policy_option_not_recommendation(self, extractor, mock_llm_client):
        """Policy option classified separately from recommendation."""
        pages = [
            _make_page(
                1,
                "Option A proposes a cap-and-trade system for industrial emissions."
            ),
        ]
        section_map = _make_section_map(
            _make_section(SectionLabel.POLICY_OPTIONS, 1, 1),
        )

        mock_llm_client.structured_completion.return_value = CandidateClassificationBatch(
            classifications=[
                CandidateClassification(
                    extraction_type=ExtractionType.POLICY_OPTION,
                    confidence=0.85,
                    actor_text_raw=None,
                    action_text_raw="cap-and-trade system",
                    target_text_raw="industrial emissions",
                )
            ]
        )

        results = extractor.extract_recommendations(pages, section_map)

        assert len(results) == 1
        assert results[0].extraction_type == ExtractionType.POLICY_OPTION
        assert results[0].actor_type_normalized is None

    def test_references_should_language_rejected(self, extractor, mock_llm_client):
        """Prescriptive language in references section is never sent to LLM."""
        pages = [
            _make_page(1, "Introduction to the brief with some background."),
            _make_page(
                2,
                "Smith (2020) recommends that governments should invest heavily. "
                "Jones et al. (2019) proposed a framework for action."
            ),
        ]
        section_map = _make_section_map(
            _make_section(SectionLabel.INTRODUCTION, 1, 1),
            _make_section(SectionLabel.REFERENCES, 2, 2),
        )

        results = extractor.extract_recommendations(pages, section_map)

        # LLM should never be called because:
        # - Page 1 (intro) has no prescriptive language
        # - Page 2 (references) is excluded
        mock_llm_client.structured_completion.assert_not_called()
        assert results == []

    def test_recommendation_with_explicit_actor(self, extractor, mock_llm_client):
        pages = [
            _make_page(
                1,
                "The European Commission should establish a monitoring framework for biodiversity."
            ),
        ]
        section_map = _make_section_map(
            _make_section(SectionLabel.RECOMMENDATIONS, 1, 1),
        )

        mock_llm_client.structured_completion.return_value = CandidateClassificationBatch(
            classifications=[
                CandidateClassification(
                    extraction_type=ExtractionType.RECOMMENDATION,
                    confidence=0.92,
                    actor_text_raw="The European Commission",
                    action_text_raw="establish a monitoring framework",
                    target_text_raw="biodiversity",
                    strength="should",
                )
            ]
        )

        results = extractor.extract_recommendations(pages, section_map)

        assert len(results) == 1
        assert results[0].actor_text_raw == "The European Commission"
        assert results[0].actor_type_normalized == ActorType.EU_INSTITUTIONS

    def test_recommendation_without_actor_stays_null(self, extractor, mock_llm_client):
        pages = [
            _make_page(
                1,
                "Carbon pricing mechanisms should be strengthened to meet climate targets."
            ),
        ]
        section_map = _make_section_map(
            _make_section(SectionLabel.RECOMMENDATIONS, 1, 1),
        )

        mock_llm_client.structured_completion.return_value = CandidateClassificationBatch(
            classifications=[
                CandidateClassification(
                    extraction_type=ExtractionType.RECOMMENDATION,
                    confidence=0.8,
                    actor_text_raw=None,
                    action_text_raw="be strengthened",
                    target_text_raw="carbon pricing mechanisms",
                    strength="should",
                )
            ]
        )

        results = extractor.extract_recommendations(pages, section_map)

        assert len(results) == 1
        assert results[0].actor_text_raw is None
        assert results[0].actor_type_normalized is None

    def test_trade_off_extraction(self, extractor, mock_llm_client):
        pages = [
            _make_page(
                1,
                "While carbon taxes should reduce emissions, they may increase energy costs for low-income households."
            ),
        ]
        section_map = _make_section_map(
            _make_section(SectionLabel.POLICY_OPTIONS, 1, 1),
        )

        mock_llm_client.structured_completion.return_value = CandidateClassificationBatch(
            classifications=[
                CandidateClassification(
                    extraction_type=ExtractionType.TRADE_OFF,
                    confidence=0.75,
                    trade_offs=["increase energy costs for low-income households"],
                )
            ]
        )

        results = extractor.extract_recommendations(pages, section_map)

        assert len(results) == 1
        assert results[0].extraction_type == ExtractionType.TRADE_OFF
        assert "energy costs" in results[0].trade_offs[0]

    def test_implementation_step_extraction(self, extractor, mock_llm_client):
        pages = [
            _make_page(
                1,
                "First, governments should establish a national emissions registry before implementing the cap-and-trade system."
            ),
        ]
        section_map = _make_section_map(
            _make_section(SectionLabel.IMPLEMENTATION, 1, 1),
        )

        mock_llm_client.structured_completion.return_value = CandidateClassificationBatch(
            classifications=[
                CandidateClassification(
                    extraction_type=ExtractionType.IMPLEMENTATION_STEP,
                    confidence=0.82,
                    actor_text_raw="governments",
                    action_text_raw="establish a national emissions registry",
                    implementation_steps=["establish a national emissions registry"],
                )
            ]
        )

        results = extractor.extract_recommendations(pages, section_map)

        assert len(results) == 1
        assert results[0].extraction_type == ExtractionType.IMPLEMENTATION_STEP
        assert len(results[0].implementation_steps) == 1

    def test_low_confidence_rejected(self, extractor, mock_llm_client):
        pages = [
            _make_page(
                1,
                "Perhaps something should be done about climate change in the future."
            ),
        ]
        section_map = _make_section_map(
            _make_section(SectionLabel.CONCLUSION, 1, 1),
        )

        mock_llm_client.structured_completion.return_value = CandidateClassificationBatch(
            classifications=[
                CandidateClassification(
                    extraction_type=ExtractionType.RECOMMENDATION,
                    confidence=0.3,  # Below threshold
                    action_text_raw="be done",
                )
            ]
        )

        results = extractor.extract_recommendations(pages, section_map)
        assert results == []

    def test_recommendation_without_action_rejected(self, extractor, mock_llm_client):
        pages = [
            _make_page(
                1,
                "Governments should consider the implications of this finding carefully."
            ),
        ]
        section_map = _make_section_map(
            _make_section(SectionLabel.RECOMMENDATIONS, 1, 1),
        )

        mock_llm_client.structured_completion.return_value = CandidateClassificationBatch(
            classifications=[
                CandidateClassification(
                    extraction_type=ExtractionType.RECOMMENDATION,
                    confidence=0.7,
                    actor_text_raw="Governments",
                    action_text_raw=None,  # No action identified
                    target_text_raw=None,
                )
            ]
        )

        results = extractor.extract_recommendations(pages, section_map)
        assert results == []


# ---------------------------------------------------------------------------
# Test: Normalization helpers
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_actor_normalization_government(self):
        assert RecommendationExtractor._normalize_actor("governments") == ActorType.GOVERNMENT
        assert RecommendationExtractor._normalize_actor("Policymakers") == ActorType.GOVERNMENT

    def test_actor_normalization_eu(self):
        assert RecommendationExtractor._normalize_actor("European Commission") == ActorType.EU_INSTITUTIONS

    def test_actor_normalization_unknown(self):
        assert RecommendationExtractor._normalize_actor("local farmers") is None

    def test_actor_normalization_null(self):
        assert RecommendationExtractor._normalize_actor(None) is None

    def test_instrument_normalization(self):
        assert RecommendationExtractor._normalize_instrument("tax") == InstrumentType.TAX
        assert RecommendationExtractor._normalize_instrument("regulatory framework") == InstrumentType.REGULATION
        assert RecommendationExtractor._normalize_instrument("unknown widget") is None
        assert RecommendationExtractor._normalize_instrument(None) is None

    def test_strength_normalization(self):
        assert RecommendationExtractor._normalize_strength("should") == RecommendationStrength.SHOULD
        assert RecommendationExtractor._normalize_strength("must") == RecommendationStrength.MUST
        assert RecommendationExtractor._normalize_strength("maybe") is None
        assert RecommendationExtractor._normalize_strength(None) is None


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_document(self, extractor, mock_llm_client):
        results = extractor.extract_recommendations([], None)
        assert results == []
        mock_llm_client.structured_completion.assert_not_called()

    def test_sparse_document_no_prescriptive(self, extractor, mock_llm_client):
        pages = [
            _make_page(1, "This is a very sparse document with little content."),
        ]
        section_map = _make_section_map(
            _make_section(SectionLabel.INTRODUCTION, 1, 1),
        )

        results = extractor.extract_recommendations(pages, section_map)
        assert results == []
        mock_llm_client.structured_completion.assert_not_called()

    def test_llm_failure_returns_empty(self, extractor, mock_llm_client):
        pages = [
            _make_page(
                1,
                "Governments should implement carbon pricing to reduce emissions."
            ),
        ]
        section_map = _make_section_map(
            _make_section(SectionLabel.RECOMMENDATIONS, 1, 1),
        )

        mock_llm_client.structured_completion.side_effect = Exception("API error")

        results = extractor.extract_recommendations(pages, section_map)
        assert results == []

    def test_max_recommendations_cap(self, extractor, mock_llm_client):
        extractor.max_recommendations = 2

        # Create 5 candidate sentences
        text = " ".join(
            f"Governments should implement policy measure {i} to address issue {i}."
            for i in range(1, 6)
        )
        pages = [_make_page(1, text)]
        section_map = _make_section_map(
            _make_section(SectionLabel.RECOMMENDATIONS, 1, 1),
        )

        # LLM returns all as recommendations
        mock_llm_client.structured_completion.return_value = CandidateClassificationBatch(
            classifications=[
                CandidateClassification(
                    extraction_type=ExtractionType.RECOMMENDATION,
                    confidence=0.9,
                    action_text_raw=f"implement policy measure {i}",
                    target_text_raw=f"issue {i}",
                    strength="should",
                )
                for i in range(1, 6)
            ]
        )

        results = extractor.extract_recommendations(pages, section_map)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# Test: Sentence splitting
# ---------------------------------------------------------------------------

class TestSentenceSplitting:

    def test_basic_split(self):
        text = "First sentence. Second sentence. Third sentence."
        sents = RecommendationExtractor._split_sentences(text)
        assert len(sents) == 3

    def test_bullet_list(self):
        text = "Header text:\n- First item should be done.\n- Second item must happen."
        sents = RecommendationExtractor._split_sentences(text)
        assert len(sents) >= 2

    def test_empty_text(self):
        assert RecommendationExtractor._split_sentences("") == []
        assert RecommendationExtractor._split_sentences(None) == []


# ---------------------------------------------------------------------------
# Test: Model serialization
# ---------------------------------------------------------------------------

class TestPolicyExtractionModel:

    def test_recommendation_requires_evidence(self):
        """Recommendation type requires at least one evidence quote."""
        with pytest.raises(Exception):
            PolicyExtraction(
                extraction_type=ExtractionType.RECOMMENDATION,
                confidence=0.9,
                source_text_raw="Test text here",
                page=1,
                evidence=[],  # Empty — should fail
            )

    def test_trade_off_no_evidence_ok(self):
        """Non-recommendation types don't require evidence."""
        ext = PolicyExtraction(
            extraction_type=ExtractionType.TRADE_OFF,
            confidence=0.7,
            source_text_raw="Carbon taxes may increase energy costs.",
            page=1,
            evidence=[],
            trade_offs=["increase energy costs"],
        )
        assert ext.extraction_type == ExtractionType.TRADE_OFF

    def test_full_serialization_roundtrip(self):
        ext = PolicyExtraction(
            rec_id="doc_001_rec_01",
            extraction_type=ExtractionType.RECOMMENDATION,
            confidence=0.9,
            source_text_raw="Governments should implement carbon pricing.",
            source_section=SectionLabel.RECOMMENDATIONS,
            page=5,
            actor_text_raw="Governments",
            actor_type_normalized=ActorType.GOVERNMENT,
            action_text_raw="implement carbon pricing",
            target_text_raw="greenhouse gas emissions",
            instrument_type=InstrumentType.TAX,
            strength=RecommendationStrength.SHOULD,
            expected_outcomes=["reduced emissions"],
            trade_offs=["higher energy costs"],
            evidence=[Evidence(page=5, quote="Governments should implement carbon pricing.")],
        )

        data = ext.model_dump()
        restored = PolicyExtraction.model_validate(data)

        assert restored.extraction_type == ExtractionType.RECOMMENDATION
        assert restored.actor_type_normalized == ActorType.GOVERNMENT
        assert restored.source_section == SectionLabel.RECOMMENDATIONS
        assert len(restored.evidence) == 1
        assert restored.trade_offs == ["higher energy costs"]


if __name__ == "__main__":
    pytest.main([__file__])
