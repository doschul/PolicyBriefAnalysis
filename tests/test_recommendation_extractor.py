"""Tests for recommendation extraction: references exclusion, filters, normalisation."""

import pytest

from src.policybrief.models import (
    ActorType,
    CandidateClassification,
    ExtractionType,
    InstrumentType,
    PageText,
    RecommendationStrength,
)
from src.policybrief.recommendation_extractor import (
    RecommendationExtractor,
    _has_prescriptive_cue,
    _is_citation_heavy,
    _normalize_actor,
    _normalize_instrument,
    _normalize_strength,
    detect_references_start_page,
)


# ── References page detection ─────────────────────────────────────────────

class TestReferencesDetection:
    def _pages(self, texts):
        return [
            PageText(page_num=i + 1, text=t, char_count=len(t), word_count=len(t.split()))
            for i, t in enumerate(texts)
        ]

    def test_finds_references_heading(self):
        pages = self._pages([
            "Introduction text here.",
            "Some policy content.",
            "References\nSmith (2020). Title. Journal.",
        ])
        assert detect_references_start_page(pages) == 3

    def test_finds_bibliography(self):
        pages = self._pages([
            "Body text.",
            "Bibliography\nSome ref.",
        ])
        assert detect_references_start_page(pages) == 2

    def test_no_references(self):
        pages = self._pages(["Some text.", "More text."])
        assert detect_references_start_page(pages) is None

    def test_case_insensitive(self):
        pages = self._pages(["Body.", "REFERENCES\nRef list."])
        assert detect_references_start_page(pages) == 2

    def test_works_cited(self):
        pages = self._pages(["Body.", "Works Cited\nRef list."])
        assert detect_references_start_page(pages) == 2

    def test_references_early_in_doc_ignored(self):
        """References heading in the front of the document is ignored (false positive guard)."""
        pages = self._pages([
            "References\nEarly refs.",
            "Body content here.",
            "More body.",
        ])
        # Page 1 is in the front 60% → not searched → returns None
        assert detect_references_start_page(pages) is None


# ── Prescriptive cue detection ────────────────────────────────────────────

class TestPrescriptiveCue:
    def test_should(self):
        assert _has_prescriptive_cue("Governments should implement new policies.")

    def test_must(self):
        assert _has_prescriptive_cue("Member states must comply with regulations.")

    def test_recommend(self):
        assert _has_prescriptive_cue("We recommend strengthening institutions.")

    def test_no_cue(self):
        assert not _has_prescriptive_cue("Forests cover 30% of the land area.")

    def test_suggest(self):
        assert _has_prescriptive_cue("The findings suggest a new approach.")


# ── Citation rejection ────────────────────────────────────────────────────

class TestCitationRejection:
    def test_clean_sentence(self):
        assert not _is_citation_heavy("Governments should implement new policies.")

    def test_citation_heavy(self):
        sent = "(Smith 2020) (Jones et al. 2019) (Brown & White 2021) states that forests"
        assert _is_citation_heavy(sent)

    def test_single_citation_ok(self):
        sent = "Policy should change (Smith 2020) to address the growing crisis."
        assert not _is_citation_heavy(sent)

    def test_numbered_citations_heavy(self):
        sent = "[1] [2] [3] [4] [5] [6] [7] [8] review of literature"
        assert _is_citation_heavy(sent)


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


# ── Candidate generation (without LLM) ───────────────────────────────────

class TestCandidateGeneration:
    def setup_method(self):
        # Create extractor with a mock LLM client
        class FakeLLM:
            pass
        self.extractor = RecommendationExtractor(
            llm_client=FakeLLM(),
            config={"min_confidence": 0.6, "batch_size": 10},
        )

    def _pages(self, texts):
        return [
            PageText(page_num=i + 1, text=t, char_count=len(t), word_count=len(t.split()))
            for i, t in enumerate(texts)
        ]

    def test_prescriptive_kept(self):
        pages = self._pages(["Governments should implement comprehensive monitoring systems for forests. This is a fact."])
        candidates = self.extractor._generate_candidates(pages)
        assert len(candidates) >= 1
        assert any("should" in c["text"] for c in candidates)

    def test_citation_heavy_rejected(self):
        pages = self._pages([
            "(Smith 2020) (Jones 2019) (Brown 2021) (White 2022) should regulate forests."
        ])
        candidates = self.extractor._generate_candidates(pages)
        # Should be rejected due to high citation density
        assert len(candidates) == 0

    def test_short_sentences_skipped(self):
        pages = self._pages(["We should. Yes."])
        candidates = self.extractor._generate_candidates(pages)
        assert len(candidates) == 0

    def test_references_pages_excluded_in_full_pipeline(self):
        """Test that extract_recommendations excludes references pages."""
        pages = self._pages([
            "Governments should strengthen forest monitoring systems urgently.",
            "References\nSmith (2020). Title. Journal. DOI. Governments should act.",
        ])
        refs_start = detect_references_start_page(pages)
        assert refs_start == 2
        eligible = [p for p in pages if p.page_num < refs_start]
        assert len(eligible) == 1
        assert eligible[0].page_num == 1
