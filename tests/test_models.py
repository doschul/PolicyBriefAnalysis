"""Tests for Pydantic models: validators, enums, serialisation."""

import pytest
from datetime import datetime

from src.policybrief.models import (
    ActorType,
    DocumentFrontMatter,
    DocumentMetrics,
    Evidence,
    ExtractionType,
    FrameAssessment,
    FrameDecision,
    FrameExtractionResponse,
    GeographicScope,
    InstrumentType,
    PDFMetadata,
    PageText,
    PerDocumentExtraction,
    PolicyExtraction,
    ProcessingStatus,
    RecommendationExtractionResponse,
    RecommendationItem,
    RecommendationStrength,
    SingleFrameResult,
    StructuralCoreResult,
    Timeframe,
)


# ── PageText ──────────────────────────────────────────────────────────────

class TestPageText:
    def test_valid(self):
        p = PageText(page_num=1, text="Hello world", char_count=11, word_count=2)
        assert p.page_num == 1

    def test_extra_field_rejected(self):
        with pytest.raises(Exception):
            PageText(page_num=1, text="hi", char_count=2, word_count=1, extra="bad")


# ── Evidence ──────────────────────────────────────────────────────────────

class TestEvidence:
    def test_valid(self):
        e = Evidence(page=1, quote="A valid quote that is long enough")
        assert e.page == 1
        assert e.quote == "A valid quote that is long enough"

    def test_short_quote_rejected(self):
        with pytest.raises(Exception):
            Evidence(page=1, quote="short")

    def test_whitespace_stripped(self):
        e = Evidence(page=1, quote="  A quote with leading spaces and more text ")
        assert e.quote == "A quote with leading spaces and more text"

    def test_empty_quote_rejected(self):
        with pytest.raises(Exception):
            Evidence(page=1, quote="          ")  # whitespace only but >= 10 chars


# ── DocumentMetrics ───────────────────────────────────────────────────────

class TestDocumentMetrics:
    def test_defaults(self):
        m = DocumentMetrics(
            page_count=5, word_count=100, char_count=500,
            sentence_count=10, paragraph_count=3
        )
        assert m.url_count == 0
        assert m.email_count == 0
        assert m.flesch_kincaid_grade is None

    def test_negative_page_count_rejected(self):
        with pytest.raises(Exception):
            DocumentMetrics(
                page_count=-1, word_count=0, char_count=0,
                sentence_count=0, paragraph_count=0
            )


# ── FrameAssessment ───────────────────────────────────────────────────────

class TestFrameAssessment:
    def test_present_requires_evidence(self):
        with pytest.raises(Exception):
            FrameAssessment(
                frame_id="f1", frame_label="Test",
                decision=FrameDecision.PRESENT,
                confidence=0.9, evidence=[], rationale="No evidence"
            )

    def test_absent_no_evidence_ok(self):
        a = FrameAssessment(
            frame_id="f1", frame_label="Test",
            decision=FrameDecision.ABSENT,
            confidence=0.1, evidence=[], rationale="Not found"
        )
        assert a.decision == FrameDecision.ABSENT

    def test_present_with_evidence(self):
        ev = Evidence(page=1, quote="Evidence quote for the frame assessment")
        a = FrameAssessment(
            frame_id="f1", frame_label="Test",
            decision=FrameDecision.PRESENT,
            confidence=0.9, evidence=[ev], rationale="Found"
        )
        assert len(a.evidence) == 1


# ── PolicyExtraction ──────────────────────────────────────────────────────

class TestPolicyExtraction:
    def test_recommendation_requires_evidence(self):
        with pytest.raises(Exception):
            PolicyExtraction(
                rec_id="r1",
                extraction_type=ExtractionType.RECOMMENDATION,
                confidence=0.8,
                source_text_raw="Governments should improve forest monitoring.",
                page=5,
                evidence=[],
            )

    def test_non_recommendation_no_evidence_ok(self):
        pe = PolicyExtraction(
            rec_id="r2",
            extraction_type=ExtractionType.TRADE_OFF,
            confidence=0.7,
            source_text_raw="There is a trade-off between conservation and development.",
            page=10,
            evidence=[],
        )
        assert pe.extraction_type == ExtractionType.TRADE_OFF

    def test_null_actor_preserved(self):
        ev = Evidence(page=5, quote="A sufficiently long quote for the recommendation evidence")
        pe = PolicyExtraction(
            rec_id="r3",
            extraction_type=ExtractionType.RECOMMENDATION,
            confidence=0.8,
            source_text_raw="Monitoring should be improved.",
            page=5,
            actor_text_raw=None,
            actor_type_normalized=None,
            evidence=[ev],
        )
        assert pe.actor_text_raw is None
        assert pe.actor_type_normalized is None


# ── StructuralCoreResult ──────────────────────────────────────────────────

class TestStructuralCoreResult:
    def test_defaults(self):
        sc = StructuralCoreResult()
        assert sc.problem_status == "absent"
        assert sc.solutions_count == 0
        assert sc.problem_explicitly_labelled is False
        assert sc.solutions_explicitly_labelled is False
        assert sc.implementation_explicitly_labelled is False
        assert sc.procedural_clarity_status == "absent"

    def test_full(self):
        sc = StructuralCoreResult(
            problem_status="present",
            problem_summary="Deforestation in the Amazon",
            solutions_count=3,
            solutions_explicit=True,
            implementation_status="weak",
            implementation_count=1,
            narrative_hook_present=True,
            narrative_hook_type="statistic",
            problem_explicitly_labelled=True,
            solutions_explicitly_labelled=True,
            implementation_explicitly_labelled=False,
            procedural_clarity_status="present",
        )
        assert sc.solutions_explicit is True
        assert sc.problem_explicitly_labelled is True
        assert sc.procedural_clarity_status == "present"

    def test_procedural_clarity_distinct_from_implementation(self):
        """procedural_clarity_status can differ from implementation_status."""
        sc = StructuralCoreResult(
            implementation_status="present",
            procedural_clarity_status="absent",
        )
        assert sc.implementation_status == "present"
        assert sc.procedural_clarity_status == "absent"

    def test_heading_labelled_with_implicit_structure(self):
        """A document can have present content but no explicit heading labels."""
        sc = StructuralCoreResult(
            problem_status="present",
            problem_explicitly_labelled=False,
            solutions_explicit=True,
            solutions_explicitly_labelled=False,
        )
        assert sc.problem_status == "present"
        assert sc.problem_explicitly_labelled is False


# ── RecommendationItem / RecommendationExtractionResponse ─────────────────

class TestRecommendationItem:
    def test_valid(self):
        item = RecommendationItem(
            extraction_type=ExtractionType.RECOMMENDATION,
            confidence=0.85,
            source_quote="Governments should strengthen forest monitoring systems",
            page=3,
            actor_text_raw="Governments",
            action_text_raw="strengthen forest monitoring systems",
        )
        assert item.confidence == 0.85
        assert item.page == 3

    def test_minimal(self):
        item = RecommendationItem(
            extraction_type=ExtractionType.TRADE_OFF,
            confidence=0.6,
            source_quote="There is a trade-off between conservation and growth",
            page=5,
        )
        assert item.actor_text_raw is None
        assert item.instrument_type is None

    def test_response_empty_items(self):
        resp = RecommendationExtractionResponse(items=[])
        assert len(resp.items) == 0

    def test_response_with_items(self):
        item = RecommendationItem(
            extraction_type=ExtractionType.RECOMMENDATION,
            confidence=0.9,
            source_quote="Member states must implement monitoring systems",
            page=1,
        )
        resp = RecommendationExtractionResponse(items=[item])
        assert len(resp.items) == 1


# ── SingleFrameResult / FrameExtractionResponse ──────────────────────────

class TestSingleFrameResult:
    def test_valid(self):
        fr = SingleFrameResult(
            frame_id="command_and_control",
            decision=FrameDecision.PRESENT,
            confidence=0.9,
            evidence=[Evidence(page=1, quote="The regulation requires compliance from all operators")],
            rationale="Clear legal compulsion",
        )
        assert fr.frame_id == "command_and_control"

    def test_absent(self):
        fr = SingleFrameResult(
            frame_id="economic_instruments",
            decision=FrameDecision.ABSENT,
            confidence=0.1,
            evidence=[],
            rationale="No financial mechanisms discussed",
        )
        assert fr.decision == FrameDecision.ABSENT

    def test_frame_extraction_response(self):
        frames = [
            SingleFrameResult(
                frame_id="f1", decision=FrameDecision.ABSENT,
                confidence=0.1, evidence=[], rationale="Not found",
            ),
            SingleFrameResult(
                frame_id="f2", decision=FrameDecision.PRESENT,
                confidence=0.85,
                evidence=[Evidence(page=2, quote="PES payments provide incentives for conservation")],
                rationale="Clear economic instrument",
            ),
        ]
        resp = FrameExtractionResponse(frames=frames)
        assert len(resp.frames) == 2


# ── Enums ─────────────────────────────────────────────────────────────────

class TestEnums:
    def test_instrument_type_values(self):
        assert InstrumentType.REGULATION.value == "regulation"
        assert InstrumentType.SUBSIDY.value == "subsidy"

    def test_actor_type_values(self):
        assert ActorType.GOVERNMENT.value == "government"

    def test_frame_decision(self):
        assert FrameDecision.PRESENT.value == "present"
        assert FrameDecision.ABSENT.value == "absent"

    def test_extraction_type(self):
        assert ExtractionType.RECOMMENDATION.value == "recommendation"
        assert ExtractionType.NON_RECOMMENDATION.value == "non_recommendation"

    def test_strength_values(self):
        assert RecommendationStrength.MUST.value == "must"
        assert RecommendationStrength.UNSPECIFIED.value == "unspecified"


# ── Serialisation round-trip ──────────────────────────────────────────────

class TestSerialisation:
    def test_document_front_matter_roundtrip(self):
        fm = DocumentFrontMatter(
            title="Test Brief",
            authors=["Author A"],
            emails=["a@b.com"],
        )
        data = fm.model_dump()
        fm2 = DocumentFrontMatter.model_validate(data)
        assert fm2.title == "Test Brief"

    def test_frame_extraction_response_json_schema(self):
        schema = FrameExtractionResponse.model_json_schema()
        assert "properties" in schema
        assert "frames" in schema["properties"]

    def test_recommendation_extraction_response_json_schema(self):
        schema = RecommendationExtractionResponse.model_json_schema()
        assert "properties" in schema
        assert "items" in schema["properties"]
