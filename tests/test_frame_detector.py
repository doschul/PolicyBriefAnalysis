"""Tests for broad-content frame detector: LLM assessment, quote validation, policy mix, aggregation."""

import pytest

from src.policybrief.models import (
    Evidence,
    FrameAssessment,
    FrameDecision,
    FrameExtractionResponse,
    PageText,
    SingleFrameResult,
)
from src.policybrief.frame_detector import FrameDetector


def _make_pages(texts):
    return [
        PageText(page_num=i + 1, text=t, char_count=len(t), word_count=len(t.split()))
        for i, t in enumerate(texts)
    ]


SAMPLE_FRAMES = [
    {
        "id": "command_and_control",
        "label": "Command-and-Control",
        "short_definition": "Legally binding rules backed by state authority.",
        "analytical_notes": "The defining logic is legal compulsion.",
        "false_positive_notes": "",
    },
    {
        "id": "economic_instruments",
        "label": "Economic Instruments",
        "short_definition": "Financial incentives including PES and subsidies.",
        "analytical_notes": "The defining logic is financial incentive.",
        "false_positive_notes": "",
    },
]


class FakeLLMAbsent:
    """Mock LLM that returns absent for all frames."""
    def structured_completion(self, messages, response_model):
        return FrameExtractionResponse(frames=[
            SingleFrameResult(
                frame_id="command_and_control",
                decision=FrameDecision.ABSENT,
                confidence=0.1,
                evidence=[],
                rationale="No legal compulsion found",
            ),
            SingleFrameResult(
                frame_id="economic_instruments",
                decision=FrameDecision.ABSENT,
                confidence=0.1,
                evidence=[],
                rationale="No financial instruments found",
            ),
        ])


class FakeLLMPresent:
    """Mock LLM that returns present for command_and_control with evidence."""
    def __init__(self, quote="The government requires all operators to obtain permits"):
        self.quote = quote

    def structured_completion(self, messages, response_model):
        return FrameExtractionResponse(frames=[
            SingleFrameResult(
                frame_id="command_and_control",
                decision=FrameDecision.PRESENT,
                confidence=0.92,
                evidence=[Evidence(page=1, quote=self.quote)],
                rationale="Strong evidence of legal compulsion",
            ),
            SingleFrameResult(
                frame_id="economic_instruments",
                decision=FrameDecision.ABSENT,
                confidence=0.1,
                evidence=[],
                rationale="No financial instruments found",
            ),
        ])


class FakeLLMBothPresent:
    """Mock LLM that returns present for both frames."""
    def __init__(self, quote1, quote2):
        self.quote1 = quote1
        self.quote2 = quote2

    def structured_completion(self, messages, response_model):
        return FrameExtractionResponse(frames=[
            SingleFrameResult(
                frame_id="command_and_control",
                decision=FrameDecision.PRESENT,
                confidence=0.9,
                evidence=[Evidence(page=1, quote=self.quote1)],
                rationale="Legal compulsion found",
            ),
            SingleFrameResult(
                frame_id="economic_instruments",
                decision=FrameDecision.PRESENT,
                confidence=0.85,
                evidence=[Evidence(page=1, quote=self.quote2)],
                rationale="Financial instruments found",
            ),
        ])


# ── Quote validation ──────────────────────────────────────────────────────

class TestQuoteValidation:
    def test_exact_match(self):
        detector = FrameDetector(
            llm_client=FakeLLMAbsent(), frames_config=SAMPLE_FRAMES,
        )
        source = "The government requires all operators to obtain permits for logging."
        evidence = [Evidence(page=1, quote="The government requires all operators to obtain permits")]
        validated = detector._validate_quotes(evidence, source)
        assert len(validated) == 1

    def test_no_match_rejected(self):
        detector = FrameDetector(
            llm_client=FakeLLMAbsent(), frames_config=SAMPLE_FRAMES,
        )
        source = "Forests cover 30% of land."
        evidence = [Evidence(page=1, quote="This quote does not appear in the source text at all")]
        validated = detector._validate_quotes(evidence, source)
        assert len(validated) == 0

    def test_prefix_match_accepted(self):
        detector = FrameDetector(
            llm_client=FakeLLMAbsent(), frames_config=SAMPLE_FRAMES,
        )
        source = "The government requires all operators to obtain permits and comply with regulations."
        evidence = [Evidence(page=1, quote="The government requires all operators to obtain permits and report annually")]
        validated = detector._validate_quotes(evidence, source)
        # First 40 chars match → accepted
        assert len(validated) == 1


# ── Policy mix detection ─────────────────────────────────────────────────

class TestPolicyMix:
    def test_two_frames_present(self):
        detector = FrameDetector(
            llm_client=FakeLLMAbsent(), frames_config=SAMPLE_FRAMES,
            min_confidence=0.7,
        )
        ev = Evidence(page=1, quote="A sufficiently long evidence quote for testing")
        assessments = [
            FrameAssessment(
                frame_id="command_and_control", frame_label="CaC",
                decision=FrameDecision.PRESENT, confidence=0.9,
                evidence=[ev], rationale="Found",
            ),
            FrameAssessment(
                frame_id="economic_instruments", frame_label="EI",
                decision=FrameDecision.PRESENT, confidence=0.85,
                evidence=[ev], rationale="Found",
            ),
        ]
        assert detector.detect_policy_mix(assessments) is True

    def test_one_frame_no_mix(self):
        detector = FrameDetector(
            llm_client=FakeLLMAbsent(), frames_config=SAMPLE_FRAMES,
            min_confidence=0.7,
        )
        ev = Evidence(page=1, quote="A sufficiently long evidence quote for testing")
        assessments = [
            FrameAssessment(
                frame_id="command_and_control", frame_label="CaC",
                decision=FrameDecision.PRESENT, confidence=0.9,
                evidence=[ev], rationale="Found",
            ),
            FrameAssessment(
                frame_id="economic_instruments", frame_label="EI",
                decision=FrameDecision.ABSENT, confidence=0.1,
                evidence=[], rationale="Not found",
            ),
        ]
        assert detector.detect_policy_mix(assessments) is False

    def test_low_confidence_no_mix(self):
        detector = FrameDetector(
            llm_client=FakeLLMAbsent(), frames_config=SAMPLE_FRAMES,
            min_confidence=0.7,
        )
        ev = Evidence(page=1, quote="A sufficiently long evidence quote for testing")
        assessments = [
            FrameAssessment(
                frame_id="command_and_control", frame_label="CaC",
                decision=FrameDecision.PRESENT, confidence=0.9,
                evidence=[ev], rationale="Found",
            ),
            FrameAssessment(
                frame_id="economic_instruments", frame_label="EI",
                decision=FrameDecision.PRESENT, confidence=0.5,
                evidence=[ev], rationale="Weak",
            ),
        ]
        assert detector.detect_policy_mix(assessments) is False


# ── Full frame detection with mock LLM ───────────────────────────────────

class TestFrameDetection:
    def test_absent_for_generic_text(self):
        detector = FrameDetector(
            llm_client=FakeLLMAbsent(), frames_config=SAMPLE_FRAMES,
            min_confidence=0.7,
        )
        pages = _make_pages(["Forests are beautiful ecosystems with many species."])
        assessments = detector.detect_frames(pages)
        assert len(assessments) == 2
        assert all(a.decision == FrameDecision.ABSENT for a in assessments)

    def test_present_with_valid_evidence(self):
        text = (
            "The government requires all operators to obtain permits "
            "and ensures compliance before any logging activity can commence."
        )
        detector = FrameDetector(
            llm_client=FakeLLMPresent(
                quote="The government requires all operators to obtain permits"
            ),
            frames_config=SAMPLE_FRAMES,
            min_confidence=0.7,
        )
        pages = _make_pages([text])
        assessments = detector.detect_frames(pages)
        assert len(assessments) == 2
        cac = next(a for a in assessments if a.frame_id == "command_and_control")
        assert cac.decision == FrameDecision.PRESENT
        assert cac.confidence > 0.7

    def test_present_downgraded_when_evidence_not_in_text(self):
        """If evidence quote doesn't match source text, decision is downgraded."""
        text = "Forests are beautiful ecosystems that support biodiversity."
        detector = FrameDetector(
            llm_client=FakeLLMPresent(
                quote="The government requires all operators to obtain permits"
            ),
            frames_config=SAMPLE_FRAMES,
            min_confidence=0.7,
        )
        pages = _make_pages([text])
        assessments = detector.detect_frames(pages)
        cac = next(a for a in assessments if a.frame_id == "command_and_control")
        # Evidence doesn't match → downgraded to insufficient_evidence
        assert cac.decision == FrameDecision.INSUFFICIENT_EVIDENCE

    def test_no_pages_returns_absent(self):
        detector = FrameDetector(
            llm_client=FakeLLMAbsent(), frames_config=SAMPLE_FRAMES,
        )
        assessments = detector.detect_frames([])
        assert len(assessments) == 2
        assert all(a.decision == FrameDecision.ABSENT for a in assessments)

    def test_excluded_pages_respected(self):
        """Pages in excluded_pages set are not processed."""
        text = "The government requires all operators to obtain permits for logging."
        detector = FrameDetector(
            llm_client=FakeLLMAbsent(), frames_config=SAMPLE_FRAMES,
        )
        pages = _make_pages([text, "Second page content."])
        # Exclude both pages → no content → absent
        assessments = detector.detect_frames(pages, excluded_pages={1, 2})
        assert all(a.decision == FrameDecision.ABSENT for a in assessments)

    def test_both_frames_present_policy_mix(self):
        quote1 = "The regulation mandates strict compliance"
        quote2 = "PES payments provide financial incentives"
        text = f"{quote1} for operators. {quote2} for conservation."
        detector = FrameDetector(
            llm_client=FakeLLMBothPresent(quote1=quote1, quote2=quote2),
            frames_config=SAMPLE_FRAMES,
            min_confidence=0.7,
        )
        pages = _make_pages([text])
        assessments = detector.detect_frames(pages)
        assert detector.detect_policy_mix(assessments) is True


# ── Absent assessment helper ──────────────────────────────────────────────

class TestAbsentAssessment:
    def test_absent_assessment_fields(self):
        detector = FrameDetector(
            llm_client=FakeLLMAbsent(), frames_config=SAMPLE_FRAMES,
        )
        a = detector._absent_assessment(SAMPLE_FRAMES[0], "No data")
        assert a.frame_id == "command_and_control"
        assert a.decision == FrameDecision.ABSENT
        assert a.confidence == 0.0
        assert a.evidence == []
        assert a.rationale == "No data"
