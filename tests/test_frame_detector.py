"""Tests for frame detector: keyword selection, quote validation, policy mix."""

import pytest
import re

from src.policybrief.models import (
    Evidence,
    FrameAssessment,
    FrameDecision,
    PageText,
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
        "inclusion_cues": ["regulation", "law", "enforcement", "penalty", "compliance"],
    },
    {
        "id": "economic_instruments",
        "label": "Economic Instruments",
        "short_definition": "Financial incentives including PES and subsidies.",
        "analytical_notes": "The defining logic is financial incentive.",
        "false_positive_notes": "",
        "inclusion_cues": ["subsidy", "payment", "PES", "carbon credit", "incentive"],
    },
]


class FakeLLM:
    """Mock LLM that returns absent for all frames."""
    def structured_completion(self, messages, response_model):
        from src.policybrief.models import FrameDetectionOutput, FrameDecision
        # Parse frame_id from user message
        user_msg = messages[-1]["content"]
        frame_id = "unknown"
        if "frame_id='" in user_msg:
            start = user_msg.index("frame_id='") + len("frame_id='")
            end = user_msg.index("'", start)
            frame_id = user_msg[start:end]
        return FrameDetectionOutput(
            frame_id=frame_id,
            decision=FrameDecision.ABSENT,
            confidence=0.2,
            evidence=[],
            rationale="No evidence found by mock LLM",
        )


class FakeLLMPresent:
    """Mock LLM that returns present with evidence."""
    def __init__(self, quote="The government requires all operators to obtain permits"):
        self.quote = quote

    def structured_completion(self, messages, response_model):
        from src.policybrief.models import FrameDetectionOutput, FrameDecision, Evidence
        user_msg = messages[-1]["content"]
        frame_id = "unknown"
        if "frame_id='" in user_msg:
            start = user_msg.index("frame_id='") + len("frame_id='")
            end = user_msg.index("'", start)
            frame_id = user_msg[start:end]
        return FrameDetectionOutput(
            frame_id=frame_id,
            decision=FrameDecision.PRESENT,
            confidence=0.92,
            evidence=[Evidence(page=1, quote=self.quote)],
            rationale="Strong evidence of the frame",
        )


# ── Keyword span selection ────────────────────────────────────────────────

class TestKeywordSpans:
    def test_finds_keyword_matches(self):
        detector = FrameDetector(
            llm_client=FakeLLM(), frames_config=SAMPLE_FRAMES,
            min_confidence=0.7, context_window=100,
        )
        text = "The new regulation requires compliance from all operators."
        text_by_page = {1: text}
        spans = detector._find_keyword_spans("command_and_control", text, text_by_page)
        assert len(spans) > 0
        assert any("regulation" in s["keyword"].lower() for s in spans)

    def test_no_matches_returns_empty(self):
        detector = FrameDetector(
            llm_client=FakeLLM(), frames_config=SAMPLE_FRAMES,
            min_confidence=0.7,
        )
        text = "Forests are beautiful and diverse ecosystems."
        text_by_page = {1: text}
        spans = detector._find_keyword_spans("command_and_control", text, text_by_page)
        assert len(spans) == 0


# ── Quote validation ──────────────────────────────────────────────────────

class TestQuoteValidation:
    def test_exact_match(self):
        detector = FrameDetector(
            llm_client=FakeLLM(), frames_config=SAMPLE_FRAMES,
        )
        source = "The government requires all operators to obtain permits for logging."
        evidence = [Evidence(page=1, quote="The government requires all operators to obtain permits")]
        validated = detector._validate_quotes(evidence, source)
        assert len(validated) == 1

    def test_no_match_rejected(self):
        detector = FrameDetector(
            llm_client=FakeLLM(), frames_config=SAMPLE_FRAMES,
        )
        source = "Forests cover 30% of land."
        evidence = [Evidence(page=1, quote="This quote does not appear in the source text at all")]
        validated = detector._validate_quotes(evidence, source)
        assert len(validated) == 0


# ── Policy mix detection ─────────────────────────────────────────────────

class TestPolicyMix:
    def test_two_frames_present(self):
        detector = FrameDetector(
            llm_client=FakeLLM(), frames_config=SAMPLE_FRAMES,
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
            llm_client=FakeLLM(), frames_config=SAMPLE_FRAMES,
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


# ── Full frame detection with mock LLM ───────────────────────────────────

class TestFrameDetection:
    def test_no_keywords_returns_absent(self):
        detector = FrameDetector(
            llm_client=FakeLLM(), frames_config=SAMPLE_FRAMES,
            min_confidence=0.7,
        )
        pages = _make_pages(["Forests are beautiful ecosystems with many species."])
        assessments = detector.detect_frames(pages)
        assert len(assessments) == 2
        assert all(a.decision == FrameDecision.ABSENT for a in assessments)

    def test_with_keywords_calls_llm(self):
        """When keywords match, the LLM is called (fake returns absent)."""
        detector = FrameDetector(
            llm_client=FakeLLM(), frames_config=SAMPLE_FRAMES,
            min_confidence=0.7,
        )
        text = (
            "The new forest regulation requires strict compliance from "
            "all logging operators. Penalties for non-compliance include fines."
        )
        pages = _make_pages([text])
        assessments = detector.detect_frames(pages)
        # command_and_control should have been sent to LLM (has keyword matches)
        # But FakeLLM returns absent
        assert len(assessments) == 2

    def test_present_with_valid_evidence(self):
        """LLM returns present + evidence that exists in source."""
        text = (
            "The new regulation requires all operators to obtain permits "
            "and ensures compliance before any logging activity can commence."
        )
        detector = FrameDetector(
            llm_client=FakeLLMPresent(
                quote="The new regulation requires all operators to obtain permits"
            ),
            frames_config=[SAMPLE_FRAMES[0]],  # Only command_and_control
            min_confidence=0.7,
        )
        pages = _make_pages([text])
        assessments = detector.detect_frames(pages)
        assert len(assessments) == 1
        assert assessments[0].decision == FrameDecision.PRESENT
        assert assessments[0].confidence > 0.7
