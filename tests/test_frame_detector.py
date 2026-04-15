"""
Tests for the policy-instrument frame detector.

Covers:
  - Candidate span selection for each of the five instrument categories
  - LLM integration via mocked structured_completion
  - False-positive guards (rhetoric, citations, pledge vs self-regulation,
    certification classification)
  - Multi-category / policy-mix detection
  - Frame definition formatting (new YAML fields)
"""

import re
import pytest
from unittest.mock import MagicMock, patch

from src.policybrief.frame_detector import (
    FrameDetector,
    _POLICY_MIX_CUES,
    _POLICY_MIX_MIN_FRAMES,
)
from src.policybrief.models import (
    Evidence,
    FrameAssessment,
    FrameDetectionInput,
    FrameDetectionOutput,
    FrameDecision,
    PageText,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _make_pages(texts: list[str]) -> list[PageText]:
    """Create PageText objects from a list of page strings."""
    return [
        PageText(
            page_num=i + 1,
            text=t,
            char_count=len(t),
            word_count=len(t.split()),
        )
        for i, t in enumerate(texts)
    ]


def _load_frames_config():
    """Load actual frames.yaml for integration-style tests."""
    from pathlib import Path
    import yaml

    frames_path = Path(__file__).resolve().parents[1] / "config" / "frames.yaml"
    with open(frames_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["frames"]


def _make_detector(frames_config=None, **kwargs):
    """Build a FrameDetector with a mocked LLM client."""
    llm_client = MagicMock()
    if frames_config is None:
        frames_config = _load_frames_config()
    defaults = dict(
        min_confidence=0.7,
        max_spans_per_frame=5,
        context_window=500,
        min_evidence_quotes=1,
        max_evidence_quotes=3,
    )
    defaults.update(kwargs)
    return FrameDetector(llm_client=llm_client, frames_config=frames_config, **defaults)


# ---------------------------------------------------------------
# Candidate span selection — one test per instrument category
# ---------------------------------------------------------------

class TestCandidateSpanSelection:
    """Verify that Stage-1 keyword matching selects relevant spans."""

    def test_command_and_control_spans(self):
        """Harvesting permits and legal compliance trigger candidate selection."""
        detector = _make_detector()
        pages = _make_pages([
            "The government requires all logging operators to obtain "
            "harvesting permits before timber extraction. Non-compliant "
            "operators face fines under forest legislation."
        ])
        frame_cfg = detector.frames["command_and_control"]
        spans = detector._select_candidate_spans(pages, frame_cfg)

        assert len(spans) > 0
        texts = " ".join(s["text"] for s in spans).lower()
        assert "harvesting permit" in texts or "permit" in texts

    def test_economic_instruments_spans(self):
        """PES and carbon credits trigger candidate selection."""
        detector = _make_detector()
        pages = _make_pages([
            "The PES programme compensates landowners for maintaining "
            "forest cover. REDD+ results-based payments reward verified "
            "emission reductions from avoided deforestation."
        ])
        frame_cfg = detector.frames["economic_instruments"]
        spans = detector._select_candidate_spans(pages, frame_cfg)

        assert len(spans) > 0
        texts = " ".join(s["text"] for s in spans).lower()
        assert "pes" in texts or "redd" in texts

    def test_self_regulation_spans(self):
        """FSC certification and industry codes trigger selection."""
        detector = _make_detector()
        pages = _make_pages([
            "The FSC certification scheme sets collective standards for "
            "sustainable forest management across the timber industry. "
            "Industry associations developed codes of practice for "
            "reduced-impact logging operations."
        ])
        frame_cfg = detector.frames["self_regulation"]
        spans = detector._select_candidate_spans(pages, frame_cfg)

        assert len(spans) > 0
        texts = " ".join(s["text"] for s in spans).lower()
        assert "fsc" in texts or "certification" in texts

    def test_voluntarism_spans(self):
        """Corporate pledges trigger candidate selection."""
        detector = _make_detector()
        pages = _make_pages([
            "The company pledged to eliminate deforestation from its "
            "supply chain by 2025. This voluntary commitment was made "
            "independently, without sector-wide enforcement."
        ])
        frame_cfg = detector.frames["voluntarism"]
        spans = detector._select_candidate_spans(pages, frame_cfg)

        assert len(spans) > 0
        texts = " ".join(s["text"] for s in spans).lower()
        assert "pledge" in texts or "voluntary" in texts

    def test_information_strategies_spans(self):
        """Transparency and traceability trigger selection."""
        detector = _make_detector()
        pages = _make_pages([
            "The government established a satellite-based forest "
            "monitoring system to track deforestation in real time. "
            "Supply-chain traceability requirements ensure that buyers "
            "can verify the origin of timber products."
        ])
        frame_cfg = detector.frames["information_strategies"]
        spans = detector._select_candidate_spans(pages, frame_cfg)

        assert len(spans) > 0
        texts = " ".join(s["text"] for s in spans).lower()
        assert "monitoring" in texts or "traceability" in texts

    def test_no_spans_for_irrelevant_text(self):
        """Unrelated text should produce no candidates."""
        detector = _make_detector()
        pages = _make_pages([
            "The weather today is sunny and warm. Nothing about policy."
        ])
        for frame_id, frame_cfg in detector.frames.items():
            spans = detector._select_candidate_spans(pages, frame_cfg)
            assert len(spans) == 0, f"Unexpected spans for {frame_id}"


# ---------------------------------------------------------------
# LLM integration (mocked) — full detect_single_frame path
# ---------------------------------------------------------------

class TestSingleFrameDetection:
    """Verify end-to-end single-frame detection with mocked LLM."""

    def _mock_llm_present(self, frame_id, quote, page=1):
        """Return a mocked FrameDetectionOutput marked present."""
        return FrameDetectionOutput(
            frame_id=frame_id,
            decision=FrameDecision.PRESENT,
            confidence=0.9,
            evidence=[Evidence(page=page, quote=quote)],
            rationale="Clear evidence of instrument category.",
        )

    def test_command_and_control_detected(self):
        """Command-and-control detected when LLM confirms."""
        text = (
            "The government requires all logging operators to obtain "
            "harvesting permits. Non-compliant operators face penalties "
            "under forest law enforcement."
        )
        detector = _make_detector()
        detector.llm_client.detect_frame.return_value = self._mock_llm_present(
            "command_and_control",
            "The government requires all logging operators to obtain harvesting permits.",
        )

        pages = _make_pages([text])
        result = detector._detect_single_frame(
            pages, "command_and_control", detector.frames["command_and_control"]
        )

        assert result.decision == FrameDecision.PRESENT
        assert result.confidence >= 0.7
        assert len(result.evidence) >= 1

    def test_economic_instruments_detected(self):
        text = (
            "The PES programme compensates landowners for maintaining "
            "forest cover. Carbon credits provide financial incentives "
            "for reforestation projects."
        )
        detector = _make_detector()
        detector.llm_client.detect_frame.return_value = self._mock_llm_present(
            "economic_instruments",
            "The PES programme compensates landowners for maintaining forest cover.",
        )

        pages = _make_pages([text])
        result = detector._detect_single_frame(
            pages, "economic_instruments", detector.frames["economic_instruments"]
        )

        assert result.decision == FrameDecision.PRESENT

    def test_self_regulation_detected(self):
        text = (
            "The FSC certification scheme sets collective standards for "
            "sustainable forest management across the industry."
        )
        detector = _make_detector()
        detector.llm_client.detect_frame.return_value = self._mock_llm_present(
            "self_regulation",
            "The FSC certification scheme sets collective standards for sustainable forest management across the industry.",
        )

        pages = _make_pages([text])
        result = detector._detect_single_frame(
            pages, "self_regulation", detector.frames["self_regulation"]
        )

        assert result.decision == FrameDecision.PRESENT

    def test_voluntarism_detected(self):
        text = (
            "The company pledged to eliminate deforestation from its "
            "supply chain by 2025. This voluntary commitment operates "
            "without sector-wide enforcement."
        )
        detector = _make_detector()
        detector.llm_client.detect_frame.return_value = self._mock_llm_present(
            "voluntarism",
            "The company pledged to eliminate deforestation from its supply chain by 2025.",
        )

        pages = _make_pages([text])
        result = detector._detect_single_frame(
            pages, "voluntarism", detector.frames["voluntarism"]
        )

        assert result.decision == FrameDecision.PRESENT

    def test_information_strategies_detected(self):
        text = (
            "Supply-chain traceability requirements ensure that buyers "
            "can verify the origin of timber products. Public disclosure "
            "of trade data enables civil society to monitor compliance."
        )
        detector = _make_detector()
        detector.llm_client.detect_frame.return_value = self._mock_llm_present(
            "information_strategies",
            "Supply-chain traceability requirements ensure that buyers can verify the origin of timber products.",
        )

        pages = _make_pages([text])
        result = detector._detect_single_frame(
            pages, "information_strategies", detector.frames["information_strategies"]
        )

        assert result.decision == FrameDecision.PRESENT

    def test_absent_when_no_spans(self):
        """Frame should be absent when no candidate spans are found."""
        detector = _make_detector()
        pages = _make_pages(["The weather is nice today."])
        result = detector._detect_single_frame(
            pages, "command_and_control", detector.frames["command_and_control"]
        )

        assert result.decision == "absent"
        assert result.confidence >= 0.8


# ---------------------------------------------------------------
# Mixed document — multiple categories co-occurring
# ---------------------------------------------------------------

class TestMixedDocument:
    """Document containing two or more instrument categories."""

    def test_two_categories_detected(self):
        """Both command-and-control and economic instruments detected."""
        text = (
            "The government requires harvesting permits for all logging "
            "operators (command-and-control). In addition, a PES programme "
            "compensates landowners for maintaining forest cover (economic "
            "instrument). This policy mix combines regulation with "
            "financial incentives."
        )
        detector = _make_detector()
        pages = _make_pages([text])

        # Command-and-control should have spans
        cc_spans = detector._select_candidate_spans(
            pages, detector.frames["command_and_control"]
        )
        assert len(cc_spans) > 0

        # Economic instruments should have spans
        ei_spans = detector._select_candidate_spans(
            pages, detector.frames["economic_instruments"]
        )
        assert len(ei_spans) > 0


# ---------------------------------------------------------------
# Policy-mix detection
# ---------------------------------------------------------------

class TestPolicyMixDetection:
    """Test detect_policy_mix heuristic."""

    def test_policy_mix_with_explicit_language(self):
        """Policy mix detected when ≥2 frames present + mix language."""
        detector = _make_detector()
        pages = _make_pages([
            "This document analyses a policy mix combining regulation "
            "with financial incentives for forest conservation."
        ])
        assessments = [
            FrameAssessment(
                frame_id="command_and_control",
                frame_label="Command-and-Control",
                decision=FrameDecision.PRESENT,
                confidence=0.9,
                evidence=[Evidence(page=1, quote="regulation combining regulation")],
                rationale="Clear legal instruments.",
            ),
            FrameAssessment(
                frame_id="economic_instruments",
                frame_label="Economic Instruments",
                decision=FrameDecision.PRESENT,
                confidence=0.85,
                evidence=[Evidence(page=1, quote="financial incentives for forest conservation")],
                rationale="PES evidence.",
            ),
        ]

        assert detector.detect_policy_mix(pages, assessments) is True

    def test_no_policy_mix_without_explicit_language(self):
        """Co-occurrence alone is not sufficient without mix language."""
        detector = _make_detector()
        pages = _make_pages([
            "Permits are required. Subsidies are available."
        ])
        assessments = [
            FrameAssessment(
                frame_id="command_and_control",
                frame_label="Command-and-Control",
                decision=FrameDecision.PRESENT,
                confidence=0.9,
                evidence=[Evidence(page=1, quote="Permits are required.")],
                rationale="Legal.",
            ),
            FrameAssessment(
                frame_id="economic_instruments",
                frame_label="Economic Instruments",
                decision=FrameDecision.PRESENT,
                confidence=0.85,
                evidence=[Evidence(page=1, quote="Subsidies are available.")],
                rationale="Financial.",
            ),
        ]

        assert detector.detect_policy_mix(pages, assessments) is False

    def test_no_policy_mix_single_frame(self):
        """Policy mix cannot be detected with only one frame present."""
        detector = _make_detector()
        pages = _make_pages([
            "This policy mix of instruments is described."
        ])
        assessments = [
            FrameAssessment(
                frame_id="command_and_control",
                frame_label="Command-and-Control",
                decision=FrameDecision.PRESENT,
                confidence=0.9,
                evidence=[Evidence(page=1, quote="This policy mix of instruments is described.")],
                rationale="Legal.",
            ),
        ]

        assert detector.detect_policy_mix(pages, assessments) is False

    def test_complementarity_language_triggers_mix(self):
        """'Complementarity' language triggers mix when ≥2 present."""
        detector = _make_detector()
        pages = _make_pages([
            "The complementarity of self-regulation and information "
            "disclosure creates effective governance."
        ])
        assessments = [
            FrameAssessment(
                frame_id="self_regulation",
                frame_label="Self-Regulation",
                decision=FrameDecision.PRESENT,
                confidence=0.8,
                evidence=[Evidence(page=1, quote="complementarity of self-regulation and information")],
                rationale="Collective standards.",
            ),
            FrameAssessment(
                frame_id="information_strategies",
                frame_label="Information Strategies",
                decision=FrameDecision.PRESENT,
                confidence=0.8,
                evidence=[Evidence(page=1, quote="information disclosure creates effective governance")],
                rationale="Disclosure mechanism.",
            ),
        ]

        assert detector.detect_policy_mix(pages, assessments) is True


# ---------------------------------------------------------------
# False-positive guards
# ---------------------------------------------------------------

class TestFalsePositiveGuards:
    """Ensure that common false-positive patterns do not produce spans."""

    def test_generic_sustainability_rhetoric(self):
        """Generic sustainability language should not trigger any frame."""
        detector = _make_detector()
        pages = _make_pages([
            "Sustainability is important for future generations. "
            "We must protect our planet and its resources for all."
        ])
        for frame_id, frame_cfg in detector.frames.items():
            spans = detector._select_candidate_spans(pages, frame_cfg)
            # May get zero or very few low-score spans, but not high-quality ones
            high_score = [s for s in spans if s["score"] >= 2.0]
            assert len(high_score) == 0, (
                f"Generic rhetoric triggered high-score spans for {frame_id}"
            )

    def test_academic_citations_not_information_strategy(self):
        """Ordinary citations should not trigger information_strategies."""
        detector = _make_detector()
        pages = _make_pages([
            "Smith (2020) found that deforestation rates increased. "
            "According to Jones et al. (2019), forest cover declined "
            "by 15% over the study period."
        ])
        frame_cfg = detector.frames["information_strategies"]
        spans = detector._select_candidate_spans(pages, frame_cfg)
        # Citations don't contain info-strategy cues like "transparency",
        # "traceability", "disclosure", etc.
        assert len(spans) == 0

    def test_individual_pledge_not_self_regulation(self):
        """A single company pledge should not trigger self_regulation."""
        detector = _make_detector()
        pages = _make_pages([
            "The company pledged to eliminate deforestation from its "
            "supply chain by 2025. This non-binding commitment was "
            "made unilaterally by the CEO."
        ])
        frame_cfg = detector.frames["self_regulation"]
        spans = detector._select_candidate_spans(pages, frame_cfg)
        # "pledge" and "commitment" are NOT self-regulation cues
        assert len(spans) == 0

    def test_certification_without_collective_framing(self):
        """Certification mentioned in passing (non-collective) should be
        lower-scored for self-regulation due to exclusion cues."""
        detector = _make_detector()
        pages = _make_pages([
            "The individual company obtained certification for its "
            "operations as a voluntary commitment to sustainability."
        ])
        frame_cfg = detector.frames["self_regulation"]
        spans = detector._select_candidate_spans(pages, frame_cfg)
        # "certification" is an inclusion cue, but "voluntary commitment"
        # is an exclusion cue, so the score should be reduced
        if spans:
            assert all(s["score"] < 2.0 for s in spans), (
                "Certification with exclusion cues should not have high score"
            )


# ---------------------------------------------------------------
# Frame definition formatting
# ---------------------------------------------------------------

class TestFrameDefinitionFormatting:
    """Verify that the new YAML fields appear in formatted definitions."""

    def test_analytical_notes_included(self):
        detector = _make_detector()
        cfg = detector.frames["command_and_control"]
        definition = detector._format_frame_definition(cfg)

        assert "Analytical guidance:" in definition
        assert "legal compulsion" in definition

    def test_positive_examples_included(self):
        detector = _make_detector()
        cfg = detector.frames["economic_instruments"]
        definition = detector._format_frame_definition(cfg)

        assert "Positive examples:" in definition
        assert "PES programme" in definition

    def test_false_positive_notes_included(self):
        detector = _make_detector()
        cfg = detector.frames["information_strategies"]
        definition = detector._format_frame_definition(cfg)

        assert "False-positive guidance:" in definition
        assert "academic citations" in definition.lower() or "studies show" in definition

    def test_all_five_frames_present(self):
        """Verify exactly 5 frames loaded from config."""
        detector = _make_detector()
        expected_ids = {
            "command_and_control",
            "economic_instruments",
            "self_regulation",
            "voluntarism",
            "information_strategies",
        }
        assert set(detector.frames.keys()) == expected_ids


# ---------------------------------------------------------------
# Validation and edge cases
# ---------------------------------------------------------------

class TestValidation:
    """Edge cases in assessment validation."""

    def test_low_confidence_downgraded(self):
        """Present decision below threshold → insufficient_evidence."""
        detector = _make_detector(min_confidence=0.7)
        llm_output = FrameDetectionOutput(
            frame_id="command_and_control",
            decision=FrameDecision.PRESENT,
            confidence=0.5,
            evidence=[Evidence(page=1, quote="some quote that exists in page text somewhere here now")],
            rationale="Weak signal.",
        )
        frame_cfg = detector.frames["command_and_control"]
        pages = _make_pages(["some quote that exists in page text somewhere here now"])

        result = detector._validate_assessment(llm_output, frame_cfg, pages)
        assert result.decision == "insufficient_evidence"

    def test_invalid_quote_removed(self):
        """Evidence with non-existent quote is dropped."""
        detector = _make_detector()
        llm_output = FrameDetectionOutput(
            frame_id="command_and_control",
            decision=FrameDecision.PRESENT,
            confidence=0.9,
            evidence=[Evidence(page=1, quote="This quote does not exist in the page")],
            rationale="Looked relevant.",
        )
        frame_cfg = detector.frames["command_and_control"]
        pages = _make_pages(["Actual page text with different content entirely."])

        result = detector._validate_assessment(llm_output, frame_cfg, pages)
        # No valid evidence → downgraded
        assert result.decision == "insufficient_evidence"

    def test_fallback_on_exception(self):
        """Detection failure produces insufficient_evidence fallback for
        frames that had candidate spans; frames without spans return absent."""
        detector = _make_detector()
        detector.llm_client.detect_frame.side_effect = RuntimeError("API error")

        pages = _make_pages([
            "The government requires harvesting permits for logging."
        ])
        results = detector.detect_frames(pages)

        # All frames should have an assessment
        assert len(results) == 5
        for r in results:
            # Frames with no matching cues return absent; frames whose
            # LLM call fails return insufficient_evidence.
            assert r.decision in ("absent", "insufficient_evidence")

        # At least command_and_control should have had spans and hit the error
        cc = next(r for r in results if r.frame_id == "command_and_control")
        assert cc.decision == "insufficient_evidence"
        assert cc.confidence == 0.0


# ---------------------------------------------------------------
# Policy-mix regex cues
# ---------------------------------------------------------------

class TestPolicyMixCues:
    """Verify the policy-mix regex patterns match expected phrases."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "policy mix",
            "instrument mix",
            "mix of instruments",
            "combination of instruments",
            "combining policies",
            "combined measures",
            "complementarity",
            "complementary",
            "regulatory pluralism",
            "smart regulation",
            "hybrid governance",
            "integrated policy approach",
            "multi-instrument",
        ],
    )
    def test_policy_mix_cue_matches(self, phrase):
        """Each expected phrase should match at least one cue pattern."""
        matched = any(p.search(phrase) for p in _POLICY_MIX_CUES)
        assert matched, f"Policy-mix phrase not matched: '{phrase}'"
