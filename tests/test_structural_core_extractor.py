"""
Tests for structural core extraction module.
"""

import pytest

from src.policybrief.models import (
    ComponentStatus,
    DocumentSection,
    DocumentSectionMap,
    Evidence,
    ImplementationType,
    NarrativeHookType,
    PageText,
    SectionLabel,
    SolutionOptionType,
    StructuralCoreResult,
)
from src.policybrief.structural_core_extractor import StructuralCoreExtractor


@pytest.fixture
def extractor():
    return StructuralCoreExtractor()


def _make_pages(page_texts: dict[int, str]) -> list[PageText]:
    """Helper: create PageText objects from {page_num: text} dict."""
    pages = []
    for num, text in sorted(page_texts.items()):
        pages.append(PageText(
            page_num=num,
            text=text,
            char_count=len(text),
            word_count=len(text.split()),
        ))
    return pages


def _make_section_map(
    sections: list[tuple[str | None, SectionLabel | None, int, int]],
    method: str = "text_heuristic",
) -> DocumentSectionMap:
    """Helper: create a section map from (title, label, start, end) tuples."""
    return DocumentSectionMap(
        sections=[
            DocumentSection(
                raw_title=title,
                normalized_label=label,
                start_page=start,
                end_page=end,
                confidence=0.6,
                rule_source="text_heuristic",
            )
            for title, label, start, end in sections
        ],
        detection_method=method,
    )


# =====================================================================
# Problem identification
# =====================================================================

class TestProblemIdentification:
    """Test problem framing detection."""

    def test_explicit_problem_section(self, extractor):
        """A document with an explicit problem heading and discourse markers."""
        pages = _make_pages({
            1: "Introduction\nThis brief provides an overview.",
            2: (
                "Problem Definition\n"
                "The core issue facing European policy-makers is the lack of "
                "coordination between member states on carbon pricing."
            ),
            3: "Recommendations\nGovernments should adopt unified pricing.",
        })
        section_map = _make_section_map([
            ("Introduction", SectionLabel.INTRODUCTION, 1, 1),
            ("Problem Definition", SectionLabel.PROBLEM_DEFINITION, 2, 2),
            ("Recommendations", SectionLabel.RECOMMENDATIONS, 3, 3),
        ])

        result = extractor.extract(pages, section_map)

        assert result.problem.status == ComponentStatus.PRESENT
        assert result.problem.is_explicitly_labeled is True
        assert result.problem.matched_section == SectionLabel.PROBLEM_DEFINITION
        assert len(result.problem.evidence) >= 1
        assert "core issue" in result.problem.cues_matched
        assert result.labeling.problem_labeled is True

    def test_implicit_problem_weak(self, extractor):
        """A labeled problem section but without strong discourse markers."""
        pages = _make_pages({
            1: (
                "The Challenge\n"
                "Climate policy has evolved significantly over the past decade."
            ),
        })
        section_map = _make_section_map([
            ("The Challenge", SectionLabel.PROBLEM_DEFINITION, 1, 1),
        ])

        result = extractor.extract(pages, section_map)

        # Labeled section exists but no problem-framing language → weak
        assert result.problem.status == ComponentStatus.WEAK
        assert result.problem.is_explicitly_labeled is True

    def test_no_problem_absent(self, extractor):
        """A document with no problem framing at all."""
        pages = _make_pages({
            1: "This report presents findings from the annual survey of member states.",
            2: "The data collection methodology follows established protocols.",
        })

        result = extractor.extract(pages)

        assert result.problem.status == ComponentStatus.ABSENT
        assert result.problem.is_explicitly_labeled is False
        assert result.problem.evidence == []

    def test_problem_in_introduction(self, extractor):
        """Problem cues in an introduction section without explicit problem heading."""
        pages = _make_pages({
            1: (
                "Introduction\n"
                "This brief addresses the growing concern over antibiotic "
                "resistance in European livestock farming."
            ),
        })
        section_map = _make_section_map([
            ("Introduction", SectionLabel.INTRODUCTION, 1, 1),
        ])

        result = extractor.extract(pages, section_map)

        assert result.problem.status == ComponentStatus.PRESENT
        assert any(c in result.problem.cues_matched
                   for c in ("growing concern", "this brief addresses"))

    def test_empty_document(self, extractor):
        """Empty input should return all-absent defaults."""
        result = extractor.extract([])

        assert result.problem.status == ComponentStatus.ABSENT
        assert result.solutions == []
        assert result.implementation == []
        assert result.narrative_hook.status == ComponentStatus.ABSENT


# =====================================================================
# Solutions / policy options
# =====================================================================

class TestSolutionDetection:
    """Test solution and policy option detection."""

    def test_explicit_policy_options(self, extractor):
        """Document with explicitly named policy options."""
        pages = _make_pages({
            1: (
                "Policy Options\n"
                "Option one is to introduce a carbon tax at the national level. "
                "Option two is to expand the emissions trading scheme."
            ),
        })
        section_map = _make_section_map([
            ("Policy Options", SectionLabel.POLICY_OPTIONS, 1, 1),
        ])

        result = extractor.extract(pages, section_map)

        explicit = [s for s in result.solutions
                     if s.option_type == SolutionOptionType.EXPLICIT_OPTION]
        assert len(explicit) >= 1
        assert explicit[0].status == ComponentStatus.PRESENT
        assert result.labeling.solutions_labeled is True

    def test_generic_solution_discussion(self, extractor):
        """Document that discusses solutions generically without named options."""
        pages = _make_pages({
            1: (
                "A potential solution to this challenge is greater intergovernmental "
                "cooperation. The policy response must address both mitigation and adaptation."
            ),
        })

        result = extractor.extract(pages)

        generic = [s for s in result.solutions
                    if s.option_type == SolutionOptionType.GENERIC_DISCUSSION]
        assert len(generic) >= 1
        assert generic[0].status == ComponentStatus.PRESENT
        assert len(generic[0].evidence) >= 1

    def test_no_solutions(self, extractor):
        """Document with no solution or option language."""
        pages = _make_pages({
            1: "The survey results are presented in the following tables.",
            2: "Methodology was designed to ensure statistical significance.",
        })

        result = extractor.extract(pages)
        assert result.solutions == []

    def test_solutions_without_recommendation_language(self, extractor):
        """Solutions discussed without using 'should', 'must', 'recommend'."""
        pages = _make_pages({
            1: (
                "One intervention that has shown promise in pilot programmes is "
                "conditional cash transfers linked to school attendance."
            ),
        })

        result = extractor.extract(pages)

        generic = [s for s in result.solutions
                    if s.option_type == SolutionOptionType.GENERIC_DISCUSSION]
        assert len(generic) >= 1
        assert any("intervention" in c for c in generic[0].cues_matched)


# =====================================================================
# Implementation considerations
# =====================================================================

class TestImplementationDetection:
    """Test implementation consideration detection."""

    def test_barrier_detection(self, extractor):
        """Document mentioning implementation barriers."""
        pages = _make_pages({
            1: (
                "Implementation\n"
                "The main barriers to implementation include weak enforcement "
                "capacity and limited cross-border cooperation."
            ),
        })
        section_map = _make_section_map([
            ("Implementation", SectionLabel.IMPLEMENTATION, 1, 1),
        ])

        result = extractor.extract(pages, section_map)

        assert result.implementation_status == ComponentStatus.PRESENT
        assert result.implementation_is_explicitly_labeled is True
        barriers = [c for c in result.implementation
                     if c.consideration_type == ImplementationType.BARRIER]
        assert len(barriers) >= 1
        assert result.labeling.implementation_labeled is True

    def test_facilitator_detection(self, extractor):
        """Document mentioning enabling factors."""
        pages = _make_pages({
            1: (
                "Key enabling factors for successful adoption include strong "
                "political will and pre-existing regulatory frameworks."
            ),
        })

        result = extractor.extract(pages)

        facilitators = [c for c in result.implementation
                         if c.consideration_type == ImplementationType.FACILITATOR]
        assert len(facilitators) >= 1

    def test_resource_and_institutional(self, extractor):
        """Document with resource and institutional considerations."""
        pages = _make_pages({
            1: (
                "The cost estimate for the first phase is EUR 50 million. "
                "Institutional capacity must be strengthened before rollout. "
                "The governance structure needs reform to accommodate new mandates."
            ),
        })

        result = extractor.extract(pages)

        types_found = {c.consideration_type for c in result.implementation}
        assert ImplementationType.RESOURCE in types_found
        assert ImplementationType.INSTITUTIONAL in types_found

    def test_no_implementation(self, extractor):
        """Document with no implementation content."""
        pages = _make_pages({
            1: "This report summarises the findings of the annual member state survey.",
        })

        result = extractor.extract(pages)

        assert result.implementation_status == ComponentStatus.ABSENT
        assert result.implementation == []

    def test_feasibility_detection(self, extractor):
        """Document discussing feasibility."""
        pages = _make_pages({
            1: (
                "The feasibility of this approach depends on available technical "
                "infrastructure and political alignment across jurisdictions."
            ),
        })

        result = extractor.extract(pages)

        feasibility = [c for c in result.implementation
                        if c.consideration_type == ImplementationType.FEASIBILITY]
        assert len(feasibility) >= 1


# =====================================================================
# Narrative / storytelling hooks
# =====================================================================

class TestNarrativeDetection:
    """Test narrative hook detection."""

    def test_case_vignette(self, extractor):
        """Document opening with a case vignette."""
        pages = _make_pages({
            1: (
                "Introduction\n"
                "Consider the case of Denmark, where a comprehensive carbon "
                "tax introduced in 1992 led to a 25% reduction in emissions."
            ),
        })

        result = extractor.extract(pages)

        assert result.narrative_hook.status == ComponentStatus.PRESENT
        assert result.narrative_hook.hook_type == NarrativeHookType.CASE_VIGNETTE
        assert result.narrative_hook.page == 1

    def test_vivid_example(self, extractor):
        """Document with a vivid example."""
        pages = _make_pages({
            1: (
                "A striking example of policy failure occurred in the summer of "
                "2019 when three consecutive heatwaves overwhelmed urban health systems."
            ),
        })

        result = extractor.extract(pages)

        assert result.narrative_hook.status == ComponentStatus.PRESENT
        assert result.narrative_hook.hook_type == NarrativeHookType.VIVID_EXAMPLE

    def test_non_narrative_background(self, extractor):
        """A plain background section should not be detected as narrative."""
        pages = _make_pages({
            1: (
                "Background\n"
                "European climate policy has evolved significantly since the "
                "adoption of the Kyoto Protocol in 1997."
            ),
        })

        result = extractor.extract(pages)

        assert result.narrative_hook.status == ComponentStatus.ABSENT

    def test_narrative_not_in_references(self, extractor):
        """Narrative cues in references section should be ignored."""
        pages = _make_pages({
            1: "Introduction\nThis report examines fiscal policy trends across the region.",
            2: "References\nSmith, J. (2023). Consider the case of Portugal: A fiscal analysis.",
        })
        section_map = _make_section_map([
            ("Introduction", SectionLabel.INTRODUCTION, 1, 1),
            ("References", SectionLabel.REFERENCES, 2, 2),
        ])

        result = extractor.extract(pages, section_map)

        # "Consider the case of" is on page 2 which is in the references section.
        # The narrative detector should only check early pages / intro sections.
        assert result.narrative_hook.status == ComponentStatus.ABSENT


# =====================================================================
# Labeling assessment
# =====================================================================

class TestLabelingAssessment:
    """Test the labeling assessment component."""

    def test_all_labeled(self, extractor):
        """Document where all three core components are labeled by headings."""
        pages = _make_pages({
            1: "Problem Definition\nThe core issue is the lack of cross-border alignment.",
            2: "Policy Options\nOption one is a carbon tax at national level.",
            3: "Implementation\nThe main barriers to implementation include weak enforcement.",
        })
        section_map = _make_section_map([
            ("Problem Definition", SectionLabel.PROBLEM_DEFINITION, 1, 1),
            ("Policy Options", SectionLabel.POLICY_OPTIONS, 2, 2),
            ("Implementation", SectionLabel.IMPLEMENTATION, 3, 3),
        ])

        result = extractor.extract(pages, section_map)

        assert result.labeling.problem_labeled is True
        assert result.labeling.solutions_labeled is True
        assert result.labeling.implementation_labeled is True

    def test_none_labeled(self, extractor):
        """Document where components are present only in prose."""
        pages = _make_pages({
            1: (
                "The key challenge is rising sea levels along the North Sea coast. "
                "A potential solution is to invest in advanced flood barriers. "
                "The main barriers to implementation include budget constraints."
            ),
        })

        result = extractor.extract(pages)

        assert result.labeling.problem_labeled is False
        assert result.labeling.solutions_labeled is False
        assert result.labeling.implementation_labeled is False
        # But content should still be detected
        assert result.problem.status == ComponentStatus.PRESENT
        assert len(result.solutions) >= 1
        assert result.implementation_status == ComponentStatus.PRESENT


# =====================================================================
# Structural core result serialization
# =====================================================================

class TestSerialization:
    """Test that StructuralCoreResult serializes cleanly."""

    def test_default_result_serializes(self):
        """An all-absent result should serialize and deserialize."""
        result = StructuralCoreResult()
        data = result.model_dump()

        assert data["problem"]["status"] == "absent"
        assert data["solutions"] == []
        assert data["implementation"] == []
        assert data["narrative_hook"]["status"] == "absent"

        restored = StructuralCoreResult.model_validate(data)
        assert restored.problem.status == ComponentStatus.ABSENT

    def test_full_result_roundtrip(self, extractor):
        """A populated result should survive serialize → deserialize."""
        pages = _make_pages({
            1: "Problem Definition\nThe core issue is antimicrobial resistance in livestock.",
            2: "Policy Options\nOption one is to ban prophylactic antibiotic use.",
            3: "Implementation\nThe main barriers to implementation include industry lobbying.",
        })
        section_map = _make_section_map([
            ("Problem Definition", SectionLabel.PROBLEM_DEFINITION, 1, 1),
            ("Policy Options", SectionLabel.POLICY_OPTIONS, 2, 2),
            ("Implementation", SectionLabel.IMPLEMENTATION, 3, 3),
        ])

        result = extractor.extract(pages, section_map)
        data = result.model_dump()
        restored = StructuralCoreResult.model_validate(data)

        assert restored.problem.status == result.problem.status
        assert len(restored.solutions) == len(result.solutions)
        assert len(restored.implementation) == len(result.implementation)
