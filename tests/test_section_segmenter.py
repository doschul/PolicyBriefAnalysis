"""
Tests for section segmentation module.
"""

import pytest

from src.policybrief.models import (
    DocumentSection,
    DocumentSectionMap,
    PageText,
    SectionLabel,
)
from src.policybrief.section_segmenter import SectionSegmenter


@pytest.fixture
def segmenter():
    return SectionSegmenter()


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


class TestHeadingNormalization:
    """Test heading detection and label assignment."""

    def test_known_headings_get_labels(self, segmenter):
        """Standard section headings should be detected and labeled."""
        pages = _make_pages({
            1: "Executive Summary\nThis brief examines climate policy options.",
            2: "Introduction\nClimate change poses significant risks.",
            3: "Recommendations\nGovernments should adopt carbon pricing.",
            4: "References\nSmith et al. (2024). Climate policy review.",
        })

        section_map = segmenter.segment_document(pages)

        labels = [s.normalized_label for s in section_map.sections]
        raw_titles = [s.raw_title for s in section_map.sections]

        assert SectionLabel.EXECUTIVE_SUMMARY in labels
        assert SectionLabel.INTRODUCTION in labels
        assert SectionLabel.RECOMMENDATIONS in labels
        assert SectionLabel.REFERENCES in labels

        # Raw titles preserved
        assert "Executive Summary" in raw_titles
        assert "Introduction" in raw_titles
        assert "Recommendations" in raw_titles
        assert "References" in raw_titles

    def test_numbered_headings(self, segmenter):
        """Numbered headings like '1. Introduction' should be recognized."""
        pages = _make_pages({
            1: "1. Introduction\nSome text here.\n2. Recommendations\nMore text.",
        })

        section_map = segmenter.segment_document(pages)
        labels = [s.normalized_label for s in section_map.sections]

        assert SectionLabel.INTRODUCTION in labels
        assert SectionLabel.RECOMMENDATIONS in labels

    def test_all_caps_headings(self, segmenter):
        """ALL-CAPS headings should be detected."""
        pages = _make_pages({
            1: "EXECUTIVE SUMMARY\nBrief overview of recommendations.",
            2: "CONCLUSION\nFinal thoughts on policy direction.",
        })

        section_map = segmenter.segment_document(pages)
        labels = [s.normalized_label for s in section_map.sections]

        assert SectionLabel.EXECUTIVE_SUMMARY in labels
        assert SectionLabel.CONCLUSION in labels

    def test_section_spans_are_correct(self, segmenter):
        """Sections should span from their heading to the next heading."""
        pages = _make_pages({
            1: "Introduction\nSome content on page 1.",
            2: "Body content continues on page 2.",
            3: "Conclusion\nFinal remarks.",
        })

        section_map = segmenter.segment_document(pages)

        # Find intro section
        intro = [s for s in section_map.sections if s.normalized_label == SectionLabel.INTRODUCTION]
        assert len(intro) == 1
        assert intro[0].start_page == 1
        assert intro[0].end_page == 3  # Extends to conclusion's page

        # Find conclusion
        conc = [s for s in section_map.sections if s.normalized_label == SectionLabel.CONCLUSION]
        assert len(conc) == 1
        assert conc[0].start_page == 3
        assert conc[0].end_page == 3


class TestNoHeadings:
    """Test documents with no clear headings."""

    def test_empty_pages(self, segmenter):
        """Empty input should return empty section map."""
        section_map = segmenter.segment_document([])
        assert section_map.sections == []
        assert section_map.detection_method == "none"

    def test_no_headings_returns_fallback_span(self, segmenter):
        """Documents with no detectable headings should return a single fallback span."""
        pages = _make_pages({
            1: "this is just plain body text without any headings or structure at all.",
            2: "more plain text. nothing that looks like a heading anywhere.",
        })

        section_map = segmenter.segment_document(pages)

        assert len(section_map.sections) == 1
        assert section_map.sections[0].raw_title is None
        assert section_map.sections[0].normalized_label is None
        assert section_map.sections[0].rule_source == "fallback"
        assert section_map.sections[0].start_page == 1
        assert section_map.sections[0].end_page == 2
        assert section_map.detection_method == "fallback"


class TestAmbiguousHeadings:
    """Test that ambiguous headings are handled conservatively."""

    def test_ambiguous_title_gets_no_label(self, segmenter):
        """A heading that doesn't match any known pattern should have no label."""
        pages = _make_pages({
            1: "Data Collection Methodology\nWe collected data from 100 sources.",
        })

        section_map = segmenter.segment_document(pages)

        # Should detect a heading but NOT assign a label
        labeled = [s for s in section_map.sections if s.normalized_label is not None]
        assert len(labeled) == 0, "Ambiguous headings should not get labels"

        # But the heading should still be detected
        titled = [s for s in section_map.sections if s.raw_title is not None]
        assert len(titled) >= 1

    def test_body_sentences_not_detected_as_headings(self, segmenter):
        """Regular sentences should not be mistaken for headings."""
        pages = _make_pages({
            1: (
                "This paper examines the impact of fiscal policy on economic growth.\n"
                "The analysis covers the period from 2010 to 2023.\n"
                "Results suggest that expansionary policy has limited effects."
            ),
        })

        section_map = segmenter.segment_document(pages)

        # All sentences end with '.' → should not be headings → fallback
        assert section_map.detection_method == "fallback"


class TestReferencesConservative:
    """Test that references section is recognized conservatively."""

    def test_references_heading_detected(self, segmenter):
        """A standalone 'References' heading should be detected and labeled."""
        pages = _make_pages({
            1: "Introduction\nSome content here.",
            2: "References\nSmith (2024). Climate Change Review.",
        })

        section_map = segmenter.segment_document(pages)
        ref_sections = [s for s in section_map.sections
                        if s.normalized_label == SectionLabel.REFERENCES]

        assert len(ref_sections) == 1
        assert ref_sections[0].raw_title == "References"

    def test_references_in_body_text_not_labeled(self, segmenter):
        """The word 'references' inside body text should not create a section."""
        pages = _make_pages({
            1: "this paragraph references several prior studies on the topic.",
        })

        section_map = segmenter.segment_document(pages)
        ref_sections = [s for s in section_map.sections
                        if s.normalized_label == SectionLabel.REFERENCES]

        assert len(ref_sections) == 0


class TestLayoutSignals:
    """Test that layout signals improve detection."""

    def test_layout_lines_boost_confidence(self, segmenter):
        """Lines with larger font / bold should have higher confidence."""
        pages = _make_pages({
            1: "Introduction\nBody text here.",
        })

        layout_lines = [
            {"page_num": 1, "text": "Introduction", "font_size": 16.0, "is_bold": True},
            {"page_num": 1, "text": "Body text here.", "font_size": 10.0, "is_bold": False},
        ]

        section_map = segmenter.segment_document(pages, layout_lines=layout_lines)

        assert len(section_map.sections) >= 1
        intro = [s for s in section_map.sections
                 if s.normalized_label == SectionLabel.INTRODUCTION]
        assert len(intro) == 1
        # Layout detection should yield higher confidence than text-only
        assert intro[0].confidence >= 0.50
        assert section_map.detection_method == "layout"

    def test_layout_without_text_fallback(self, segmenter):
        """If layout_lines is empty, text heuristics should still work."""
        pages = _make_pages({
            1: "Executive Summary\nKey findings of this brief.",
        })

        section_map = segmenter.segment_document(pages, layout_lines=[])

        assert section_map.detection_method == "text_heuristic"
        labels = [s.normalized_label for s in section_map.sections]
        assert SectionLabel.EXECUTIVE_SUMMARY in labels


class TestPipelineIntegration:
    """Test that section_map integrates with PerDocumentExtraction."""

    def test_section_map_serialization(self):
        """Section map should serialize/deserialize correctly."""
        section_map = DocumentSectionMap(
            sections=[
                DocumentSection(
                    raw_title="Introduction",
                    normalized_label=SectionLabel.INTRODUCTION,
                    start_page=1,
                    end_page=3,
                    confidence=0.65,
                    rule_source="text_heuristic",
                ),
                DocumentSection(
                    raw_title="Conclusion",
                    normalized_label=SectionLabel.CONCLUSION,
                    start_page=4,
                    end_page=5,
                    confidence=0.61,
                    rule_source="text_heuristic",
                ),
            ],
            detection_method="text_heuristic",
        )

        data = section_map.model_dump()
        restored = DocumentSectionMap.model_validate(data)

        assert len(restored.sections) == 2
        assert restored.sections[0].normalized_label == SectionLabel.INTRODUCTION
        assert restored.sections[1].raw_title == "Conclusion"

    def test_backward_compatible_heading_list(self, segmenter):
        """extract_headings() should return a flat list of raw titles."""
        pages = _make_pages({
            1: "Introduction\nContent.",
            2: "Conclusion\nFinal thoughts.",
        })

        headings = segmenter.extract_headings(pages)

        assert "Introduction" in headings
        assert "Conclusion" in headings
