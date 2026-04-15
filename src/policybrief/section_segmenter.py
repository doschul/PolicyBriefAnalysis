"""
Deterministic section segmenter for policy brief documents.

Creates a structural map of document sections using layout signals (font size,
bold) when available from fitz, with regex/text heuristics as fallback.
No LLM is used. Conservative: false positives are worse than missing labels.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from .models import (
    DocumentSection,
    DocumentSectionMap,
    PageText,
    SectionLabel,
)

logger = logging.getLogger(__name__)


# Regex patterns mapping normalized labels to heading text patterns.
# Order matters: more specific patterns first within each label.
LABEL_PATTERNS: Dict[SectionLabel, List[re.Pattern]] = {
    SectionLabel.EXECUTIVE_SUMMARY: [
        re.compile(r"^executive\s+summary$", re.IGNORECASE),
        re.compile(r"^summary$", re.IGNORECASE),
    ],
    SectionLabel.KEY_MESSAGES: [
        re.compile(r"^key\s+messages?$", re.IGNORECASE),
        re.compile(r"^key\s+findings?$", re.IGNORECASE),
        re.compile(r"^key\s+points?$", re.IGNORECASE),
        re.compile(r"^highlights?$", re.IGNORECASE),
    ],
    SectionLabel.INTRODUCTION: [
        re.compile(r"^introduction$", re.IGNORECASE),
        re.compile(r"^background$", re.IGNORECASE),
        re.compile(r"^overview$", re.IGNORECASE),
        re.compile(r"^context$", re.IGNORECASE),
    ],
    SectionLabel.PROBLEM_DEFINITION: [
        re.compile(r"^problem\s+(definition|statement)$", re.IGNORECASE),
        re.compile(r"^the\s+problem$", re.IGNORECASE),
        re.compile(r"^issue\s+(definition|overview)$", re.IGNORECASE),
        re.compile(r"^challenges?$", re.IGNORECASE),
    ],
    SectionLabel.POLICY_OPTIONS: [
        re.compile(r"^policy\s+options?$", re.IGNORECASE),
        re.compile(r"^options?\s+analysis$", re.IGNORECASE),
        re.compile(r"^policy\s+alternatives?$", re.IGNORECASE),
    ],
    SectionLabel.RECOMMENDATIONS: [
        re.compile(r"^recommendations?$", re.IGNORECASE),
        re.compile(r"^policy\s+recommendations?$", re.IGNORECASE),
    ],
    SectionLabel.IMPLEMENTATION: [
        re.compile(r"^implementation(\s+plan)?$", re.IGNORECASE),
        re.compile(r"^action\s+plan$", re.IGNORECASE),
        re.compile(r"^next\s+steps$", re.IGNORECASE),
    ],
    SectionLabel.CONCLUSION: [
        re.compile(r"^conclusions?$", re.IGNORECASE),
        re.compile(r"^concluding\s+remarks?$", re.IGNORECASE),
    ],
    SectionLabel.REFERENCES: [
        re.compile(r"^references?$", re.IGNORECASE),
        re.compile(r"^bibliography$", re.IGNORECASE),
        re.compile(r"^works?\s+cited$", re.IGNORECASE),
    ],
    SectionLabel.ACKNOWLEDGEMENTS: [
        re.compile(r"^acknowledg[e]?ments?$", re.IGNORECASE),
    ],
    SectionLabel.ABOUT_AUTHORS: [
        re.compile(r"^about\s+the\s+authors?$", re.IGNORECASE),
        re.compile(r"^authors?$", re.IGNORECASE),
        re.compile(r"^author\s+biograph(y|ies)$", re.IGNORECASE),
    ],
    SectionLabel.CONTACT: [
        re.compile(r"^contact(\s+information)?$", re.IGNORECASE),
        re.compile(r"^for\s+more\s+information$", re.IGNORECASE),
    ],
    SectionLabel.APPENDIX: [
        re.compile(r"^app?endix", re.IGNORECASE),
        re.compile(r"^annex", re.IGNORECASE),
    ],
}

# Minimum confidence to keep a candidate heading (text-only detection).
_MIN_CANDIDATE_CONFIDENCE = 0.30

# Minimum confidence to assign a normalized label to a section.
_MIN_LABEL_CONFIDENCE = 0.40


class SectionSegmenter:
    """Deterministic section segmenter using layout + text heuristics."""

    def segment_document(
        self,
        pages: List[PageText],
        layout_lines: Optional[List[Dict]] = None,
    ) -> DocumentSectionMap:
        """Build a section map for a document.

        Args:
            pages: Extracted page texts.
            layout_lines: Optional list of dicts with keys:
                page_num (int, 1-based), text (str), font_size (float),
                is_bold (bool).  Produced by pdf_extractor when using fitz.

        Returns:
            DocumentSectionMap with detected sections.
        """
        if not pages:
            return DocumentSectionMap(sections=[], detection_method="none")

        # Phase 1: detect heading candidates
        candidates = self._detect_candidates(pages, layout_lines or [])

        if not candidates:
            # No headings found — return one fallback span covering the whole doc
            fallback = DocumentSection(
                raw_title=None,
                normalized_label=None,
                start_page=pages[0].page_num,
                end_page=pages[-1].page_num,
                confidence=0.0,
                rule_source="fallback",
            )
            return DocumentSectionMap(
                sections=[fallback], detection_method="fallback"
            )

        # Phase 2: build ordered, non-overlapping sections from candidates
        sections = self._build_sections(candidates, pages)

        method = "layout" if layout_lines else "text_heuristic"
        return DocumentSectionMap(sections=sections, detection_method=method)

    def extract_headings(
        self,
        pages: List[PageText],
        layout_lines: Optional[List[Dict]] = None,
    ) -> List[str]:
        """Return a flat heading list (backward-compatible helper)."""
        section_map = self.segment_document(pages, layout_lines)
        return [
            s.raw_title
            for s in section_map.sections
            if s.raw_title is not None
        ]

    # ------------------------------------------------------------------
    # Internal: candidate detection
    # ------------------------------------------------------------------

    def _detect_candidates(
        self,
        pages: List[PageText],
        layout_lines: List[Dict],
    ) -> List[Dict]:
        """Return heading candidate dicts sorted by (page, position).

        Each candidate dict: {
            "text": str,           # cleaned heading text
            "page_num": int,       # 1-based
            "confidence": float,   # 0-1
            "rule_source": str,    # "layout" | "text_heuristic"
        }
        """
        candidates: List[Dict] = []

        # --- layout-based detection (preferred) ---
        if layout_lines:
            candidates.extend(self._candidates_from_layout(layout_lines))

        # --- text-heuristic fallback (always run, deduplicated later) ---
        text_candidates = self._candidates_from_text(pages)
        candidates.extend(text_candidates)

        # Deduplicate: if layout and text found the same heading on the same
        # page, keep the one with higher confidence.
        candidates = self._deduplicate(candidates)

        # Filter low-confidence noise
        candidates = [
            c for c in candidates if c["confidence"] >= _MIN_CANDIDATE_CONFIDENCE
        ]

        # Sort by page then position within page
        candidates.sort(key=lambda c: (c["page_num"], c.get("_sort_key", 0)))

        return candidates

    def _candidates_from_layout(self, layout_lines: List[Dict]) -> List[Dict]:
        """Detect headings using font-size / bold signals."""
        if not layout_lines:
            return []

        # Compute median font size for comparison baseline
        sizes = [ll["font_size"] for ll in layout_lines if ll.get("font_size")]
        if not sizes:
            return []
        sizes.sort()
        median_size = sizes[len(sizes) // 2]

        candidates = []
        for i, ll in enumerate(layout_lines):
            text = (ll.get("text") or "").strip()
            if not text or not self._looks_like_heading(text):
                continue

            font_size = ll.get("font_size", 0)
            is_bold = ll.get("is_bold", False)

            # Layout confidence starts at 0.35 (above text-only 0.28)
            conf = 0.35

            # Bonus for larger-than-median font
            if median_size > 0 and font_size > median_size * 1.15:
                conf += 0.20
            # Bonus for bold
            if is_bold:
                conf += 0.12
            # Bonus if it matches a known label pattern
            if self._match_label(text) is not None:
                conf += 0.15

            conf = min(conf, 1.0)

            candidates.append({
                "text": text,
                "page_num": ll.get("page_num", 1),
                "confidence": round(conf, 2),
                "rule_source": "layout",
                "_sort_key": i,
            })

        return candidates

    def _candidates_from_text(self, pages: List[PageText]) -> List[Dict]:
        """Detect headings purely from text heuristics."""
        candidates = []
        for page in pages:
            lines = page.text.split("\n")
            for line_idx, raw_line in enumerate(lines):
                text = raw_line.strip()
                if not text or not self._looks_like_heading(text):
                    continue

                conf = 0.28  # text-heuristic baseline

                if self._is_structured_heading(text):
                    conf += 0.18  # title case / all-caps / numbered

                if self._match_label(text) is not None:
                    conf += 0.15  # matches known label

                conf = min(conf, 1.0)

                candidates.append({
                    "text": text,
                    "page_num": page.page_num,
                    "confidence": round(conf, 2),
                    "rule_source": "text_heuristic",
                    "_sort_key": line_idx,
                })

        return candidates

    # ------------------------------------------------------------------
    # Internal: heading shape tests
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_like_heading(text: str) -> bool:
        """Quick filter: is this line plausibly a heading?"""
        if len(text) > 120 or len(text) < 2:
            return False
        word_count = len(text.split())
        if word_count < 1 or word_count > 15:
            return False
        if text.endswith(".") or text.endswith(","):
            return False
        return True

    @staticmethod
    def _is_structured_heading(text: str) -> bool:
        """Does the text have heading-like formatting?"""
        # Numbered heading: "1. Introduction", "2.3 Policy Options"
        if re.match(r"^\d+(\.\d+)*\.?\s+[A-Z]", text):
            return True
        # All-caps: "RECOMMENDATIONS"
        if text.isupper() and len(text.split()) <= 8:
            return True
        # Title case: "Executive Summary" (allow small words lowercase)
        if text.istitle():
            return True
        return False

    # ------------------------------------------------------------------
    # Internal: label matching
    # ------------------------------------------------------------------

    @staticmethod
    def _match_label(text: str) -> Optional[SectionLabel]:
        """Return the best matching SectionLabel or None."""
        # Strip leading numbers: "3. Recommendations" → "Recommendations"
        cleaned = re.sub(r"^\d+(\.\d+)*\.?\s*", "", text).strip()
        for label, patterns in LABEL_PATTERNS.items():
            for pat in patterns:
                if pat.search(cleaned):
                    return label
        return None

    # ------------------------------------------------------------------
    # Internal: dedup, section building
    # ------------------------------------------------------------------

    def _deduplicate(self, candidates: List[Dict]) -> List[Dict]:
        """Keep highest-confidence candidate per (page, normalized_text)."""
        best: Dict[Tuple[int, str], Dict] = {}
        for c in candidates:
            key = (c["page_num"], c["text"].lower().strip())
            if key not in best or c["confidence"] > best[key]["confidence"]:
                best[key] = c
        return list(best.values())

    def _build_sections(
        self,
        candidates: List[Dict],
        pages: List[PageText],
    ) -> List[DocumentSection]:
        """Convert heading candidates into an ordered list of DocumentSections."""
        last_page = pages[-1].page_num

        sections: List[DocumentSection] = []
        for i, cand in enumerate(candidates):
            # End page = page before next heading, or last page
            if i + 1 < len(candidates):
                end_page = candidates[i + 1]["page_num"]
                # If next heading is on same page, end_page == start_page
                if end_page > cand["page_num"]:
                    end_page = end_page  # next heading's page is the boundary
            else:
                end_page = last_page

            raw_title = cand["text"]
            confidence = cand["confidence"]
            label = self._match_label(raw_title)

            # Only assign label if confidence is sufficient
            if label is not None and confidence < _MIN_LABEL_CONFIDENCE:
                label = None

            sections.append(
                DocumentSection(
                    raw_title=raw_title,
                    normalized_label=label,
                    start_page=cand["page_num"],
                    end_page=end_page,
                    confidence=confidence,
                    rule_source=cand["rule_source"],
                )
            )

        return sections
