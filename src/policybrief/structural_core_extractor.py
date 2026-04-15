"""
Deterministic structural-core extractor for policy briefs.

Detects core analytical components — problem identification, solutions/options,
implementation considerations, labeling assessment, and narrative hooks — using
discourse markers, section labels, and text heuristics.  No LLM is used.

Design principles:
- Conservative: missing uncertain content is better than inventing structure.
- Null-first: every component defaults to absent with empty evidence.
- Reproducible: same input always produces the same output.
- Evidence-traceable: every detected component carries page numbers and quotes.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from .models import (
    ComponentStatus,
    DocumentSectionMap,
    Evidence,
    ImplementationConsideration,
    ImplementationType,
    LabelingAssessment,
    NarrativeHook,
    NarrativeHookType,
    PageText,
    ProblemIdentification,
    SectionLabel,
    SolutionOption,
    SolutionOptionType,
    StructuralCoreResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Discourse-marker lexicons
# ---------------------------------------------------------------------------

# Problem-framing cues — must appear as distinct phrases, not substrings of
# unrelated words.  Each entry is compiled to a word-boundary regex.
_PROBLEM_CUES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bthe\s+problem\b", re.I), "the problem"),
    (re.compile(r"\bpolicy\s+problem\b", re.I), "policy problem"),
    (re.compile(r"\bpolicy\s+challenge\b", re.I), "policy challenge"),
    (re.compile(r"\bkey\s+challenge\b", re.I), "key challenge"),
    (re.compile(r"\bmajor\s+challenge\b", re.I), "major challenge"),
    (re.compile(r"\bcore\s+issue\b", re.I), "core issue"),
    (re.compile(r"\bcentral\s+issue\b", re.I), "central issue"),
    (re.compile(r"\bkey\s+issue\b", re.I), "key issue"),
    (re.compile(r"\bcritical\s+issue\b", re.I), "critical issue"),
    (re.compile(r"\bproblem\s+definition\b", re.I), "problem definition"),
    (re.compile(r"\bproblem\s+statement\b", re.I), "problem statement"),
    (re.compile(r"\bthis\s+(?:brief|paper|report)\s+addresses\b", re.I), "this brief addresses"),
    (re.compile(r"\bthe\s+crisis\b", re.I), "the crisis"),
    (re.compile(r"\bgrowing\s+concern\b", re.I), "growing concern"),
    (re.compile(r"\burgent\s+need\b", re.I), "urgent need"),
    (re.compile(r"\bfundamental\s+gap\b", re.I), "fundamental gap"),
    (re.compile(r"\bpersistent\s+failure\b", re.I), "persistent failure"),
]

# Solution / policy-option cues
_SOLUTION_GENERIC_CUES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bpotential\s+solution\b", re.I), "potential solution"),
    (re.compile(r"\bpossible\s+solution\b", re.I), "possible solution"),
    (re.compile(r"\bproposed\s+solution\b", re.I), "proposed solution"),
    (re.compile(r"\bpolicy\s+response\b", re.I), "policy response"),
    (re.compile(r"\bpolicy\s+approach\b", re.I), "policy approach"),
    (re.compile(r"\bto\s+address\s+this\b", re.I), "to address this"),
    (re.compile(r"\bstrategy\s+for\b", re.I), "strategy for"),
    (re.compile(r"\bintervention\b", re.I), "intervention"),
    (re.compile(r"\bmeasures?\s+(?:to|for|that)\b", re.I), "measures to/for"),
]

_SOLUTION_EXPLICIT_CUES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\boption\s+(?:one|two|three|four|five|1|2|3|4|5|[a-e])\b", re.I), "option N"),
    (re.compile(r"\bpolicy\s+option\b", re.I), "policy option"),
    (re.compile(r"\balternative\s+(?:one|two|three|1|2|3|[a-c])\b", re.I), "alternative N"),
    (re.compile(r"\bscenario\s+(?:one|two|three|1|2|3|[a-c])\b", re.I), "scenario N"),
    (re.compile(r"\bfirst\s+option\b", re.I), "first option"),
    (re.compile(r"\bsecond\s+option\b", re.I), "second option"),
    (re.compile(r"\bthird\s+option\b", re.I), "third option"),
]

# Implementation-consideration cues, keyed by sub-type
_IMPL_CUES: Dict[ImplementationType, List[Tuple[re.Pattern, str]]] = {
    ImplementationType.BARRIER: [
        (re.compile(r"\bbarrier(?:s)?\s+to\b", re.I), "barrier to"),
        (re.compile(r"\bobstacle(?:s)?\b", re.I), "obstacle"),
        (re.compile(r"\bimpediment(?:s)?\b", re.I), "impediment"),
        (re.compile(r"\bchallenge(?:s)?\s+(?:in|to|of)\s+implement", re.I), "challenge in implementation"),
        (re.compile(r"\bdifficult(?:y|ies)\s+(?:in|of)\s+implement", re.I), "difficulty in implementation"),
    ],
    ImplementationType.FACILITATOR: [
        (re.compile(r"\benabling\s+(?:factors?|conditions?)\b", re.I), "enabling factor"),
        (re.compile(r"\bfacilitators?\b", re.I), "facilitator"),
        (re.compile(r"\bdrivers?\s+of\s+(?:success|adoption)\b", re.I), "driver of success"),
        (re.compile(r"\bpreconditions?\b", re.I), "precondition"),
    ],
    ImplementationType.FEASIBILITY: [
        (re.compile(r"\bfeasibil", re.I), "feasibility"),
        (re.compile(r"\bviabil", re.I), "viability"),
        (re.compile(r"\bpracticab", re.I), "practicability"),
    ],
    ImplementationType.SEQUENCING: [
        (re.compile(r"\bsequenc(?:e|ing)\b", re.I), "sequencing"),
        (re.compile(r"\bphased?\s+(?:approach|implementation|rollout)\b", re.I), "phased approach"),
        (re.compile(r"\bstep\s+(?:one|two|three|1|2|3)\b", re.I), "step N"),
        (re.compile(r"\btimeline\b", re.I), "timeline"),
    ],
    ImplementationType.RESOURCE: [
        (re.compile(r"\bresource\s+(?:requirement|need|implication|constraint)\b", re.I), "resource requirement"),
        (re.compile(r"\bcost\s+(?:estimate|implication|consideration)\b", re.I), "cost estimate"),
        (re.compile(r"\bbudget(?:ary)?\s+(?:implication|requirement|constraint)\b", re.I), "budget implication"),
        (re.compile(r"\bfunding\s+(?:requirement|need|gap)\b", re.I), "funding requirement"),
        (re.compile(r"\bcapacity\s+(?:need|gap|constraint|building)\b", re.I), "capacity need"),
    ],
    ImplementationType.INSTITUTIONAL: [
        (re.compile(r"\binstitutional\s+(?:requirement|capacity|arrangement|reform|framework)\b", re.I), "institutional requirement"),
        (re.compile(r"\bgovernance\s+(?:structure|mechanism|arrangement|framework)\b", re.I), "governance structure"),
        (re.compile(r"\bregulatory\s+(?:framework|reform|requirement)\b", re.I), "regulatory framework"),
        (re.compile(r"\bstakeholder\s+(?:coordination|engagement|buy-in)\b", re.I), "stakeholder coordination"),
    ],
    ImplementationType.RISK: [
        (re.compile(r"\bimplementation\s+risk\b", re.I), "implementation risk"),
        (re.compile(r"\bunintended\s+consequence\b", re.I), "unintended consequence"),
        (re.compile(r"\bdownside\s+risk\b", re.I), "downside risk"),
        (re.compile(r"\bpotential\s+(?:pitfall|risk)\b", re.I), "potential risk"),
    ],
}

# Narrative-hook cues
_NARRATIVE_CUES: Dict[NarrativeHookType, List[Tuple[re.Pattern, str]]] = {
    NarrativeHookType.CASE_VIGNETTE: [
        (re.compile(r"\bin\s+(?:the\s+)?case\s+of\b", re.I), "in the case of"),
        (re.compile(r"\bconsider\s+the\s+(?:case|example)\s+of\b", re.I), "consider the case of"),
        (re.compile(r"\ba\s+case\s+study\b", re.I), "a case study"),
    ],
    NarrativeHookType.ANECDOTE: [
        (re.compile(r"\bwhen\s+\w+\s+first\s+(?:arrived|visited|saw|encountered)\b", re.I), "when X first"),
        (re.compile(r"\bimagine\s+(?:a|that|you)\b", re.I), "imagine"),
        (re.compile(r"\bpicture\s+(?:a|this)\b", re.I), "picture"),
    ],
    NarrativeHookType.VIVID_EXAMPLE: [
        (re.compile(r"\bfor\s+(?:example|instance),?\s+in\b", re.I), "for example, in"),
        (re.compile(r"\ba\s+(?:striking|vivid|dramatic|notable|telling)\s+example\b", re.I), "a striking example"),
    ],
    NarrativeHookType.NARRATIVE_OPENING: [
        (re.compile(r"\bon\s+\w+\s+\d{1,2},?\s+\d{4}\b", re.I), "on [date]"),
        (re.compile(r"\bin\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b", re.I), "in [month year]"),
    ],
}

# Sections relevant to each component
_PROBLEM_SECTIONS: Set[SectionLabel] = {
    SectionLabel.INTRODUCTION,
    SectionLabel.PROBLEM_DEFINITION,
    SectionLabel.EXECUTIVE_SUMMARY,
    SectionLabel.KEY_MESSAGES,
}

_SOLUTION_SECTIONS: Set[SectionLabel] = {
    SectionLabel.POLICY_OPTIONS,
    SectionLabel.RECOMMENDATIONS,
    SectionLabel.EXECUTIVE_SUMMARY,
    SectionLabel.KEY_MESSAGES,
}

_IMPL_SECTIONS: Set[SectionLabel] = {
    SectionLabel.IMPLEMENTATION,
    SectionLabel.POLICY_OPTIONS,
    SectionLabel.RECOMMENDATIONS,
}

# Max evidence quote length (must be ≥10 for Evidence model)
_MAX_QUOTE = 450
_MIN_QUOTE = 10


class StructuralCoreExtractor:
    """Deterministic extractor for structural core components."""

    def extract(
        self,
        pages: List[PageText],
        section_map: Optional[DocumentSectionMap] = None,
    ) -> StructuralCoreResult:
        """Extract structural core components from a document.

        Args:
            pages: Extracted page texts.
            section_map: Optional section map from the segmenter.

        Returns:
            StructuralCoreResult with all component analyses.
        """
        if not pages:
            return StructuralCoreResult()

        problem = self._detect_problem(pages, section_map)
        solutions = self._detect_solutions(pages, section_map)
        impl_considerations, impl_status, impl_section, impl_labeled = (
            self._detect_implementation(pages, section_map)
        )
        narrative = self._detect_narrative(pages, section_map)
        labeling = LabelingAssessment(
            problem_labeled=problem.is_explicitly_labeled,
            solutions_labeled=any(s.is_explicitly_labeled for s in solutions),
            implementation_labeled=impl_labeled,
        )

        return StructuralCoreResult(
            problem=problem,
            solutions=solutions,
            implementation=impl_considerations,
            implementation_status=impl_status,
            implementation_matched_section=impl_section,
            implementation_is_explicitly_labeled=impl_labeled,
            narrative_hook=narrative,
            labeling=labeling,
        )

    # ------------------------------------------------------------------
    # Problem identification
    # ------------------------------------------------------------------

    def _detect_problem(
        self,
        pages: List[PageText],
        section_map: Optional[DocumentSectionMap],
    ) -> ProblemIdentification:
        # Check if there's a heading-labeled problem section
        is_labeled, labeled_section = self._has_labeled_section(
            section_map, _PROBLEM_SECTIONS
        )

        # Gather candidate pages: prefer labeled sections, scan all as fallback
        target_pages = self._pages_for_sections(
            pages, section_map, _PROBLEM_SECTIONS
        ) or pages

        cues_found: List[str] = []
        evidence_spans: List[Evidence] = []

        for page in target_pages:
            for sent in self._iter_sentences(page.text):
                for pat, cue_name in _PROBLEM_CUES:
                    if pat.search(sent):
                        ev = self._make_evidence(page.page_num, sent)
                        if ev is not None:
                            evidence_spans.append(ev)
                            if cue_name not in cues_found:
                                cues_found.append(cue_name)
                        break  # one cue per sentence is enough

        # Determine status
        if evidence_spans:
            status = ComponentStatus.PRESENT
        elif is_labeled:
            # A heading says "Problem" but no discourse markers found → weak
            status = ComponentStatus.WEAK
        else:
            status = ComponentStatus.ABSENT

        return ProblemIdentification(
            status=status,
            matched_section=labeled_section,
            evidence=evidence_spans[:5],  # cap at 5 spans
            cues_matched=cues_found,
            is_explicitly_labeled=is_labeled,
        )

    # ------------------------------------------------------------------
    # Solutions / policy options
    # ------------------------------------------------------------------

    def _detect_solutions(
        self,
        pages: List[PageText],
        section_map: Optional[DocumentSectionMap],
    ) -> List[SolutionOption]:
        is_labeled, labeled_section = self._has_labeled_section(
            section_map, _SOLUTION_SECTIONS
        )

        target_pages = self._pages_for_sections(
            pages, section_map, _SOLUTION_SECTIONS
        ) or pages

        generic_cues: List[str] = []
        generic_evidence: List[Evidence] = []
        explicit_cues: List[str] = []
        explicit_evidence: List[Evidence] = []

        for page in target_pages:
            for sent in self._iter_sentences(page.text):
                # Check explicit option cues first (more specific)
                matched_explicit = False
                for pat, cue_name in _SOLUTION_EXPLICIT_CUES:
                    if pat.search(sent):
                        ev = self._make_evidence(page.page_num, sent)
                        if ev is not None:
                            explicit_evidence.append(ev)
                            if cue_name not in explicit_cues:
                                explicit_cues.append(cue_name)
                        matched_explicit = True
                        break

                if not matched_explicit:
                    for pat, cue_name in _SOLUTION_GENERIC_CUES:
                        if pat.search(sent):
                            ev = self._make_evidence(page.page_num, sent)
                            if ev is not None:
                                generic_evidence.append(ev)
                                if cue_name not in generic_cues:
                                    generic_cues.append(cue_name)
                            break

        results: List[SolutionOption] = []

        if explicit_evidence:
            results.append(SolutionOption(
                status=ComponentStatus.PRESENT,
                option_type=SolutionOptionType.EXPLICIT_OPTION,
                matched_section=labeled_section,
                evidence=explicit_evidence[:5],
                cues_matched=explicit_cues,
                is_explicitly_labeled=is_labeled,
            ))

        if generic_evidence:
            status = ComponentStatus.PRESENT if generic_cues else ComponentStatus.WEAK
            results.append(SolutionOption(
                status=status,
                option_type=SolutionOptionType.GENERIC_DISCUSSION,
                matched_section=labeled_section,
                evidence=generic_evidence[:5],
                cues_matched=generic_cues,
                is_explicitly_labeled=is_labeled,
            ))

        # If a labeled section exists but no cues found, record as weak
        if not results and is_labeled:
            results.append(SolutionOption(
                status=ComponentStatus.WEAK,
                option_type=SolutionOptionType.GENERIC_DISCUSSION,
                matched_section=labeled_section,
                evidence=[],
                cues_matched=[],
                is_explicitly_labeled=True,
            ))

        return results

    # ------------------------------------------------------------------
    # Implementation considerations
    # ------------------------------------------------------------------

    def _detect_implementation(
        self,
        pages: List[PageText],
        section_map: Optional[DocumentSectionMap],
    ) -> Tuple[
        List[ImplementationConsideration],
        ComponentStatus,
        Optional[SectionLabel],
        bool,
    ]:
        is_labeled, labeled_section = self._has_labeled_section(
            section_map, _IMPL_SECTIONS
        )

        target_pages = self._pages_for_sections(
            pages, section_map, _IMPL_SECTIONS
        ) or pages

        considerations: List[ImplementationConsideration] = []
        seen_cues: Set[str] = set()

        for page in target_pages:
            for sent in self._iter_sentences(page.text):
                for impl_type, cue_list in _IMPL_CUES.items():
                    for pat, cue_name in cue_list:
                        if pat.search(sent) and cue_name not in seen_cues:
                            ev = self._make_evidence(page.page_num, sent)
                            if ev is not None:
                                considerations.append(ImplementationConsideration(
                                    consideration_type=impl_type,
                                    evidence=[ev],
                                    cues_matched=[cue_name],
                                    page=page.page_num,
                                ))
                                seen_cues.add(cue_name)
                            break  # one type match per sentence

        if considerations:
            status = ComponentStatus.PRESENT
        elif is_labeled:
            status = ComponentStatus.WEAK
        else:
            status = ComponentStatus.ABSENT

        return considerations[:10], status, labeled_section, is_labeled

    # ------------------------------------------------------------------
    # Narrative / storytelling hooks
    # ------------------------------------------------------------------

    def _detect_narrative(
        self,
        pages: List[PageText],
        section_map: Optional[DocumentSectionMap],
    ) -> NarrativeHook:
        """Scan early pages for narrative hooks.

        Conservative: only checks the first 3 pages (or pages in
        introduction/executive_summary sections) to avoid confusing
        examples in later content or references for narrative openings.
        """
        early_labels = {
            SectionLabel.INTRODUCTION,
            SectionLabel.EXECUTIVE_SUMMARY,
            SectionLabel.KEY_MESSAGES,
        }
        target_pages = self._pages_for_sections(
            pages, section_map, early_labels
        )
        # Fallback: first 3 pages
        if not target_pages:
            target_pages = pages[:3]

        for page in target_pages:
            for sent in self._iter_sentences(page.text):
                for hook_type, cue_list in _NARRATIVE_CUES.items():
                    for pat, _cue_name in cue_list:
                        if pat.search(sent):
                            ev = self._make_evidence(page.page_num, sent)
                            if ev is not None:
                                return NarrativeHook(
                                    status=ComponentStatus.PRESENT,
                                    hook_type=hook_type,
                                    evidence=[ev],
                                    page=page.page_num,
                                )

        return NarrativeHook()  # absent by default

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_labeled_section(
        section_map: Optional[DocumentSectionMap],
        target_labels: Set[SectionLabel],
    ) -> Tuple[bool, Optional[SectionLabel]]:
        """Check if any section has a normalized label in target_labels.

        When multiple sections match, prefers the most specific label
        (i.e. not INTRODUCTION / EXECUTIVE_SUMMARY / KEY_MESSAGES).
        """
        if not section_map:
            return False, None
        generic = {SectionLabel.INTRODUCTION, SectionLabel.EXECUTIVE_SUMMARY,
                   SectionLabel.KEY_MESSAGES}
        best: Optional[SectionLabel] = None
        for sec in section_map.sections:
            if sec.normalized_label in target_labels:
                if best is None or (best in generic and sec.normalized_label not in generic):
                    best = sec.normalized_label
        if best is not None:
            return True, best
        return False, None

    @staticmethod
    def _pages_for_sections(
        pages: List[PageText],
        section_map: Optional[DocumentSectionMap],
        target_labels: Set[SectionLabel],
    ) -> List[PageText]:
        """Return pages that fall within sections matching target_labels."""
        if not section_map:
            return []

        page_nums: Set[int] = set()
        for sec in section_map.sections:
            if sec.normalized_label in target_labels:
                for p in range(sec.start_page, sec.end_page + 1):
                    page_nums.add(p)

        if not page_nums:
            return []

        return [p for p in pages if p.page_num in page_nums]

    @staticmethod
    def _iter_sentences(text: str):
        """Yield sentence-like spans from page text.

        Uses a simple split on sentence-ending punctuation followed by
        whitespace or end-of-string.  Good enough for discourse-marker
        matching without requiring NLP tokenisation.
        """
        # Split on . ! ? followed by whitespace or end
        parts = re.split(r"(?<=[.!?])\s+", text)
        for part in parts:
            part = part.strip()
            if len(part) >= _MIN_QUOTE:
                yield part

    @staticmethod
    def _make_evidence(page_num: int, sentence: str) -> Optional[Evidence]:
        """Create an Evidence object from a sentence, or None if invalid."""
        quote = sentence.strip()
        if len(quote) < _MIN_QUOTE:
            return None
        if len(quote) > _MAX_QUOTE:
            quote = quote[:_MAX_QUOTE - 3] + "..."
        try:
            return Evidence(page=page_num, quote=quote)
        except Exception:
            return None
