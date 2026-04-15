"""
Pydantic models for structured data in the policy brief analysis pipeline.

All models use strict JSON schema validation for OpenAI structured outputs.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


# Enums for controlled vocabularies
class InstrumentType(str, Enum):
    """Policy instrument types."""
    REGULATION = "regulation"
    SUBSIDY = "subsidy" 
    TAX = "tax"
    INFORMATION = "information"
    VOLUNTARY = "voluntary"
    PLANNING = "planning"
    MONITORING = "monitoring"
    RESEARCH = "research"
    PROCUREMENT = "procurement"
    INFRASTRUCTURE = "infrastructure"
    INSTITUTIONAL = "institutional"
    OTHER = "other"


class GeographicScope(str, Enum):
    """Geographic scope of policy recommendations."""
    LOCAL = "local"
    REGIONAL = "regional"
    NATIONAL = "national"
    INTERNATIONAL = "international"
    GLOBAL = "global"
    EU = "eu"
    BILATERAL = "bilateral"
    MULTILATERAL = "multilateral"
    SUBNATIONAL = "subnational"
    TRANSBOUNDARY = "transboundary"
    UNSPECIFIED = "unspecified"


class Timeframe(str, Enum):
    """Implementation timeframe."""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    ONGOING = "ongoing"
    UNSPECIFIED = "unspecified"


class RecommendationStrength(str, Enum):
    """Strength/urgency of recommendations."""
    MUST = "must"
    SHOULD = "should"
    COULD = "could"
    MAY = "may"
    CONSIDER = "consider"
    UNSPECIFIED = "unspecified"


class FrameDecision(str, Enum):
    """Decision values for frame presence."""
    PRESENT = "present"
    ABSENT = "absent"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ActorType(str, Enum):
    """Actor types for policy implementation."""
    GOVERNMENT = "government"
    EU_INSTITUTIONS = "eu_institutions"
    INTERNATIONAL_ORGANIZATIONS = "international_organizations"
    PRIVATE_SECTOR = "private_sector"
    CIVIL_SOCIETY = "civil_society"
    RESEARCH_INSTITUTIONS = "research_institutions"
    INDIVIDUALS = "individuals"
    MULTIPLE_ACTORS = "multiple_actors"
    UNSPECIFIED = "unspecified"


# Core data models
class PageText(BaseModel):
    """Text content from a single PDF page."""
    page_num: int = Field(description="1-based page number")
    text: str = Field(description="Extracted text content")
    char_count: int = Field(description="Character count for this page")
    word_count: int = Field(description="Word count for this page")
    
    class Config:
        extra = "forbid"


class Evidence(BaseModel):
    """Evidence quote with location information."""
    page: int
    quote: str = Field(min_length=10, max_length=500)
    
    @validator('quote')
    def quote_not_empty(cls, v):
        """Ensure quote is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Quote cannot be empty")
        return v.strip()
    
    class Config:
        extra = "forbid"


class DocumentMetrics(BaseModel):
    """Comprehensive document-level metrics."""
    # Basic structure
    page_count: int = Field(ge=0, description="Total number of pages")
    word_count: int = Field(ge=0, description="Total word count")
    char_count: int = Field(ge=0, description="Total character count")
    
    # Text structure signals
    heading_count: int = Field(ge=0, description="Number of detected headings")
    paragraph_count: int = Field(ge=0, description="Number of paragraphs")
    sentence_count: int = Field(ge=0, description="Number of sentences")
    list_item_count: int = Field(ge=0, description="Number of list items/bullets")
    
    # Linguistic metrics
    avg_sentence_length: float = Field(description="Average sentence length in words")
    lexical_diversity: float = Field(description="Type-token ratio (unique words/total words)")
    avg_word_length: float = Field(description="Average word length in characters")
    
    # Readability metrics (using textstat)
    flesch_kincaid_grade: Optional[float] = Field(
        default=None,
        description="Flesch-Kincaid grade level"
    )
    flesch_reading_ease: Optional[float] = Field(
        default=None, 
        description="Flesch reading ease score"
    )
    
    # Content density signals
    table_count: int = Field(default=0, description="Number of detected tables")
    figure_count: int = Field(default=0, description="Number of detected figures")
    reference_count: int = Field(default=0, description="Number of references/citations")
    url_count: int = Field(default=0, description="Number of URLs")
    
    # Language and style
    passive_voice_percent: Optional[float] = Field(
        default=None,
        description="Estimated percentage of passive voice constructions"
    )
    
    class Config:
        extra = "forbid"


class PDFMetadata(BaseModel):
    """PDF document metadata."""
    title: Optional[str] = Field(default=None, description="Document title")
    author: Optional[str] = Field(default=None, description="Document author")
    subject: Optional[str] = Field(default=None, description="Document subject")
    creator: Optional[str] = Field(default=None, description="Document creator software")
    producer: Optional[str] = Field(default=None, description="PDF producer")
    creation_date: Optional[datetime] = Field(default=None, description="Creation date")
    modification_date: Optional[datetime] = Field(default=None, description="Last modification date")
    
    class Config:
        extra = "forbid"


class FrameAssessment(BaseModel):
    """Assessment of theoretical frame presence in document."""
    frame_id: str = Field(description="Unique frame identifier")
    frame_label: str = Field(description="Human-readable frame name")
    
    # Core assessment
    decision: FrameDecision = Field(description="Present/absent/insufficient evidence")
    confidence: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Confidence in assessment (0-1)"
    )
    
    # Evidence and reasoning
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="Supporting evidence quotes"
    )
    counterevidence: List[Evidence] = Field(
        default_factory=list,
        description="Evidence that argues against frame presence"
    )
    rationale: str = Field(
        description="Brief explanation grounded in evidence quotes only"
    )
    
    @validator('evidence')
    def evidence_required_for_present(cls, v, values):
        """Ensure evidence is provided if frame is marked as present."""
        if 'decision' in values and values['decision'] == FrameDecision.PRESENT:
            if not v or len(v) == 0:
                raise ValueError("Evidence required when frame is marked as present")
        return v
    
    class Config:
        extra = "forbid"


class PolicyRecommendation(BaseModel):
    """Structured policy recommendation extraction.

    MIGRATION NOTE (Prompt 4): This model is retained for backward compatibility
    with existing tests and serialized audit JSON.  New extraction results use
    PolicyExtraction below.  Pipeline code now populates ``recommendations`` with
    PolicyExtraction instances (which is a superset of these fields).
    """
    rec_id: str
    
    # Core recommendation structure
    actor: ActorType
    action: str
    target: str
    
    # Classification
    instrument_type: InstrumentType
    policy_domain: str
    geographic_scope: GeographicScope
    timeframe: Timeframe
    strength: RecommendationStrength
    
    # Evidence
    evidence: List[Evidence]
    
    @validator('evidence')
    def evidence_required(cls, v):
        """Ensure at least one evidence quote is provided."""
        if not v or len(v) == 0:
            raise ValueError("At least one evidence quote is required")
        return v
    
    class Config:
        extra = "forbid"


class ProcessingStatus(BaseModel):
    """Document processing status and metadata."""
    doc_id: str = Field(description="Unique document identifier")
    file_path: str = Field(description="Original file path")
    file_hash: str = Field(description="Content hash for change detection")
    file_size_bytes: int = Field(description="File size in bytes")
    
    # Processing metadata
    processing_timestamp: datetime = Field(description="When processing completed")
    processing_duration_seconds: float = Field(description="Processing time")
    parser_used: str = Field(description="PDF parser used (pypdf/pymupdf)")
    
    # Quality indicators
    likely_scanned: bool = Field(description="Whether document appears to be scanned")
    text_extraction_quality: float = Field(
        ge=0.0, 
        le=1.0,
        description="Estimated text extraction quality (0-1)"
    )
    
    # Processing results
    pages_processed: int = Field(description="Number of pages successfully processed")
    frames_processed: int = Field(description="Number of frames assessed")
    recommendations_extracted: int = Field(description="Number of recommendations extracted")
    
    # Error tracking
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal processing warnings"
    )
    
    class Config:
        extra = "forbid"


class DocumentFrontMatter(BaseModel):
    """Content-derived document front matter."""
    title: Optional[str] = Field(default=None, description="Title extracted from content")
    authors: List[str] = Field(
        default_factory=list,
        description="Author names extracted from content"
    )
    affiliations: List[str] = Field(
        default_factory=list,
        description="Institutional affiliations"
    )
    emails: List[str] = Field(
        default_factory=list,
        description="Email addresses found in document"
    )
    urls: List[str] = Field(
        default_factory=list,
        description="URLs found in document"
    )
    funding_statements: List[str] = Field(
        default_factory=list,
        description="Funding acknowledgments and statements"
    )
    linked_studies: List[str] = Field(
        default_factory=list,
        description="References to companion studies or reports"
    )
    
    class Config:
        extra = "forbid"


# Section segmentation models
class SectionLabel(str, Enum):
    """Normalized section labels for policy briefs."""
    TITLE_PAGE = "title_page"
    EXECUTIVE_SUMMARY = "executive_summary"
    KEY_MESSAGES = "key_messages"
    INTRODUCTION = "introduction"
    PROBLEM_DEFINITION = "problem_definition"
    POLICY_OPTIONS = "policy_options"
    RECOMMENDATIONS = "recommendations"
    IMPLEMENTATION = "implementation"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    ACKNOWLEDGEMENTS = "acknowledgements"
    ABOUT_AUTHORS = "about_authors"
    CONTACT = "contact"
    APPENDIX = "appendix"


class DocumentSection(BaseModel):
    """A single detected section in the document."""
    raw_title: Optional[str] = Field(
        default=None,
        description="Original heading text as found in the document"
    )
    normalized_label: Optional[SectionLabel] = Field(
        default=None,
        description="Normalized section label, None if uncertain"
    )
    start_page: int = Field(description="1-based start page")
    end_page: int = Field(description="1-based end page (inclusive)")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in heading detection and label assignment"
    )
    rule_source: str = Field(
        default="text_heuristic",
        description="Which rule produced this section (layout, text_heuristic, fallback)"
    )

    class Config:
        extra = "forbid"


class DocumentSectionMap(BaseModel):
    """Structural map of all detected sections in a document."""
    sections: List[DocumentSection] = Field(
        default_factory=list,
        description="Ordered list of detected sections"
    )
    detection_method: str = Field(
        default="text_heuristic",
        description="Primary detection method used (layout, text_heuristic, fallback)"
    )

    class Config:
        extra = "forbid"


# Structural core extraction models
class ComponentStatus(str, Enum):
    """Detection status for a structural component."""
    PRESENT = "present"
    ABSENT = "absent"
    WEAK = "weak"


class ProblemIdentification(BaseModel):
    """Whether the document explicitly defines a policy problem."""
    status: ComponentStatus = Field(
        default=ComponentStatus.ABSENT,
        description="present if explicitly framed, weak if only implicit, absent otherwise"
    )
    matched_section: Optional[SectionLabel] = Field(
        default=None,
        description="Section where problem framing was found"
    )
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="Supporting evidence spans"
    )
    cues_matched: List[str] = Field(
        default_factory=list,
        description="Discourse markers / keywords that triggered detection"
    )
    is_explicitly_labeled: bool = Field(
        default=False,
        description="True if a heading explicitly labels the problem section"
    )

    class Config:
        extra = "forbid"


class SolutionOptionType(str, Enum):
    """Distinguishes generic discussion from explicit policy options."""
    GENERIC_DISCUSSION = "generic_discussion"
    EXPLICIT_OPTION = "explicit_option"


class SolutionOption(BaseModel):
    """A detected solution or policy option."""
    status: ComponentStatus = Field(
        default=ComponentStatus.ABSENT,
        description="present if explicit, weak if only implied"
    )
    option_type: SolutionOptionType = Field(
        default=SolutionOptionType.GENERIC_DISCUSSION,
        description="Whether this is a generic discussion or a named policy option"
    )
    matched_section: Optional[SectionLabel] = Field(
        default=None,
        description="Section where the option was found"
    )
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="Supporting evidence spans"
    )
    cues_matched: List[str] = Field(
        default_factory=list,
        description="Discourse markers that triggered detection"
    )
    is_explicitly_labeled: bool = Field(
        default=False,
        description="True if a heading explicitly labels the solutions/options section"
    )

    class Config:
        extra = "forbid"


class ImplementationType(str, Enum):
    """Type of implementation consideration."""
    BARRIER = "barrier"
    FACILITATOR = "facilitator"
    FEASIBILITY = "feasibility"
    SEQUENCING = "sequencing"
    RESOURCE = "resource"
    INSTITUTIONAL = "institutional"
    RISK = "risk"
    GENERAL = "general"


class ImplementationConsideration(BaseModel):
    """A detected implementation consideration."""
    consideration_type: ImplementationType = Field(
        default=ImplementationType.GENERAL,
        description="Category of implementation consideration"
    )
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="Supporting evidence spans"
    )
    cues_matched: List[str] = Field(
        default_factory=list,
        description="Discourse markers that triggered detection"
    )
    page: int = Field(description="Page where the consideration was found")

    class Config:
        extra = "forbid"


class NarrativeHookType(str, Enum):
    """Type of narrative / storytelling device."""
    CASE_VIGNETTE = "case_vignette"
    ANECDOTE = "anecdote"
    VIVID_EXAMPLE = "vivid_example"
    NARRATIVE_OPENING = "narrative_opening"


class NarrativeHook(BaseModel):
    """A detected narrative or storytelling device."""
    status: ComponentStatus = Field(
        default=ComponentStatus.ABSENT,
        description="present if clearly narrative, weak if borderline"
    )
    hook_type: Optional[NarrativeHookType] = Field(
        default=None,
        description="Type of narrative device, None if absent"
    )
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="Supporting evidence spans"
    )
    page: Optional[int] = Field(
        default=None,
        description="Page where the hook was found"
    )

    class Config:
        extra = "forbid"


class LabelingAssessment(BaseModel):
    """Whether core components are explicitly labeled by headings or only implicit."""
    problem_labeled: bool = Field(
        default=False,
        description="True if a heading explicitly labels the problem section"
    )
    solutions_labeled: bool = Field(
        default=False,
        description="True if a heading explicitly labels the solutions/options section"
    )
    implementation_labeled: bool = Field(
        default=False,
        description="True if a heading explicitly labels the implementation section"
    )

    class Config:
        extra = "forbid"


class StructuralCoreResult(BaseModel):
    """Complete structural core analysis for a document."""
    problem: ProblemIdentification = Field(
        default_factory=ProblemIdentification,
        description="Problem identification analysis"
    )
    solutions: List[SolutionOption] = Field(
        default_factory=list,
        description="Detected solutions / policy options"
    )
    implementation: List[ImplementationConsideration] = Field(
        default_factory=list,
        description="Detected implementation considerations"
    )
    implementation_status: ComponentStatus = Field(
        default=ComponentStatus.ABSENT,
        description="Overall status of implementation content"
    )
    implementation_matched_section: Optional[SectionLabel] = Field(
        default=None,
        description="Section where implementation content was found"
    )
    implementation_is_explicitly_labeled: bool = Field(
        default=False,
        description="True if a heading explicitly labels implementation"
    )
    narrative_hook: NarrativeHook = Field(
        default_factory=NarrativeHook,
        description="Narrative / storytelling hook analysis"
    )
    labeling: LabelingAssessment = Field(
        default_factory=LabelingAssessment,
        description="Whether core components are explicitly labeled"
    )

    class Config:
        extra = "forbid"


# --- Rewritten recommendation / policy-extraction models (Prompt 4) -----------

class ExtractionType(str, Enum):
    """Distinguishes the function a candidate span serves."""
    RECOMMENDATION = "recommendation"
    POLICY_OPTION = "policy_option"
    IMPLEMENTATION_STEP = "implementation_step"
    EXPECTED_OUTCOME = "expected_outcome"
    TRADE_OFF = "trade_off"
    ACTOR_RESPONSIBILITY = "actor_responsibility"
    NON_RECOMMENDATION = "non_recommendation"


class CandidateSpan(BaseModel):
    """A sentence-level candidate extracted from a target section.

    Used internally by the recommendation extractor; not surfaced to
    final output unless it survives classification.
    """
    text: str = Field(description="Raw sentence / span text")
    page: int = Field(description="1-based page number")
    source_section: Optional[SectionLabel] = Field(
        default=None,
        description="Section label the span was found in"
    )
    has_prescriptive_language: bool = Field(
        default=False,
        description="Whether span contains prescriptive verbs/modal constructs"
    )
    prescriptive_cues: List[str] = Field(
        default_factory=list,
        description="Which prescriptive cues were matched"
    )

    class Config:
        extra = "forbid"


class PolicyExtraction(BaseModel):
    """A classified policy extraction — recommendation, option, step, etc.

    Replaces the old PolicyRecommendation with richer, section-aware fields
    and explicit null-first defaults.
    """
    rec_id: str = Field(default="", description="Stable identifier, set by pipeline")

    # --- classification ---
    extraction_type: ExtractionType = Field(
        description="Functional role of this extraction"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Classifier confidence (0-1)"
    )

    # --- raw source ---
    source_text_raw: str = Field(
        description="Verbatim source text of the candidate span"
    )
    source_section: Optional[SectionLabel] = Field(
        default=None,
        description="Section label where the span was found"
    )
    page: int = Field(description="1-based page number")

    # --- actor ---
    actor_text_raw: Optional[str] = Field(
        default=None,
        description="Raw actor text as it appears in source (null if absent)"
    )
    actor_type_normalized: Optional[ActorType] = Field(
        default=None,
        description="Normalized actor type, only if safely mappable"
    )

    # --- action / target ---
    action_text_raw: Optional[str] = Field(
        default=None,
        description="Raw action/verb phrase from source"
    )
    target_text_raw: Optional[str] = Field(
        default=None,
        description="Raw target/object of the action"
    )

    # --- classification fields (null if absent) ---
    instrument_type: Optional[InstrumentType] = Field(
        default=None,
        description="Policy instrument, only if explicit in text"
    )
    policy_domain: Optional[str] = Field(
        default=None,
        description="Policy domain if identifiable"
    )
    geographic_scope: Optional[GeographicScope] = Field(
        default=None,
        description="Geographic scope if explicit"
    )
    timeframe: Optional[Timeframe] = Field(
        default=None,
        description="Timeframe if explicit"
    )
    strength: Optional[RecommendationStrength] = Field(
        default=None,
        description="Recommendation strength/modality"
    )

    # --- structured sub-components (null / empty if absent) ---
    expected_outcomes: List[str] = Field(
        default_factory=list,
        description="Anticipated outcomes/impacts mentioned in context"
    )
    implementation_steps: List[str] = Field(
        default_factory=list,
        description="Implementation steps mentioned in context"
    )
    trade_offs: List[str] = Field(
        default_factory=list,
        description="Trade-offs, downsides, or risks mentioned"
    )

    # --- evidence ---
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="Verbatim evidence quotes"
    )

    @validator('evidence')
    def evidence_required_for_recommendation(cls, v, values):
        """Recommendations and options require at least one evidence quote."""
        etype = values.get('extraction_type')
        if etype in (ExtractionType.RECOMMENDATION, ExtractionType.POLICY_OPTION):
            if not v or len(v) == 0:
                raise ValueError(
                    "At least one evidence quote is required for recommendations and options"
                )
        return v

    class Config:
        extra = "forbid"


class PerDocumentExtraction(BaseModel):
    """Complete extraction results for a single document."""
    # Document identification
    doc_id: str = Field(description="Unique document identifier")
    
    # Source content
    pages: List[PageText] = Field(description="Extracted text by page")
    headings: List[str] = Field(
        default_factory=list,
        description="Detected document headings"
    )
    
    # Section structure
    section_map: Optional[DocumentSectionMap] = Field(
        default=None,
        description="Structural section map of the document"
    )
    
    # Structural core analysis
    structural_core: Optional[StructuralCoreResult] = Field(
        default=None,
        description="Structural core component analysis"
    )
    
    # Computed metrics
    metadata: PDFMetadata = Field(description="PDF metadata")
    front_matter: Optional[DocumentFrontMatter] = Field(
        default=None,
        description="Content-derived front matter"
    )
    metrics: DocumentMetrics = Field(description="Computed document metrics")
    
    # Analysis results
    frame_assessments: List[FrameAssessment] = Field(
        description="Theoretical frame assessments"
    )
    # Whether the document explicitly discusses policy mixes / instrument
    # complementarity (requires ≥2 frames present AND explicit mix language).
    policy_mix_present: bool = Field(
        default=False,
        description="Document explicitly discusses combinations or complementarity of policy instruments"
    )
    # MIGRATION (Prompt 4): ``recommendations`` is kept for backward-compat
    # serialisation and tests that construct PolicyRecommendation objects.
    # New pipeline code populates ``policy_extractions`` instead.
    recommendations: List[PolicyRecommendation] = Field(
        default_factory=list,
        description="Legacy policy recommendations (kept for backward compatibility)"
    )
    policy_extractions: List[PolicyExtraction] = Field(
        default_factory=list,
        description="Section-aware policy extractions (Prompt 4 replacement)"
    )
    
    # Processing metadata
    processing_status: ProcessingStatus = Field(description="Processing metadata")
    
    class Config:
        extra = "forbid"


# LLM interaction models
class FrameDetectionInput(BaseModel):
    """Input for frame detection LLM call."""
    frame_id: str = Field(description="Frame to assess")
    frame_definition: str = Field(description="Frame definition for context")
    text_spans: List[Dict[str, Any]] = Field(
        description="Relevant text spans with page numbers and context"
    )
    
    class Config:
        extra = "forbid"


class FrameDetectionOutput(BaseModel):
    """Structured output from frame detection LLM call."""
    frame_id: str
    decision: FrameDecision
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[Evidence]
    rationale: str
    
    class Config:
        extra = "forbid"


class RecommendationExtractionOutput(BaseModel):
    """Structured output from recommendation extraction LLM call.

    MIGRATION (Prompt 4): Now wraps CandidateClassification objects
    returned by the narrow candidate-span classifier.
    """
    recommendations: List[PolicyRecommendation]
    
    class Config:
        extra = "forbid"


class CandidateClassification(BaseModel):
    """LLM classification result for a single candidate span."""
    extraction_type: ExtractionType = Field(
        description="Functional role of this span"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in classification (0-1)"
    )
    actor_text_raw: Optional[str] = Field(
        default=None,
        description="Raw actor text from span, null if none"
    )
    action_text_raw: Optional[str] = Field(
        default=None,
        description="Raw action phrase, null if none"
    )
    target_text_raw: Optional[str] = Field(
        default=None,
        description="Raw target/object, null if none"
    )
    instrument_type: Optional[str] = Field(
        default=None,
        description="Policy instrument if explicit, null otherwise"
    )
    strength: Optional[str] = Field(
        default=None,
        description="Recommendation strength modal verb, null if none"
    )
    expected_outcomes: List[str] = Field(
        default_factory=list,
        description="Anticipated outcomes mentioned in span context"
    )
    implementation_steps: List[str] = Field(
        default_factory=list,
        description="Implementation steps mentioned"
    )
    trade_offs: List[str] = Field(
        default_factory=list,
        description="Trade-offs/risks/downsides mentioned"
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Why span was classified non_recommendation, null if accepted"
    )

    class Config:
        extra = "forbid"


class CandidateClassificationBatch(BaseModel):
    """LLM output for a batch of candidate span classifications."""
    classifications: List[CandidateClassification] = Field(
        description="One classification per input candidate, same order"
    )

    class Config:
        extra = "forbid"