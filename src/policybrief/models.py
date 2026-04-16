"""
Pydantic models for the policy brief analysis pipeline.
All models use strict JSON schema validation for OpenAI structured outputs.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────

class InstrumentType(str, Enum):
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
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"
    ONGOING = "ongoing"
    UNSPECIFIED = "unspecified"


class RecommendationStrength(str, Enum):
    MUST = "must"
    SHOULD = "should"
    COULD = "could"
    MAY = "may"
    CONSIDER = "consider"
    UNSPECIFIED = "unspecified"


class FrameDecision(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ActorType(str, Enum):
    GOVERNMENT = "government"
    EU_INSTITUTIONS = "eu_institutions"
    INTERNATIONAL_ORGANIZATIONS = "international_organizations"
    PRIVATE_SECTOR = "private_sector"
    CIVIL_SOCIETY = "civil_society"
    RESEARCH_INSTITUTIONS = "research_institutions"
    INDIVIDUALS = "individuals"
    MULTIPLE_ACTORS = "multiple_actors"
    UNSPECIFIED = "unspecified"


class ExtractionType(str, Enum):
    RECOMMENDATION = "recommendation"
    POLICY_OPTION = "policy_option"
    IMPLEMENTATION_STEP = "implementation_step"
    EXPECTED_OUTCOME = "expected_outcome"
    TRADE_OFF = "trade_off"
    ACTOR_RESPONSIBILITY = "actor_responsibility"
    NON_RECOMMENDATION = "non_recommendation"


# ── Core data models ──────────────────────────────────────────────────────

class PageText(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_num: int = Field(description="1-based page number")
    text: str = Field(description="Extracted text content")
    char_count: int = Field(description="Character count")
    word_count: int = Field(description="Word count")


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int
    quote: str = Field(min_length=10, max_length=500)

    @model_validator(mode="after")
    def quote_not_empty(self):
        if not self.quote or not self.quote.strip():
            raise ValueError("Quote cannot be empty")
        self.quote = self.quote.strip()
        return self


class PDFMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None


class DocumentMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    char_count: int = Field(ge=0)
    sentence_count: int = Field(ge=0)
    paragraph_count: int = Field(ge=0)
    avg_sentence_length: float = 0.0
    lexical_diversity: float = 0.0
    avg_word_length: float = 0.0
    flesch_kincaid_grade: Optional[float] = None
    flesch_reading_ease: Optional[float] = None
    url_count: int = 0
    email_count: int = 0
    passive_voice_share: Optional[float] = Field(
        default=None,
        description="Heuristic share of sentences with passive voice constructions (0.0-1.0)",
    )


class DocumentFrontMatter(BaseModel):
    """Content-derived front matter.  Also used as the LLM output schema."""
    model_config = ConfigDict(extra="forbid")
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    affiliations: List[str] = Field(default_factory=list)
    emails: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)
    funding_statements: List[str] = Field(default_factory=list)
    linked_studies: List[str] = Field(default_factory=list)


# ── Frame detection models ────────────────────────────────────────────────

class FrameAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_id: str
    frame_label: str
    decision: FrameDecision
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[Evidence] = Field(default_factory=list)
    counterevidence: List[Evidence] = Field(default_factory=list)
    rationale: str

    @model_validator(mode="after")
    def evidence_required_for_present(self):
        if self.decision == FrameDecision.PRESENT and not self.evidence:
            raise ValueError("Evidence required when frame is present")
        return self


class SingleFrameResult(BaseModel):
    """LLM output for one frame within a broad-content frame extraction."""
    model_config = ConfigDict(extra="forbid")
    frame_id: str
    decision: FrameDecision
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: List[Evidence] = Field(default_factory=list)
    rationale: str


class FrameExtractionResponse(BaseModel):
    """LLM response containing assessments for all frames in one call."""
    model_config = ConfigDict(extra="forbid")
    frames: List[SingleFrameResult]


# ── Structural core (LLM-extracted) ──────────────────────────────────────

class StructuralCoreResult(BaseModel):
    """Simplified structural core analysis produced by a single LLM call."""
    model_config = ConfigDict(extra="forbid")
    problem_status: str = Field(default="absent", description="present/absent/weak")
    problem_summary: Optional[str] = None
    solutions_count: int = 0
    solutions_explicit: bool = False
    implementation_status: str = Field(default="absent", description="present/absent/weak")
    implementation_count: int = 0
    narrative_hook_present: bool = False
    narrative_hook_type: Optional[str] = None
    problem_explicitly_labelled: bool = Field(
        default=False,
        description="Whether the problem section is clearly labelled with a heading",
    )
    solutions_explicitly_labelled: bool = Field(
        default=False,
        description="Whether solutions/recommendations are clearly labelled with a heading",
    )
    implementation_explicitly_labelled: bool = Field(
        default=False,
        description="Whether implementation considerations are clearly labelled with a heading",
    )
    procedural_clarity_status: str = Field(
        default="absent",
        description="present/weak/absent — whether the document provides concrete procedural guidance on how actions should be carried out",
    )


# ── Recommendation extraction models ─────────────────────────────────────

class RecommendationItem(BaseModel):
    """Single recommendation/extraction returned by the LLM from broad content."""
    model_config = ConfigDict(extra="forbid")
    extraction_type: ExtractionType
    confidence: float = Field(ge=0.0, le=1.0)
    source_quote: str = Field(description="Verbatim quote from the document text")
    page: int = Field(description="Page number where the quote appears")
    actor_text_raw: Optional[str] = None
    action_text_raw: Optional[str] = None
    target_text_raw: Optional[str] = None
    instrument_type: Optional[str] = None
    strength: Optional[str] = None
    geographic_scope: Optional[str] = None
    timeframe: Optional[str] = None
    policy_domain: Optional[str] = None
    expected_outcomes: List[str] = Field(default_factory=list)
    implementation_steps: List[str] = Field(default_factory=list)
    trade_offs: List[str] = Field(default_factory=list)


class RecommendationExtractionResponse(BaseModel):
    """LLM response for broad-content recommendation extraction."""
    model_config = ConfigDict(extra="forbid")
    items: List[RecommendationItem] = Field(default_factory=list)


class PolicyExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rec_id: str = Field(default="")
    extraction_type: ExtractionType
    confidence: float = Field(ge=0.0, le=1.0)
    source_text_raw: str
    source_section: Optional[str] = None
    page: int
    actor_text_raw: Optional[str] = None
    actor_type_normalized: Optional[ActorType] = None
    action_text_raw: Optional[str] = None
    target_text_raw: Optional[str] = None
    instrument_type: Optional[InstrumentType] = None
    policy_domain: Optional[str] = None
    geographic_scope: Optional[GeographicScope] = None
    timeframe: Optional[Timeframe] = None
    strength: Optional[RecommendationStrength] = None
    expected_outcomes: List[str] = Field(default_factory=list)
    implementation_steps: List[str] = Field(default_factory=list)
    trade_offs: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def evidence_required_for_recommendation(self):
        if self.extraction_type in (
            ExtractionType.RECOMMENDATION,
            ExtractionType.POLICY_OPTION,
        ) and not self.evidence:
            raise ValueError(
                "At least one evidence quote is required for recommendations and options"
            )
        return self


# ── Processing and output models ─────────────────────────────────────────

class ProcessingStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    file_path: str
    file_hash: str
    file_size_bytes: int
    processing_timestamp: datetime
    processing_duration_seconds: float
    parser_used: str
    likely_scanned: bool
    text_extraction_quality: float = Field(ge=0.0, le=1.0)
    pages_processed: int
    frames_processed: int
    recommendations_extracted: int
    warnings: List[str] = Field(default_factory=list)


class PerDocumentExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    pages: List[PageText]
    metadata: PDFMetadata
    front_matter: Optional[DocumentFrontMatter] = None
    metrics: DocumentMetrics
    structural_core: Optional[StructuralCoreResult] = None
    frame_assessments: List[FrameAssessment] = Field(default_factory=list)
    policy_mix_present: bool = False
    policy_extractions: List[PolicyExtraction] = Field(default_factory=list)
    processing_status: ProcessingStatus

