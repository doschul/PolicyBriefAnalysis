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
    """Structured policy recommendation extraction."""
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
    
    # Computed metrics
    metadata: PDFMetadata = Field(description="PDF metadata")
    metrics: DocumentMetrics = Field(description="Computed document metrics")
    
    # Analysis results
    frame_assessments: List[FrameAssessment] = Field(
        description="Theoretical frame assessments"
    )
    recommendations: List[PolicyRecommendation] = Field(
        description="Extracted policy recommendations"
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
    """Structured output from recommendation extraction LLM call."""
    recommendations: List[PolicyRecommendation]
    
    class Config:
        extra = "forbid"