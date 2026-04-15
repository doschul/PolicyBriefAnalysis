"""
Tests for Pydantic models and data validation.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.policybrief.models import (
    Evidence,
    DocumentMetrics,
    DocumentFrontMatter,
    FrameAssessment,
    PolicyRecommendation,
    PerDocumentExtraction,
    PageText,
    PDFMetadata,
    ProcessingStatus,
    FrameDecision,
    InstrumentType,
    GeographicScope,
    Timeframe,
    RecommendationStrength,
    ActorType
)


class TestEvidence:
    """Test Evidence model validation."""
    
    def test_valid_evidence(self):
        """Test valid evidence creation."""
        evidence = Evidence(
            page=1,
            quote="This is a valid quote with sufficient length."
        )
        
        assert evidence.page == 1
        assert evidence.quote == "This is a valid quote with sufficient length."
    
    def test_quote_too_short(self):
        """Test validation fails for quotes that are too short."""
        with pytest.raises(ValidationError):
            Evidence(
                page=1,
                quote="Short"  # Too short
            )
    
    def test_quote_too_long(self):
        """Test validation fails for quotes that are too long."""
        long_quote = "x" * 501  # Exceeds max_length=500
        
        with pytest.raises(ValidationError):
            Evidence(
                page=1,
                quote=long_quote
            )
    
    def test_empty_quote(self):
        """Test validation fails for empty quotes."""
        with pytest.raises(ValidationError):
            Evidence(
                page=1,
                quote=""
            )
    
    def test_whitespace_only_quote(self):
        """Test validation fails for whitespace-only quotes."""
        with pytest.raises(ValidationError):
            Evidence(
                page=1,
                quote="   \n\t   "
            )

    def test_quote_cleaning(self):
        """Test that quotes are properly cleaned."""
        evidence = Evidence(
            page=1,
            quote="  This quote has extra whitespace  "
        )
        
        assert evidence.quote == "This quote has extra whitespace"


class TestDocumentMetrics:
    """Test DocumentMetrics model validation."""
    
    def test_valid_metrics(self):
        """Test valid metrics creation."""
        metrics = DocumentMetrics(
            page_count=10,
            word_count=2500,
            char_count=15000,
            heading_count=5,
            paragraph_count=25,
            sentence_count=150,
            list_item_count=10,
            avg_sentence_length=16.7,
            lexical_diversity=0.65,
            avg_word_length=5.2,
            flesch_kincaid_grade=12.5,
            flesch_reading_ease=45.2,
            table_count=2,
            figure_count=3,
            reference_count=15,
            url_count=5,
            passive_voice_percent=12.5
        )
        
        assert metrics.page_count == 10
        assert metrics.word_count == 2500
        assert metrics.avg_sentence_length == 16.7
        assert metrics.lexical_diversity == 0.65
        assert metrics.flesch_kincaid_grade == 12.5
        assert metrics.passive_voice_percent == 12.5
    
    def test_optional_fields(self):
        """Test that optional fields can be None."""
        metrics = DocumentMetrics(
            page_count=1,
            word_count=100,
            char_count=600,
            heading_count=0,
            paragraph_count=5,
            sentence_count=10,
            list_item_count=0,
            avg_sentence_length=10.0,
            lexical_diversity=0.8,
            avg_word_length=4.5
            # Optional fields not provided
        )
        
        assert metrics.flesch_kincaid_grade is None
        assert metrics.flesch_reading_ease is None
        assert metrics.passive_voice_percent is None
        assert metrics.table_count == 0  # Has default
    
    def test_negative_values(self):
        """Test validation of negative values where inappropriate."""
        with pytest.raises(ValidationError):
            DocumentMetrics(
                page_count=-1,  # Should not be negative
                word_count=100,
                char_count=600,
                heading_count=0,
                paragraph_count=5, 
                sentence_count=10,
                list_item_count=0,
                avg_sentence_length=10.0,
                lexical_diversity=0.8,
                avg_word_length=4.5
            )


class TestDocumentFrontMatter:
    """Test DocumentFrontMatter model validation."""
    
    def test_valid_front_matter(self):
        """Test valid front matter creation."""
        front_matter = DocumentFrontMatter(
            title="Test Policy Brief",
            authors=["John Smith", "Jane Doe"],
            affiliations=["University of Policy"], 
            emails=["john@example.com"],
            urls=["https://example.org"],
            funding_statements=["Supported by Research Grant 123"],
            linked_studies=["Full report: Policy Analysis 2024"]
        )
        
        assert front_matter.title == "Test Policy Brief"
        assert len(front_matter.authors) == 2
        assert "John Smith" in front_matter.authors
        assert len(front_matter.emails) == 1
        assert front_matter.emails[0] == "john@example.com"
    
    def test_empty_front_matter(self):
        """Test front matter with empty/default fields."""
        front_matter = DocumentFrontMatter()
        
        assert front_matter.title is None
        assert front_matter.authors == []
        assert front_matter.affiliations == []
        assert front_matter.emails == []
        assert front_matter.urls == []
        assert front_matter.funding_statements == []
        assert front_matter.linked_studies == []
    
    def test_partial_front_matter(self):
        """Test front matter with only some fields populated."""
        front_matter = DocumentFrontMatter(
            title="Partial Brief",
            authors=["Author Only"]
        )
        
        assert front_matter.title == "Partial Brief"
        assert len(front_matter.authors) == 1
        assert front_matter.affiliations == []  # Should default to empty
        assert front_matter.emails == []
    
    def test_front_matter_serialization(self):
        """Test front matter serialization."""
        front_matter = DocumentFrontMatter(
            title="Serialization Test",
            authors=["Test Author"],
            emails=["test@example.com"]
        )
        
        data = front_matter.model_dump()
        assert isinstance(data, dict)
        assert data["title"] == "Serialization Test"
        assert data["authors"] == ["Test Author"]
        
        # Test deserialization
        new_front_matter = DocumentFrontMatter.model_validate(data)
        assert new_front_matter.title == front_matter.title
        assert new_front_matter.authors == front_matter.authors


class TestFrameAssessment:
    """Test FrameAssessment model validation."""
    
    def test_valid_assessment(self):
        """Test valid frame assessment."""
        evidence = Evidence(
            page=1,
            quote="This is evidence for the frame being present."
        )
        
        assessment = FrameAssessment(
            frame_id="test_frame",
            frame_label="Test Frame",
            decision=FrameDecision.PRESENT,
            confidence=0.85,
            evidence=[evidence],
            rationale="Clear evidence found in the document."
        )
        
        assert assessment.frame_id == "test_frame"
        assert assessment.decision == FrameDecision.PRESENT
        assert assessment.confidence == 0.85
        assert len(assessment.evidence) == 1
        assert assessment.rationale == "Clear evidence found in the document."
    
    def test_confidence_bounds(self):
        """Test confidence must be between 0 and 1."""
        evidence = Evidence(
            page=1,
            quote="Valid evidence quote here."
        )
        
        # Test confidence > 1
        with pytest.raises(ValidationError):
            FrameAssessment(
                frame_id="test_frame",
                frame_label="Test Frame", 
                decision=FrameDecision.PRESENT,
                confidence=1.5,  # Invalid
                evidence=[evidence],
                rationale="Test"
            )
        
        # Test confidence < 0
        with pytest.raises(ValidationError):
            FrameAssessment(
                frame_id="test_frame",
                frame_label="Test Frame",
                decision=FrameDecision.PRESENT,
                confidence=-0.1,  # Invalid
                evidence=[evidence],
                rationale="Test"
            )
    
    def test_present_requires_evidence(self):
        """Test that 'present' decision requires evidence."""
        with pytest.raises(ValidationError, match="Evidence required"):
            FrameAssessment(
                frame_id="test_frame",
                frame_label="Test Frame",
                decision=FrameDecision.PRESENT,
                confidence=0.85,
                evidence=[],  # No evidence provided
                rationale="Test"
            )
    
    def test_absent_allows_no_evidence(self):
        """Test that 'absent' decision allows no evidence."""
        assessment = FrameAssessment(
            frame_id="test_frame",
            frame_label="Test Frame",
            decision=FrameDecision.ABSENT,
            confidence=0.95,
            evidence=[],  # No evidence is fine for absent
            rationale="No evidence found."
        )
        
        assert assessment.decision == FrameDecision.ABSENT
        assert len(assessment.evidence) == 0


class TestPolicyRecommendation:
    """Test PolicyRecommendation model validation."""
    
    def test_valid_recommendation(self):
        """Test valid recommendation creation."""
        evidence = Evidence(
            page=2,
            quote="The government should implement carbon pricing mechanisms."
        )
        
        recommendation = PolicyRecommendation(
            rec_id="doc_001_rec_01",
            actor=ActorType.GOVERNMENT,
            action="implement carbon pricing mechanisms",
            target="greenhouse gas emissions",
            instrument_type=InstrumentType.TAX,
            policy_domain="climate_change",
            geographic_scope=GeographicScope.NATIONAL,
            timeframe=Timeframe.SHORT_TERM,
            strength=RecommendationStrength.SHOULD,
            evidence=[evidence]
        )
        
        assert recommendation.rec_id == "doc_001_rec_01"
        assert recommendation.actor == ActorType.GOVERNMENT
        assert recommendation.action == "implement carbon pricing mechanisms"
        assert recommendation.instrument_type == InstrumentType.TAX
        assert recommendation.geographic_scope == GeographicScope.NATIONAL
        assert len(recommendation.evidence) == 1
    
    def test_recommendation_requires_evidence(self):
        """Test that recommendations require at least one evidence quote."""
        with pytest.raises(ValidationError, match="At least one evidence quote"):
            PolicyRecommendation(
                rec_id="doc_001_rec_01",
                actor=ActorType.GOVERNMENT,
                action="implement policy",
                target="environmental issues",
                instrument_type=InstrumentType.REGULATION,
                policy_domain="environment",
                geographic_scope=GeographicScope.NATIONAL,
                timeframe=Timeframe.MEDIUM_TERM,
                strength=RecommendationStrength.SHOULD,
                evidence=[]  # No evidence provided
            )
    
    def test_enum_validation(self):
        """Test enum field validation."""
        evidence = Evidence(
            page=1,
            quote="Valid evidence quote here."
        )
        
        # Test invalid enum value
        with pytest.raises(ValidationError):
            PolicyRecommendation(
                rec_id="test_rec",
                actor="invalid_actor",  # Not a valid ActorType
                action="test action",
                target="test target",
                instrument_type=InstrumentType.REGULATION,
                policy_domain="test_domain",
                geographic_scope=GeographicScope.NATIONAL,
                timeframe=Timeframe.SHORT_TERM,
                strength=RecommendationStrength.SHOULD,
                evidence=[evidence]
            )


class TestPageText:
    """Test PageText model validation."""
    
    def test_valid_page_text(self):
        """Test valid page text creation."""
        page = PageText(
            page_num=1,
            text="This is the content of page 1 with some text.",
            char_count=45,
            word_count=10
        )
        
        assert page.page_num == 1
        assert page.text == "This is the content of page 1 with some text."
        assert page.char_count == 45
        assert page.word_count == 10


class TestProcessingStatus:
    """Test ProcessingStatus model validation."""
    
    def test_valid_processing_status(self):
        """Test valid processing status creation."""
        status = ProcessingStatus(
            doc_id="test_doc_001",
            file_path="/path/to/document.pdf",
            file_hash="abc123def456",
            file_size_bytes=1024000,
            processing_timestamp=datetime.now(),
            processing_duration_seconds=45.2,
            parser_used="pymupdf",
            likely_scanned=False,
            text_extraction_quality=0.95,
            pages_processed=10,
            frames_processed=5,
            recommendations_extracted=3,
            warnings=["Low text density on page 5"]
        )
        
        assert status.doc_id == "test_doc_001"
        assert status.parser_used == "pymupdf"
        assert status.text_extraction_quality == 0.95
        assert status.likely_scanned is False
        assert len(status.warnings) == 1
    
    def test_quality_bounds(self):
        """Test text extraction quality bounds."""
        # Test quality > 1
        with pytest.raises(ValidationError):
            ProcessingStatus(
                doc_id="test",
                file_path="/test.pdf",
                file_hash="hash",
                file_size_bytes=1000,
                processing_timestamp=datetime.now(),
                processing_duration_seconds=1.0,
                parser_used="pymupdf",
                likely_scanned=False,
                text_extraction_quality=1.5,  # Invalid
                pages_processed=1,
                frames_processed=0,
                recommendations_extracted=0
            )
        
        # Test quality < 0  
        with pytest.raises(ValidationError):
            ProcessingStatus(
                doc_id="test",
                file_path="/test.pdf", 
                file_hash="hash",
                file_size_bytes=1000,
                processing_timestamp=datetime.now(),
                processing_duration_seconds=1.0,
                parser_used="pymupdf",
                likely_scanned=False,
                text_extraction_quality=-0.1,  # Invalid
                pages_processed=1,
                frames_processed=0,
                recommendations_extracted=0
            )


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_evidence_serialization(self):
        """Test Evidence model JSON serialization."""
        evidence = Evidence(
            page=1,
            quote="This is a test quote for serialization testing purposes."
        )
        
        # Test serialization to dict
        data = evidence.model_dump()
        assert isinstance(data, dict)
        assert data["page"] == 1
        assert data["quote"] == "This is a test quote for serialization testing purposes."
        
        # Test deserialization from dict
        new_evidence = Evidence.model_validate(data)
        assert new_evidence.page == evidence.page
        assert new_evidence.quote == evidence.quote
    
    def test_frame_assessment_serialization(self):
        """Test FrameAssessment model serialization."""
        evidence = Evidence(
            page=1,
            quote="Test evidence quote here for serialization."
        )
        
        assessment = FrameAssessment(
            frame_id="test_frame",
            frame_label="Test Frame",
            decision=FrameDecision.PRESENT,
            confidence=0.75,
            evidence=[evidence],
            rationale="Test rationale"
        )
        
        # Serialize and deserialize
        data = assessment.model_dump()
        new_assessment = FrameAssessment.model_validate(data)
        
        assert new_assessment.frame_id == assessment.frame_id
        assert new_assessment.decision == assessment.decision
        assert new_assessment.confidence == assessment.confidence
        assert len(new_assessment.evidence) == 1
        assert new_assessment.evidence[0].quote == evidence.quote
    
    def test_per_document_extraction_with_front_matter(self):
        """Test PerDocumentExtraction with front matter field."""
        from datetime import datetime
        
        # Create test components
        page = PageText(page_num=1, text="Test content", char_count=12, word_count=2)
        
        metadata = PDFMetadata(title="PDF Title")
        
        front_matter = DocumentFrontMatter(
            title="Content Title",
            authors=["Test Author"],
            emails=["test@example.com"]
        )
        
        metrics = DocumentMetrics(
            page_count=1,
            word_count=2,
            char_count=12,
            heading_count=0,
            paragraph_count=1,
            sentence_count=1,
            list_item_count=0,
            avg_sentence_length=2.0,
            lexical_diversity=1.0,
            avg_word_length=6.0
        )
        
        processing_status = ProcessingStatus(
            doc_id="test_doc",
            file_path="test.pdf",
            file_hash="abc123",
            file_size_bytes=1000,
            processing_timestamp=datetime.now(),
            processing_duration_seconds=1.0,
            parser_used="pymupdf",
            likely_scanned=False,
            text_extraction_quality=1.0,
            pages_processed=1,
            frames_processed=0,
            recommendations_extracted=0
        )
        
        # Create PerDocumentExtraction with front matter
        extraction = PerDocumentExtraction(
            doc_id="test_doc",
            pages=[page],
            headings=[], 
            metadata=metadata,
            front_matter=front_matter,
            metrics=metrics,
            frame_assessments=[],
            recommendations=[],
            processing_status=processing_status
        )
        
        # Test field access
        assert extraction.front_matter is not None
        assert extraction.front_matter.title == "Content Title"
        assert "Test Author" in extraction.front_matter.authors
        
        # Test serialization includes front matter
        data = extraction.model_dump()
        assert "front_matter" in data
        assert data["front_matter"]["title"] == "Content Title"
        assert data["front_matter"]["authors"] == ["Test Author"]
        
        # Test deserialization
        new_extraction = PerDocumentExtraction.model_validate(data)
        assert new_extraction.front_matter.title == front_matter.title
        assert new_extraction.front_matter.authors == front_matter.authors


if __name__ == "__main__":
    pytest.main([__file__])