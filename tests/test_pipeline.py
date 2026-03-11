"""
Tests for the main pipeline functionality.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.policybrief.pipeline import PolicyBriefPipeline
from src.policybrief.models import (
    PageText,
    PDFMetadata,
    DocumentMetrics,
    FrameAssessment,
    PolicyRecommendation,
    ProcessingStatus,
    PerDocumentExtraction,
    Evidence,
    FrameDecision,
    InstrumentType,
    GeographicScope,
    Timeframe,
    RecommendationStrength,
    ActorType
)


@pytest.fixture
def temp_config_dir():
    """Create temporary config directory with test configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir()
        
        # Create main config
        main_config = {
            "openai": {
                "model": "gpt-4o-2024-08-06",
                "temperature": 0.1,
                "max_tokens": 4000
            },
            "pdf": {
                "extract_method": "pymupdf",
                "preserve_layout": True,
                "max_pages": 0
            },
            "frames": {
                "min_confidence": 0.7,
                "max_spans_per_frame": 3
            },
            "recommendations": {
                "min_confidence": 0.6,
                "max_recommendations": 5
            },
            "output": {
                "formats": ["csv"],
                "generate_audit": True
            }
        }
        
        with open(config_dir / "config.yaml", "w") as f:
            import yaml
            yaml.dump(main_config, f)
        
        # Create frames config
        frames_config = {
            "frames": [
                {
                    "id": "test_frame_1",
                    "label": "Test Frame 1", 
                    "short_definition": "A test theoretical framework",
                    "inclusion_cues": ["test", "framework", "theory"],
                    "exclusion_cues": ["anti-test"],
                    "must_have": [["test", "theory"]]
                },
                {
                    "id": "test_frame_2", 
                    "label": "Test Frame 2",
                    "short_definition": "Another test framework",
                    "inclusion_cues": ["policy", "mechanism", "approach"],
                    "exclusion_cues": [],
                    "must_have": []
                }
            ]
        }
        
        with open(config_dir / "frames.yaml", "w") as f:
            import yaml
            yaml.dump(frames_config, f)
        
        # Create enums config
        enums_config = {
            "instrument_types": ["regulation", "subsidy", "tax", "other"],
            "geographic_scopes": ["local", "national", "international", "unspecified"], 
            "timeframes": ["immediate", "short_term", "medium_term", "long_term", "unspecified"],
            "strengths": ["must", "should", "could", "may", "unspecified"],
            "actor_types": ["government", "private_sector", "civil_society", "unspecified"],
            "policy_domains": ["climate_change", "environment", "other"]
        }
        
        with open(config_dir / "enums.yaml", "w") as f:
            import yaml
            yaml.dump(enums_config, f)
        
        yield config_dir


@pytest.fixture  
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_pdf_file():
    """Create a mock PDF file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
        # Write some minimal PDF content
        temp_file.write(b"%PDF-1.4\n%Mock PDF for testing\n%%EOF")
        temp_path = Path(temp_file.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_extraction_result():
    """Create sample extraction result for testing."""
    evidence = Evidence(
        page=1,
        quote="This is a test evidence quote from the document.",
        start_char=100,
        end_char=150
    )
    
    frame_assessment = FrameAssessment(
        frame_id="test_frame_1",
        frame_label="Test Frame 1",
        decision=FrameDecision.PRESENT,
        confidence=0.85,
        evidence=[evidence],
        rationale="Strong evidence found for this frame."
    )
    
    recommendation = PolicyRecommendation(
        rec_id="test_doc_rec_01",
        actor=ActorType.GOVERNMENT,
        action="implement test policies",
        target="environmental protection",
        instrument_type=InstrumentType.REGULATION,
        policy_domain="environment", 
        geographic_scope=GeographicScope.NATIONAL,
        timeframe=Timeframe.SHORT_TERM,
        strength=RecommendationStrength.SHOULD,
        evidence=[evidence]
    )
    
    extraction = PerDocumentExtraction(
        doc_id="test_doc",
        pages=[PageText(
            page_num=1,
            text="This is test document content with policy recommendations.",
            char_count=65,
            word_count=10
        )],
        headings=["Introduction", "Policy Recommendations"],
        metadata=PDFMetadata(title="Test Document"),
        metrics=DocumentMetrics(
            page_count=1,
            word_count=100,
            char_count=500,
            heading_count=2,
            paragraph_count=5,
            sentence_count=8,
            list_item_count=0,
            avg_sentence_length=12.5,
            lexical_diversity=0.7,
            avg_word_length=4.2
        ),
        frame_assessments=[frame_assessment],
        recommendations=[recommendation],
        processing_status=ProcessingStatus(
            doc_id="test_doc",
            file_path="/test.pdf",
            file_hash="test_hash",
            file_size_bytes=1024,
            processing_timestamp=datetime.now(),
            processing_duration_seconds=5.0,
            parser_used="pymupdf",
            likely_scanned=False,
            text_extraction_quality=0.95,
            pages_processed=1,
            frames_processed=1, 
            recommendations_extracted=1
        )
    )
    
    return extraction


class TestPipelineInitialization:
    """Test pipeline initialization and configuration loading."""
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_pipeline_init_success(self, temp_config_dir, temp_output_dir):
        """Test successful pipeline initialization."""
        pipeline = PolicyBriefPipeline(
            config_dir=temp_config_dir,
            output_dir=temp_output_dir,
            max_workers=2,
            force_reprocess=False
        )
        
        assert pipeline.config_dir == temp_config_dir
        assert pipeline.output_dir == temp_output_dir
        assert pipeline.max_workers == 2
        assert pipeline.force_reprocess is False
        assert len(pipeline.frames) == 2  # Two test frames
        assert pipeline.llm_client is not None
        assert pipeline.pdf_extractor is not None
    
    def test_missing_api_key(self, temp_config_dir, temp_output_dir):
        """Test pipeline initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Required environment variable"):
                PolicyBriefPipeline(
                    config_dir=temp_config_dir,
                    output_dir=temp_output_dir
                )
    
    def test_missing_config_files(self, temp_output_dir):
        """Test pipeline initialization fails with missing config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_config_dir = Path(temp_dir)
            
            with pytest.raises(FileNotFoundError):
                PolicyBriefPipeline(
                    config_dir=empty_config_dir,
                    output_dir=temp_output_dir
                )
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_frames_loading(self, temp_config_dir, temp_output_dir):
        """Test frames configuration is loaded correctly."""
        pipeline = PolicyBriefPipeline(
            config_dir=temp_config_dir,
            output_dir=temp_output_dir
        )
        
        assert len(pipeline.frames) == 2
        
        # Check that frames are loaded as list of dicts with proper IDs
        frame_ids = [frame["id"] for frame in pipeline.frames]
        assert "test_frame_1" in frame_ids
        assert "test_frame_2" in frame_ids
        
        # Find frame 1 and check its properties
        frame1 = next(frame for frame in pipeline.frames if frame["id"] == "test_frame_1")
        assert frame1["label"] == "Test Frame 1"
        assert "test" in frame1["inclusion_cues"]
        assert "anti-test" in frame1["exclusion_cues"]


class TestDocumentProcessing:
    """Test document processing functionality."""
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.policybrief.pdf_extractor.PDFExtractor.extract_document")
    @patch("src.policybrief.frame_detector.FrameDetector.detect_frames")
    @patch("src.policybrief.recommendation_extractor.RecommendationExtractor.extract_recommendations")
    def test_process_single_document_success(
        self, 
        mock_extract_recs,
        mock_detect_frames, 
        mock_extract_pdf,
        temp_config_dir,
        temp_output_dir,
        mock_pdf_file
    ):
        """Test successful processing of a single document."""
        # Setup mocks
        mock_pages = [PageText(
            page_num=1,
            text="Test document content with policy frameworks.",
            char_count=45,
            word_count=7
        )]
        
        mock_metadata = PDFMetadata(title="Test Document")
        mock_extraction_info = {
            "likely_scanned": False,
            "text_extraction_quality": 0.95,
            "warnings": []
        }
        
        mock_extract_pdf.return_value = (mock_pages, mock_metadata, mock_extraction_info)
        
        mock_frames = [FrameAssessment(
            frame_id="test_frame_1",
            frame_label="Test Frame 1",
            decision=FrameDecision.PRESENT,
            confidence=0.8,
            evidence=[Evidence(
                page=1,
                quote="Test evidence quote from document content.",
                start_char=10,
                end_char=50
            )],
            rationale="Evidence found."
        )]
        mock_detect_frames.return_value = mock_frames
        
        mock_recommendations = [PolicyRecommendation(
            rec_id="test_rec_1",
            actor=ActorType.GOVERNMENT,
            action="implement test policy",
            target="test outcomes",
            instrument_type=InstrumentType.REGULATION,
            policy_domain="environment",
            geographic_scope=GeographicScope.NATIONAL,
            timeframe=Timeframe.SHORT_TERM,
            strength=RecommendationStrength.SHOULD,
            evidence=[Evidence(
                page=1,
                quote="Government should implement test policy for outcomes.",
                start_char=20,
                end_char=70
            )]
        )]
        mock_extract_recs.return_value = mock_recommendations
        
        # Initialize pipeline
        pipeline = PolicyBriefPipeline(
            config_dir=temp_config_dir,
            output_dir=temp_output_dir,
            max_workers=1
        )
        
        # Process document
        result = pipeline._process_single_document(mock_pdf_file)
        
        # Verify results
        assert result is not None
        assert result.doc_id is not None
        assert len(result.pages) == 1
        assert len(result.frame_assessments) == 1
        assert len(result.recommendations) == 1
        assert result.processing_status.parser_used == "pymupdf"
        
        # Verify mocks were called
        mock_extract_pdf.assert_called_once_with(mock_pdf_file)
        mock_detect_frames.assert_called_once()
        mock_extract_recs.assert_called_once()
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_needs_processing_logic(self, temp_config_dir, temp_output_dir, mock_pdf_file):
        """Test file change detection logic."""
        pipeline = PolicyBriefPipeline(
            config_dir=temp_config_dir,
            output_dir=temp_output_dir,
            force_reprocess=False
        )
        
        # First time should need processing
        assert pipeline._needs_processing(mock_pdf_file) is True
        
        # Add file to cache
        file_hash = pipeline.pdf_extractor.compute_file_hash(mock_pdf_file)
        pipeline.processing_cache[str(mock_pdf_file)] = file_hash
        
        # Now should not need processing
        assert pipeline._needs_processing(mock_pdf_file) is False
        
        # With force_reprocess should always need processing
        pipeline.force_reprocess = True
        assert pipeline._needs_processing(mock_pdf_file) is True


class TestOutputGeneration:
    """Test output file generation."""
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_create_documents_dataframe(self, temp_config_dir, temp_output_dir, sample_extraction_result):
        """Test documents dataframe generation."""
        pipeline = PolicyBriefPipeline(
            config_dir=temp_config_dir,
            output_dir=temp_output_dir
        )
        
        df = pipeline._create_documents_dataframe([sample_extraction_result])
        
        assert len(df) == 1
        assert "doc_id" in df.columns
        assert "file_path" in df.columns
        assert "word_count" in df.columns
        assert "frames_present" in df.columns
        assert "recommendations_count" in df.columns
        
        row = df.iloc[0]
        assert row["doc_id"] == "test_doc"
        assert row["frames_present"] == 1
        assert row["recommendations_count"] == 1
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_create_frames_dataframe(self, temp_config_dir, temp_output_dir, sample_extraction_result):
        """Test frames dataframe generation."""
        pipeline = PolicyBriefPipeline(
            config_dir=temp_config_dir,
            output_dir=temp_output_dir
        )
        
        df = pipeline._create_frames_dataframe([sample_extraction_result])
        
        assert len(df) == 1
        assert "doc_id" in df.columns
        assert "frame_id" in df.columns  
        assert "decision" in df.columns
        assert "confidence" in df.columns
        assert "evidence_quotes" in df.columns
        
        row = df.iloc[0]
        assert row["doc_id"] == "test_doc"
        assert row["frame_id"] == "test_frame_1"
        assert row["decision"] == "present"
        assert row["confidence"] == 0.85
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_create_recommendations_dataframe(self, temp_config_dir, temp_output_dir, sample_extraction_result):
        """Test recommendations dataframe generation."""
        pipeline = PolicyBriefPipeline(
            config_dir=temp_config_dir,
            output_dir=temp_output_dir
        )
        
        df = pipeline._create_recommendations_dataframe([sample_extraction_result])
        
        assert len(df) == 1
        assert "doc_id" in df.columns
        assert "rec_id" in df.columns
        assert "actor" in df.columns
        assert "action" in df.columns
        assert "instrument_type" in df.columns
        assert "evidence_quotes" in df.columns
        
        row = df.iloc[0]
        assert row["doc_id"] == "test_doc"
        assert row["actor"] == "government"
        assert row["instrument_type"] == "regulation"
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.policybrief.pipeline.save_dataframe")
    def test_generate_output_files(self, mock_save_df, temp_config_dir, temp_output_dir, sample_extraction_result):
        """Test output file generation calls save functions."""
        pipeline = PolicyBriefPipeline(
            config_dir=temp_config_dir,
            output_dir=temp_output_dir
        )
        
        pipeline._generate_output_files([sample_extraction_result])
        
        # Should have called save_dataframe 3 times (documents, frames, recommendations)
        assert mock_save_df.call_count == 3
        
        # Check that dataframes were saved to correct paths
        calls = mock_save_df.call_args_list
        saved_files = [call[0][1].name for call in calls]
        
        assert "documents.csv" in saved_files
        assert "frames.csv" in saved_files  
        assert "recommendations.csv" in saved_files


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.policybrief.pdf_extractor.PDFExtractor.extract_document")
    def test_pdf_extraction_failure(self, mock_extract, temp_config_dir, temp_output_dir, mock_pdf_file):
        """Test handling of PDF extraction failures."""
        # Make PDF extraction fail
        mock_extract.side_effect = Exception("PDF extraction failed")
        
        pipeline = PolicyBriefPipeline(
            config_dir=temp_config_dir,
            output_dir=temp_output_dir
        )
        
        result = pipeline._process_single_document(mock_pdf_file)
        
        # Should return None on failure
        assert result is None
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_empty_file_list(self, temp_config_dir, temp_output_dir):
        """Test processing with empty file list."""
        pipeline = PolicyBriefPipeline(
            config_dir=temp_config_dir,
            output_dir=temp_output_dir
        )
        
        results = pipeline.process_documents([])
        
        assert results["processed"] == []
        assert results["skipped"] == []
        assert results["errors"] == []
    
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_nonexistent_files(self, temp_config_dir, temp_output_dir):
        """Test processing with nonexistent files."""
        pipeline = PolicyBriefPipeline(
            config_dir=temp_config_dir,
            output_dir=temp_output_dir
        )
        
        fake_files = [Path("/nonexistent/file1.pdf"), Path("/nonexistent/file2.pdf")]
        results = pipeline.process_documents(fake_files)
        
        # Should filter out nonexistent files
        assert results["processed"] == []
        assert results["skipped"] == []
        assert results["errors"] == []


if __name__ == "__main__":
    pytest.main([__file__])