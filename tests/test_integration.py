"""
Integration and regression tests for the full pipeline.

Tests graceful degradation, config switches, output table generation,
evaluation helpers, and end-to-end processing with mocked extractors.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.policybrief.models import (
    ComponentStatus,
    DocumentFrontMatter,
    DocumentMetrics,
    DocumentSection,
    DocumentSectionMap,
    Evidence,
    ExtractionType,
    FrameAssessment,
    FrameDecision,
    GeographicScope,
    ImplementationConsideration,
    ImplementationType,
    InstrumentType,
    LabelingAssessment,
    NarrativeHook,
    PageText,
    PDFMetadata,
    PerDocumentExtraction,
    PolicyExtraction,
    ProblemIdentification,
    ProcessingStatus,
    RecommendationStrength,
    SectionLabel,
    SolutionOption,
    SolutionOptionType,
    StructuralCoreResult,
    Timeframe,
    ActorType,
)
from src.policybrief.pipeline import PolicyBriefPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_config_files(config_dir: Path, modules_overrides: dict | None = None):
    """Write minimal config files; optionally override module switches."""
    modules = {
        "front_matter": True,
        "section_segmentation": True,
        "structural_core": True,
        "frames": True,
        "recommendations": True,
    }
    if modules_overrides:
        modules.update(modules_overrides)

    main_config = {
        "modules": modules,
        "openai": {"model": "gpt-4o-2024-08-06", "temperature": 0.1, "max_tokens": 4000},
        "pdf": {"extract_method": "pymupdf", "preserve_layout": True, "max_pages": 0},
        "frames": {"min_confidence": 0.7, "max_spans_per_frame": 3},
        "recommendations": {"min_confidence": 0.6, "max_recommendations": 5},
        "output": {"formats": ["csv"], "generate_audit": True},
    }

    with open(config_dir / "config.yaml", "w") as f:
        yaml.dump(main_config, f)

    frames_config = {
        "frames": [
            {
                "id": "test_frame",
                "label": "Test Frame",
                "short_definition": "A test framework",
                "inclusion_cues": ["test", "framework"],
                "exclusion_cues": [],
                "must_have": [],
            }
        ]
    }
    with open(config_dir / "frames.yaml", "w") as f:
        yaml.dump(frames_config, f)

    enums_config = {
        "instrument_types": ["regulation", "other"],
        "geographic_scopes": ["national", "unspecified"],
        "timeframes": ["short_term", "unspecified"],
        "strengths": ["should", "unspecified"],
        "actor_types": ["government", "unspecified"],
        "policy_domains": ["environment", "other"],
    }
    with open(config_dir / "enums.yaml", "w") as f:
        yaml.dump(enums_config, f)


@pytest.fixture
def pipeline_dirs():
    """Temporary config + output directories."""
    with tempfile.TemporaryDirectory() as td:
        config_dir = Path(td) / "config"
        config_dir.mkdir()
        output_dir = Path(td) / "output"
        output_dir.mkdir()
        yield config_dir, output_dir


@pytest.fixture
def mock_pdf():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4\n%%EOF")
        path = Path(f.name)
    yield path
    if path.exists():
        path.unlink()


def _make_full_extraction(doc_id: str = "test_doc") -> PerDocumentExtraction:
    """Build a complete PerDocumentExtraction with all optional fields populated."""
    pages = [PageText(page_num=1, text="Test content here.", char_count=18, word_count=3)]

    section_map = DocumentSectionMap(
        sections=[
            DocumentSection(
                raw_title="Introduction",
                normalized_label=SectionLabel.INTRODUCTION,
                start_page=1,
                end_page=1,
                confidence=0.9,
            ),
            DocumentSection(
                raw_title="Recommendations",
                normalized_label=SectionLabel.RECOMMENDATIONS,
                start_page=1,
                end_page=1,
                confidence=0.85,
            ),
        ],
        detection_method="text_heuristic",
    )

    structural_core = StructuralCoreResult(
        problem=ProblemIdentification(
            status=ComponentStatus.PRESENT,
            matched_section=SectionLabel.INTRODUCTION,
            evidence=[Evidence(page=1, quote="a policy problem exists")],
            cues_matched=["problem"],
            is_explicitly_labeled=True,
        ),
        solutions=[
            SolutionOption(
                status=ComponentStatus.PRESENT,
                option_type=SolutionOptionType.EXPLICIT_OPTION,
                matched_section=SectionLabel.RECOMMENDATIONS,
                evidence=[Evidence(page=1, quote="solution evidence")],
            )
        ],
        implementation=[
            ImplementationConsideration(
                consideration_type=ImplementationType.BARRIER,
                evidence=[Evidence(page=1, quote="barrier evidence")],
                page=1,
            )
        ],
        implementation_status=ComponentStatus.PRESENT,
        labeling=LabelingAssessment(problem_labeled=True),
    )

    front_matter = DocumentFrontMatter(
        title="Test Title",
        authors=["Author One"],
        emails=["a@example.com"],
    )

    evidence = Evidence(page=1, quote="Test evidence")
    frame_assessment = FrameAssessment(
        frame_id="test_frame",
        frame_label="Test Frame",
        decision=FrameDecision.PRESENT,
        confidence=0.85,
        evidence=[evidence],
        rationale="Found.",
    )

    policy_extraction = PolicyExtraction(
        rec_id=f"{doc_id}_rec_01",
        extraction_type=ExtractionType.RECOMMENDATION,
        confidence=0.9,
        source_text_raw="Governments should act.",
        source_section=SectionLabel.RECOMMENDATIONS,
        page=1,
        actor_text_raw="Governments",
        actor_type_normalized=ActorType.GOVERNMENT,
        action_text_raw="act",
        target_text_raw="environment",
        instrument_type=InstrumentType.REGULATION,
        policy_domain="environment",
        geographic_scope=GeographicScope.NATIONAL,
        timeframe=Timeframe.SHORT_TERM,
        strength=RecommendationStrength.SHOULD,
        evidence=[evidence],
    )

    return PerDocumentExtraction(
        doc_id=doc_id,
        pages=pages,
        headings=["Introduction", "Recommendations"],
        section_map=section_map,
        structural_core=structural_core,
        metadata=PDFMetadata(title="PDF Title"),
        front_matter=front_matter,
        metrics=DocumentMetrics(
            page_count=1, word_count=3, char_count=18,
            heading_count=2, paragraph_count=1, sentence_count=1,
            list_item_count=0, avg_sentence_length=3.0,
            lexical_diversity=1.0, avg_word_length=4.0,
        ),
        frame_assessments=[frame_assessment],
        policy_mix_present=False,
        policy_extractions=[policy_extraction],
        processing_status=ProcessingStatus(
            doc_id=doc_id, file_path="/test.pdf", file_hash="abc",
            file_size_bytes=512, processing_timestamp=datetime.now(),
            processing_duration_seconds=1.0, parser_used="pymupdf",
            likely_scanned=False, text_extraction_quality=0.95,
            pages_processed=1, frames_processed=1,
            recommendations_extracted=1,
        ),
    )


# ---------------------------------------------------------------------------
# Config switch tests
# ---------------------------------------------------------------------------

class TestModuleSwitches:
    """Verify config switches disable individual pipeline stages."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_switches_default_enabled(self, pipeline_dirs):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        assert p.enable_front_matter is True
        assert p.enable_section_segmentation is True
        assert p.enable_structural_core is True
        assert p.enable_frames is True
        assert p.enable_recommendations is True

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_switches_disable_all(self, pipeline_dirs):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir, {
            "front_matter": False,
            "section_segmentation": False,
            "structural_core": False,
            "frames": False,
            "recommendations": False,
        })
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        assert p.enable_front_matter is False
        assert p.enable_frames is False

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.policybrief.pdf_extractor.PDFExtractor.extract_document")
    def test_disabled_frames_skips_detection(
        self, mock_extract_pdf, pipeline_dirs, mock_pdf
    ):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir, {"frames": False})
        mock_extract_pdf.return_value = (
            [PageText(page_num=1, text="Policy content.", char_count=15, word_count=2)],
            PDFMetadata(title="T"),
            {"likely_scanned": False, "text_extraction_quality": 0.9, "warnings": []},
        )
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        result = p._process_single_document(mock_pdf)
        assert result is not None
        assert result.frame_assessments == []
        assert result.policy_mix_present is False

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.policybrief.pdf_extractor.PDFExtractor.extract_document")
    def test_disabled_recommendations_skips_extraction(
        self, mock_extract_pdf, pipeline_dirs, mock_pdf
    ):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir, {"recommendations": False})
        mock_extract_pdf.return_value = (
            [PageText(page_num=1, text="Policy content.", char_count=15, word_count=2)],
            PDFMetadata(title="T"),
            {"likely_scanned": False, "text_extraction_quality": 0.9, "warnings": []},
        )
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        result = p._process_single_document(mock_pdf)
        assert result is not None
        assert result.policy_extractions == []

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.policybrief.pdf_extractor.PDFExtractor.extract_document")
    def test_disabled_section_segmentation_gives_none(
        self, mock_extract_pdf, pipeline_dirs, mock_pdf
    ):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir, {"section_segmentation": False})
        mock_extract_pdf.return_value = (
            [PageText(page_num=1, text="Policy content.", char_count=15, word_count=2)],
            PDFMetadata(title="T"),
            {"likely_scanned": False, "text_extraction_quality": 0.9, "warnings": []},
        )
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        result = p._process_single_document(mock_pdf)
        assert result is not None
        assert result.section_map is None
        assert result.headings == []

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.policybrief.pdf_extractor.PDFExtractor.extract_document")
    def test_disabled_front_matter_gives_none(
        self, mock_extract_pdf, pipeline_dirs, mock_pdf
    ):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir, {"front_matter": False})
        mock_extract_pdf.return_value = (
            [PageText(page_num=1, text="Policy content.", char_count=15, word_count=2)],
            PDFMetadata(title="T"),
            {"likely_scanned": False, "text_extraction_quality": 0.9, "warnings": []},
        )
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        result = p._process_single_document(mock_pdf)
        assert result is not None
        assert result.front_matter is None


# ---------------------------------------------------------------------------
# Graceful degradation tests
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """Pipeline should continue if individual extractors raise."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.policybrief.pdf_extractor.PDFExtractor.extract_document")
    @patch("src.policybrief.section_segmenter.SectionSegmenter.segment_document")
    def test_section_segmenter_failure(
        self, mock_segment, mock_extract_pdf, pipeline_dirs, mock_pdf
    ):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        mock_extract_pdf.return_value = (
            [PageText(page_num=1, text="Content.", char_count=8, word_count=1)],
            PDFMetadata(title="T"),
            {"likely_scanned": False, "text_extraction_quality": 0.9, "warnings": []},
        )
        mock_segment.side_effect = RuntimeError("segment boom")
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        result = p._process_single_document(mock_pdf)
        assert result is not None
        assert result.section_map is None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.policybrief.pdf_extractor.PDFExtractor.extract_document")
    @patch("src.policybrief.frame_detector.FrameDetector.detect_frames")
    def test_frame_detector_failure(
        self, mock_detect, mock_extract_pdf, pipeline_dirs, mock_pdf
    ):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        mock_extract_pdf.return_value = (
            [PageText(page_num=1, text="Content.", char_count=8, word_count=1)],
            PDFMetadata(title="T"),
            {"likely_scanned": False, "text_extraction_quality": 0.9, "warnings": []},
        )
        mock_detect.side_effect = RuntimeError("frame boom")
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        result = p._process_single_document(mock_pdf)
        assert result is not None
        assert result.frame_assessments == []

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.policybrief.pdf_extractor.PDFExtractor.extract_document")
    @patch("src.policybrief.recommendation_extractor.RecommendationExtractor.extract_recommendations")
    def test_recommendation_extractor_failure(
        self, mock_recs, mock_extract_pdf, pipeline_dirs, mock_pdf
    ):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        mock_extract_pdf.return_value = (
            [PageText(page_num=1, text="Content.", char_count=8, word_count=1)],
            PDFMetadata(title="T"),
            {"likely_scanned": False, "text_extraction_quality": 0.9, "warnings": []},
        )
        mock_recs.side_effect = RuntimeError("rec boom")
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        result = p._process_single_document(mock_pdf)
        assert result is not None
        assert result.policy_extractions == []

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.policybrief.pdf_extractor.PDFExtractor.extract_document")
    @patch("src.policybrief.frontmatter_extractor.FrontMatterExtractor.extract_front_matter")
    def test_frontmatter_failure(
        self, mock_fm, mock_extract_pdf, pipeline_dirs, mock_pdf
    ):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        mock_extract_pdf.return_value = (
            [PageText(page_num=1, text="Content.", char_count=8, word_count=1)],
            PDFMetadata(title="T"),
            {"likely_scanned": False, "text_extraction_quality": 0.9, "warnings": []},
        )
        mock_fm.side_effect = RuntimeError("fm boom")
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        result = p._process_single_document(mock_pdf)
        assert result is not None
        assert result.front_matter is None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.policybrief.pdf_extractor.PDFExtractor.extract_document")
    @patch("src.policybrief.structural_core_extractor.StructuralCoreExtractor.extract")
    def test_structural_core_failure(
        self, mock_sc, mock_extract_pdf, pipeline_dirs, mock_pdf
    ):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        mock_extract_pdf.return_value = (
            [PageText(page_num=1, text="Content.", char_count=8, word_count=1)],
            PDFMetadata(title="T"),
            {"likely_scanned": False, "text_extraction_quality": 0.9, "warnings": []},
        )
        mock_sc.side_effect = RuntimeError("sc boom")
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        result = p._process_single_document(mock_pdf)
        assert result is not None
        assert result.structural_core is None


# ---------------------------------------------------------------------------
# Output table tests
# ---------------------------------------------------------------------------

class TestNewOutputTables:
    """Test sections and structural core dataframes."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_sections_dataframe(self, pipeline_dirs):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        extraction = _make_full_extraction()
        df = p._create_sections_dataframe([extraction])
        assert len(df) == 2
        assert list(df.columns) == [
            "doc_id", "section_index", "raw_title", "normalized_label",
            "start_page", "end_page", "confidence", "rule_source",
            "detection_method",
        ]
        assert df.iloc[0]["normalized_label"] == "introduction"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_sections_empty_when_no_section_map(self, pipeline_dirs):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        extraction = _make_full_extraction()
        extraction.section_map = None
        df = p._create_sections_dataframe([extraction])
        assert df.empty

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_structural_core_dataframe(self, pipeline_dirs):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        extraction = _make_full_extraction()
        df = p._create_structural_core_dataframe([extraction])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["problem_status"] == "present"
        assert row["solutions_count"] == 1
        assert row["solutions_explicit"] == 1
        assert row["implementation_count"] == 1

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_structural_core_empty_when_missing(self, pipeline_dirs):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        extraction = _make_full_extraction()
        extraction.structural_core = None
        df = p._create_structural_core_dataframe([extraction])
        assert df.empty

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("src.policybrief.pipeline.save_dataframe")
    def test_generate_output_includes_new_tables(self, mock_save, pipeline_dirs):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        extraction = _make_full_extraction()
        p._generate_output_files([extraction])
        saved = [call[0][1].name for call in mock_save.call_args_list]
        assert "sections.csv" in saved
        assert "structural_core.csv" in saved


# ---------------------------------------------------------------------------
# Evaluation helper tests
# ---------------------------------------------------------------------------

class TestEvaluationSummary:

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_summary_all_populated(self, pipeline_dirs):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        results = [_make_full_extraction("doc1"), _make_full_extraction("doc2")]
        summary = p.compute_extraction_summary(results)
        assert summary["document_count"] == 2
        assert summary["front_matter_null"] == 0
        assert summary["total_extractions"] == 2
        assert summary["avg_extractions_per_doc"] == 1.0
        assert summary["warnings"] == []

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_summary_empty_results(self, pipeline_dirs):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        summary = p.compute_extraction_summary([])
        assert summary["document_count"] == 0

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_summary_warns_all_null(self, pipeline_dirs):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        ext = _make_full_extraction()
        ext.front_matter = None
        ext.section_map = None
        ext.policy_extractions = []
        summary = p.compute_extraction_summary([ext])
        assert "all_front_matter_null" in summary["warnings"]
        assert "all_section_maps_null" in summary["warnings"]
        assert "zero_recommendations_extracted" in summary["warnings"]

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_summary_warns_high_reference_extractions(self, pipeline_dirs):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        ext = _make_full_extraction()
        # Add more extractions from references section
        for i in range(4):
            ext.policy_extractions.append(PolicyExtraction(
                rec_id=f"ref_{i}",
                extraction_type=ExtractionType.RECOMMENDATION,
                confidence=0.7,
                source_text_raw="ref text",
                source_section=SectionLabel.REFERENCES,
                page=1,
                instrument_type=InstrumentType.OTHER,
                evidence=[Evidence(page=1, quote="some reference text here")],
            ))
        summary = p.compute_extraction_summary([ext])
        assert "high_reference_section_extractions" in summary["warnings"]


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

class TestCacheBehaviour:

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_needs_processing_first_time(self, pipeline_dirs, mock_pdf):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        assert p._needs_processing(mock_pdf) is True

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_needs_processing_cached(self, pipeline_dirs, mock_pdf):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        from src.policybrief.pdf_extractor import PDFExtractor
        file_hash = PDFExtractor.compute_file_hash(mock_pdf)
        p.processing_cache[str(mock_pdf)] = file_hash
        assert p._needs_processing(mock_pdf) is False

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    def test_force_reprocess_ignores_cache(self, pipeline_dirs, mock_pdf):
        config_dir, output_dir = pipeline_dirs
        _write_config_files(config_dir)
        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir,
                                force_reprocess=True)
        from src.policybrief.pdf_extractor import PDFExtractor
        file_hash = PDFExtractor.compute_file_hash(mock_pdf)
        p.processing_cache[str(mock_pdf)] = file_hash
        assert p._needs_processing(mock_pdf) is True


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:

    def test_legacy_recommendations_field_still_accepted(self):
        """PerDocumentExtraction still accepts recommendations=[]."""
        ext = _make_full_extraction()
        assert ext.recommendations == []
        data = ext.model_dump()
        assert "recommendations" in data

    def test_model_dump_includes_all_new_fields(self):
        ext = _make_full_extraction()
        data = ext.model_dump()
        assert "section_map" in data
        assert "structural_core" in data
        assert "front_matter" in data
        assert "policy_mix_present" in data
        assert "policy_extractions" in data
