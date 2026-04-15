"""Tests for the pipeline: init, single-doc processing, output generation."""

import json
import os
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.policybrief.models import (
    DocumentFrontMatter,
    DocumentMetrics,
    FrameAssessment,
    FrameDecision,
    PageText,
    PDFMetadata,
    PerDocumentExtraction,
    PolicyExtraction,
    ProcessingStatus,
    StructuralCoreResult,
)
from src.policybrief.pipeline import PolicyBriefPipeline


@pytest.fixture
def config_dir(tmp_path):
    """Create minimal config directory."""
    import yaml

    config = {
        "modules": {
            "front_matter": True,
            "structural_core": True,
            "frames": True,
            "recommendations": True,
        },
        "openai": {"model": "gpt-4o-mini", "temperature": 0.1, "max_tokens": 4000},
        "pdf": {"extract_method": "pymupdf", "max_pages": 0, "max_file_size_mb": 50},
        "frames": {"min_confidence": 0.7, "max_spans_per_frame": 5},
        "recommendations": {"min_confidence": 0.6},
    }
    frames = {"frames": []}

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(yaml.dump(config))
    (cfg_dir / "frames.yaml").write_text(yaml.dump(frames))
    return cfg_dir


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def pipeline(config_dir, output_dir):
    return PolicyBriefPipeline(
        config_dir=config_dir,
        output_dir=output_dir,
        max_workers=1,
    )


# ── Pipeline initialisation ──────────────────────────────────────────────

class TestPipelineInit:
    def test_basic_init(self, pipeline):
        assert pipeline.max_workers == 1
        assert pipeline.enable_front_matter is True
        assert pipeline.enable_frames is True

    def test_module_switches(self, config_dir, output_dir):
        """Modules can be disabled via config."""
        import yaml
        cfg_path = config_dir / "config.yaml"
        config = yaml.safe_load(cfg_path.read_text())
        config["modules"]["frames"] = False
        config["modules"]["recommendations"] = False
        cfg_path.write_text(yaml.dump(config))

        p = PolicyBriefPipeline(config_dir=config_dir, output_dir=output_dir)
        assert p.enable_frames is False
        assert p.enable_recommendations is False


# ── Extraction summary ───────────────────────────────────────────────────

class TestExtractionSummary:
    def _make_result(self, doc_id="test_doc", n_frames=0, n_recs=0):
        pages = [PageText(page_num=1, text="text", char_count=4, word_count=1)]
        metrics = DocumentMetrics(
            page_count=1, word_count=100, char_count=500,
            sentence_count=5, paragraph_count=2,
        )
        status = ProcessingStatus(
            doc_id=doc_id, file_path="/tmp/test.pdf",
            file_hash="abc123", file_size_bytes=1000,
            processing_timestamp=datetime.now(),
            processing_duration_seconds=1.0,
            parser_used="pymupdf", likely_scanned=False,
            text_extraction_quality=0.9, pages_processed=1,
            frames_processed=n_frames, recommendations_extracted=n_recs,
        )
        return PerDocumentExtraction(
            doc_id=doc_id, pages=pages,
            metadata=PDFMetadata(), metrics=metrics,
            processing_status=status,
        )

    def test_empty_results(self, pipeline):
        summary = pipeline.compute_extraction_summary([])
        assert summary["documents_processed"] == 0
        assert summary["total_extractions"] == 0

    def test_with_results(self, pipeline):
        results = [self._make_result("doc1"), self._make_result("doc2")]
        summary = pipeline.compute_extraction_summary(results)
        assert summary["documents_processed"] == 2
        assert summary["total_pages"] == 2


# ── Output file generation ───────────────────────────────────────────────

class TestOutputFiles:
    def _make_result(self, doc_id="test_doc"):
        pages = [PageText(page_num=1, text="test text", char_count=9, word_count=2)]
        metrics = DocumentMetrics(
            page_count=1, word_count=100, char_count=500,
            sentence_count=5, paragraph_count=2,
        )
        status = ProcessingStatus(
            doc_id=doc_id, file_path="/tmp/test.pdf",
            file_hash="abc123", file_size_bytes=1000,
            processing_timestamp=datetime.now(),
            processing_duration_seconds=1.0,
            parser_used="pymupdf", likely_scanned=False,
            text_extraction_quality=0.9, pages_processed=1,
            frames_processed=0, recommendations_extracted=0,
        )
        return PerDocumentExtraction(
            doc_id=doc_id, pages=pages,
            metadata=PDFMetadata(title="Test Doc"),
            front_matter=DocumentFrontMatter(title="Test Brief", authors=["Auth A"]),
            metrics=metrics,
            structural_core=StructuralCoreResult(
                problem_status="present",
                problem_summary="Deforestation",
                solutions_count=2,
            ),
            processing_status=status,
        )

    def test_documents_csv_created(self, pipeline):
        results = [self._make_result()]
        pipeline._generate_output_files(results)
        assert (pipeline.output_dir / "documents.csv").exists()

    def test_frames_csv_created(self, pipeline):
        results = [self._make_result()]
        pipeline._generate_output_files(results)
        assert (pipeline.output_dir / "frames.csv").exists()

    def test_recommendations_csv_created(self, pipeline):
        results = [self._make_result()]
        pipeline._generate_output_files(results)
        assert (pipeline.output_dir / "recommendations.csv").exists()

    def test_structural_core_csv_created(self, pipeline):
        results = [self._make_result()]
        pipeline._generate_output_files(results)
        assert (pipeline.output_dir / "structural_core.csv").exists()

    def test_documents_csv_content(self, pipeline):
        import pandas as pd
        results = [self._make_result()]
        pipeline._generate_output_files(results)
        df = pd.read_csv(pipeline.output_dir / "documents.csv")
        assert len(df) == 1
        assert df.iloc[0]["doc_id"] == "test_doc"
        assert df.iloc[0]["fm_title"] == "Test Brief"

    def test_structural_core_csv_content(self, pipeline):
        import pandas as pd
        results = [self._make_result()]
        pipeline._generate_output_files(results)
        df = pd.read_csv(pipeline.output_dir / "structural_core.csv")
        assert len(df) == 1
        assert df.iloc[0]["problem_status"] == "present"
        assert df.iloc[0]["solutions_count"] == 2


# ── Sparse / scanned document handling ───────────────────────────────────

class TestSparseDocuments:
    def test_scanned_detection(self, pipeline):
        """Scanned documents should not crash the pipeline."""
        sparse_pages = [
            PageText(page_num=1, text="", char_count=0, word_count=0),
            PageText(page_num=2, text="a", char_count=1, word_count=1),
        ]
        likely_scanned, quality = pipeline.pdf_extractor.detect_scanned(sparse_pages)
        assert likely_scanned is True
        assert quality == 0.0

    def test_empty_pages(self, pipeline):
        pages = []
        likely_scanned, quality = pipeline.pdf_extractor.detect_scanned(pages)
        assert likely_scanned is True
