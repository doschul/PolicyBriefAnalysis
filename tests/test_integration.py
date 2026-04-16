"""Integration tests: module switches, graceful degradation, metrics, utils."""

import pytest
from datetime import datetime
from pathlib import Path

import yaml

from src.policybrief.models import (
    DocumentMetrics,
    PageText,
    PDFMetadata,
    PerDocumentExtraction,
    ProcessingStatus,
)
from src.policybrief.metrics_calculator import MetricsCalculator
from src.policybrief.pipeline import PolicyBriefPipeline
from src.policybrief.utils import (
    clean_text_for_csv,
    create_document_id,
    validate_file_paths,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _make_config(tmp_path, modules=None):
    modules = modules or {
        "front_matter": True, "structural_core": True,
        "frames": True, "recommendations": True,
    }
    config = {
        "modules": modules,
        "openai": {"model": "gpt-4o-mini"},
        "pdf": {"extract_method": "pymupdf"},
        "frames": {"min_confidence": 0.7},
        "recommendations": {"min_confidence": 0.6},
    }
    frames = {"frames": []}
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "config.yaml").write_text(yaml.dump(config))
    (cfg_dir / "frames.yaml").write_text(yaml.dump(frames))
    return cfg_dir


# ── Module switch tests ───────────────────────────────────────────────────

class TestModuleSwitches:
    def test_all_disabled(self, tmp_path):
        cfg_dir = _make_config(tmp_path, modules={
            "front_matter": False, "structural_core": False,
            "frames": False, "recommendations": False,
        })
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        p = PolicyBriefPipeline(config_dir=cfg_dir, output_dir=out_dir)
        assert p.enable_front_matter is False
        assert p.enable_structural_core is False
        assert p.enable_frames is False
        assert p.enable_recommendations is False

    def test_partial_enable(self, tmp_path):
        cfg_dir = _make_config(tmp_path, modules={
            "front_matter": True, "structural_core": False,
            "frames": True, "recommendations": False,
        })
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        p = PolicyBriefPipeline(config_dir=cfg_dir, output_dir=out_dir)
        assert p.enable_front_matter is True
        assert p.enable_structural_core is False


# ── Metrics calculator ────────────────────────────────────────────────────

class TestMetricsCalculator:
    def setup_method(self):
        self.calc = MetricsCalculator()

    def _pages(self, texts):
        return [
            PageText(page_num=i + 1, text=t, char_count=len(t), word_count=len(t.split()))
            for i, t in enumerate(texts)
        ]

    def test_basic_metrics(self):
        pages = self._pages(["Hello world. This is a test. Another sentence here."])
        m = self.calc.calculate_metrics(pages)
        assert m.page_count == 1
        assert m.word_count > 0
        assert m.sentence_count >= 2
        assert m.paragraph_count >= 1

    def test_empty_text(self):
        pages = self._pages([""])
        m = self.calc.calculate_metrics(pages)
        assert m.word_count == 0
        assert m.sentence_count == 0
        assert m.lexical_diversity == 0.0

    def test_multi_page(self):
        pages = self._pages(["Page one content.", "Page two has more words."])
        m = self.calc.calculate_metrics(pages)
        assert m.page_count == 2
        assert m.word_count > 5

    def test_url_count(self):
        pages = self._pages(["Visit https://example.com and http://test.org today."])
        m = self.calc.calculate_metrics(pages)
        assert m.url_count == 2

    def test_email_count(self):
        pages = self._pages(["Contact us at info@example.com or support@test.org."])
        m = self.calc.calculate_metrics(pages)
        assert m.email_count == 2

    def test_lexical_diversity(self):
        pages = self._pages(["the the the dog dog cat bird."])
        m = self.calc.calculate_metrics(pages)
        assert 0 < m.lexical_diversity < 1.0

    def test_readability_populated(self):
        text = (
            "Governments should implement comprehensive monitoring systems "
            "for forest management. These systems need to track deforestation "
            "rates and biodiversity indicators over time. Regular reporting "
            "ensures transparency and accountability in forest governance."
        )
        pages = self._pages([text])
        m = self.calc.calculate_metrics(pages)
        # textstat should return values for English text
        assert m.flesch_kincaid_grade is not None or m.flesch_reading_ease is not None

    def test_passive_voice_share_computed(self):
        """passive_voice_share is populated as a float."""
        text = (
            "The report was written by the team. "
            "Forests are being destroyed at alarming rates. "
            "The government announced new policies."
        )
        pages = self._pages([text])
        m = self.calc.calculate_metrics(pages)
        assert m.passive_voice_share is not None
        assert 0.0 <= m.passive_voice_share <= 1.0

    def test_passive_voice_high_share(self):
        """Mostly passive text should have a high share."""
        text = (
            "The policy was implemented by the government. "
            "The forest was destroyed by logging companies. "
            "The report was written by researchers. "
            "New measures were adopted by the parliament."
        )
        pages = self._pages([text])
        m = self.calc.calculate_metrics(pages)
        assert m.passive_voice_share is not None
        assert m.passive_voice_share >= 0.5

    def test_passive_voice_low_share(self):
        """Mostly active text should have a low share."""
        text = (
            "The government implemented the policy. "
            "Logging companies destroyed the forest. "
            "Researchers wrote the report. "
            "Parliament adopted new measures."
        )
        pages = self._pages([text])
        m = self.calc.calculate_metrics(pages)
        assert m.passive_voice_share is not None
        assert m.passive_voice_share <= 0.5

    def test_passive_voice_empty(self):
        """Empty text returns None for passive_voice_share."""
        pages = self._pages([""])
        m = self.calc.calculate_metrics(pages)
        assert m.passive_voice_share is None


# ── Utility functions ─────────────────────────────────────────────────────

class TestUtils:
    def test_create_document_id(self):
        doc_id = create_document_id(Path("My Policy Brief.pdf"))
        assert doc_id == "my_policy_brief"
        assert doc_id[0].isalpha()

    def test_create_document_id_numeric_start(self):
        doc_id = create_document_id(Path("123report.pdf"))
        assert doc_id.startswith("doc_")

    def test_clean_text_for_csv(self):
        text = "line one\nline two\n\nline three"
        cleaned = clean_text_for_csv(text)
        assert "\n" not in cleaned

    def test_clean_text_truncation(self):
        text = "x" * 2000
        cleaned = clean_text_for_csv(text, max_length=100)
        assert len(cleaned) <= 100
        assert cleaned.endswith("...")

    def test_clean_text_empty(self):
        assert clean_text_for_csv("") == ""

    def test_validate_file_paths(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-")
        txt = tmp_path / "test.txt"
        txt.write_text("not a pdf")
        missing = tmp_path / "missing.pdf"

        valid = validate_file_paths([pdf, txt, missing])
        assert len(valid) == 1
        assert valid[0] == pdf


# ── Graceful degradation ─────────────────────────────────────────────────

class TestGracefulDegradation:
    def test_structural_core_none_ok(self):
        """PerDocumentExtraction accepts None structural_core."""
        pages = [PageText(page_num=1, text="test", char_count=4, word_count=1)]
        metrics = DocumentMetrics(
            page_count=1, word_count=1, char_count=4,
            sentence_count=1, paragraph_count=1,
        )
        status = ProcessingStatus(
            doc_id="test", file_path="/tmp/t.pdf",
            file_hash="h", file_size_bytes=100,
            processing_timestamp=datetime.now(),
            processing_duration_seconds=0.1,
            parser_used="pymupdf", likely_scanned=False,
            text_extraction_quality=0.9, pages_processed=1,
            frames_processed=0, recommendations_extracted=0,
        )
        ext = PerDocumentExtraction(
            doc_id="test", pages=pages, metadata=PDFMetadata(),
            metrics=metrics, processing_status=status,
            structural_core=None, front_matter=None,
        )
        assert ext.structural_core is None
        assert ext.front_matter is None
        assert ext.frame_assessments == []
        assert ext.policy_extractions == []

    def test_output_with_no_frames(self, tmp_path):
        """Pipeline generates output even with zero frames."""
        cfg_dir = _make_config(tmp_path, modules={
            "front_matter": False, "structural_core": False,
            "frames": False, "recommendations": False,
        })
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        p = PolicyBriefPipeline(config_dir=cfg_dir, output_dir=out_dir)

        pages = [PageText(page_num=1, text="test", char_count=4, word_count=1)]
        metrics = DocumentMetrics(
            page_count=1, word_count=1, char_count=4,
            sentence_count=1, paragraph_count=1,
        )
        status = ProcessingStatus(
            doc_id="test", file_path="/tmp/t.pdf",
            file_hash="h", file_size_bytes=100,
            processing_timestamp=datetime.now(),
            processing_duration_seconds=0.1,
            parser_used="pymupdf", likely_scanned=False,
            text_extraction_quality=0.9, pages_processed=1,
            frames_processed=0, recommendations_extracted=0,
        )
        result = PerDocumentExtraction(
            doc_id="test", pages=pages, metadata=PDFMetadata(),
            metrics=metrics, processing_status=status,
        )
        p._generate_output_files([result])
        assert (out_dir / "documents.csv").exists()
        assert (out_dir / "frames.csv").exists()
        assert (out_dir / "recommendations.csv").exists()
