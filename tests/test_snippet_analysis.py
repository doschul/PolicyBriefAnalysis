"""Tests for SnippetAnalyzer (snippet-level extraction for both ai and manual modes)."""
import csv
from pathlib import Path

import pandas as pd
import pytest

from src.policybrief.models import (
    ExtractionType,
    RecommendationExtractionResponse,
    RecommendationItem,
)
from src.policybrief.snippet_analysis import SnippetAnalyzer


# -- Fake LLMs -------------------------------------------------------------

class FakeLLMSnippetRec:
    """Returns one recommendation whose quote is the first 60 chars of the input."""

    def structured_completion(self, messages, response_model):
        user_msg = messages[-1]["content"]
        lines = user_msg.split("\n\n", 1)
        snippet = lines[1].strip() if len(lines) > 1 else user_msg
        quote = snippet[:60] if len(snippet) >= 10 else "governments should act now"
        return RecommendationExtractionResponse(items=[
            RecommendationItem(
                extraction_type=ExtractionType.RECOMMENDATION,
                confidence=0.85,
                source_quote=quote,
                page=1,
                actor_text_raw="governments",
                action_text_raw="act on policy",
                strength="should",
            )
        ])


class FakeLLMEmpty:
    def structured_completion(self, messages, response_model):
        return RecommendationExtractionResponse(items=[])


# -- CSV helpers -----------------------------------------------------------

def _write_recs_csv(path: Path, rows):
    fieldnames = ["rec_id", "doc_id", "source_text_raw", "page", "extraction_type"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manual_excel(path: Path, rows):
    df = pd.DataFrame(rows, columns=["Document name", "Solution type", "Segment"])
    df.to_excel(path, index=False)


# -- AI mode tests ---------------------------------------------------------

class TestSnippetAnalyzerAI:
    def test_writes_ai_output_csv(self, tmp_path):
        recs_csv = tmp_path / "recommendations.csv"
        snippet_text = "Governments should strengthen forest monitoring systems nationally."
        _write_recs_csv(recs_csv, [
            {"rec_id": "doc1_rec_001", "doc_id": "doc1", "source_text_raw": snippet_text,
             "page": 2, "extraction_type": "recommendation"},
        ])

        analyzer = SnippetAnalyzer(llm_client=FakeLLMSnippetRec(), config={"min_confidence": 0.6})
        result = analyzer.analyze(recs_csv, tmp_path, input_source="ai")

        out_path = tmp_path / "recommendations_snippet_ai.csv"
        assert out_path.exists()
        df = pd.read_csv(out_path)
        assert len(df) >= 1
        assert df.iloc[0]["doc_id"] == "doc1"
        assert df.iloc[0]["parent_id"] == "doc1_rec_001"
        assert df.iloc[0]["input_source"] == "ai"

    def test_ai_empty_source_text_skipped(self, tmp_path):
        recs_csv = tmp_path / "recommendations.csv"
        _write_recs_csv(recs_csv, [
            {"rec_id": "doc1_rec_001", "doc_id": "doc1", "source_text_raw": "",
             "page": 1, "extraction_type": "recommendation"},
            {"rec_id": "doc1_rec_002", "doc_id": "doc1", "source_text_raw": None,
             "page": 1, "extraction_type": "recommendation"},
        ])

        analyzer = SnippetAnalyzer(llm_client=FakeLLMEmpty(), config={})
        result = analyzer.analyze(recs_csv, tmp_path, input_source="ai")
        assert len(result) == 0

    def test_ai_missing_column_raises(self, tmp_path):
        recs_csv = tmp_path / "bad.csv"
        pd.DataFrame({"doc_id": ["doc1"]}).to_csv(recs_csv, index=False)

        analyzer = SnippetAnalyzer(llm_client=FakeLLMEmpty(), config={})
        with pytest.raises(ValueError, match="source_text"):
            analyzer.analyze(recs_csv, tmp_path, input_source="ai")

    def test_ai_multiple_snippets_same_doc(self, tmp_path):
        recs_csv = tmp_path / "recommendations.csv"
        _write_recs_csv(recs_csv, [
            {"rec_id": "doc1_rec_001", "doc_id": "doc1",
             "source_text_raw": "Governments should strengthen forest monitoring systems nationally.",
             "page": 1, "extraction_type": "recommendation"},
            {"rec_id": "doc1_rec_002", "doc_id": "doc1",
             "source_text_raw": "Policymakers must improve biodiversity data collection processes.",
             "page": 3, "extraction_type": "recommendation"},
        ])

        analyzer = SnippetAnalyzer(llm_client=FakeLLMSnippetRec(), config={"min_confidence": 0.6})
        result = analyzer.analyze(recs_csv, tmp_path, input_source="ai")
        assert len(result) >= 2
        assert set(result["doc_id"]) == {"doc1"}

    def test_ai_required_output_columns(self, tmp_path):
        recs_csv = tmp_path / "recommendations.csv"
        _write_recs_csv(recs_csv, [
            {"rec_id": "doc1_rec_001", "doc_id": "doc1",
             "source_text_raw": "Governments should strengthen forest monitoring systems nationally.",
             "page": 1, "extraction_type": "recommendation"},
        ])

        analyzer = SnippetAnalyzer(llm_client=FakeLLMSnippetRec(), config={"min_confidence": 0.6})
        result = analyzer.analyze(recs_csv, tmp_path, input_source="ai")

        required = {"doc_id", "parent_id", "input_source", "rec_id", "extraction_type", "confidence", "source_text_raw", "page"}
        assert required.issubset(set(result.columns))

    def test_ai_default_input_source(self, tmp_path):
        """Default input_source is ai."""
        recs_csv = tmp_path / "recommendations.csv"
        _write_recs_csv(recs_csv, [
            {"rec_id": "doc1_rec_001", "doc_id": "doc1",
             "source_text_raw": "Governments should strengthen forest monitoring systems nationally.",
             "page": 1, "extraction_type": "recommendation"},
        ])

        analyzer = SnippetAnalyzer(llm_client=FakeLLMSnippetRec(), config={"min_confidence": 0.6})
        result = analyzer.analyze(recs_csv, tmp_path)  # no input_source kwarg
        assert (tmp_path / "recommendations_snippet_ai.csv").exists()


# -- Manual mode tests -----------------------------------------------------

class TestSnippetAnalyzerManual:
    def test_writes_manual_output_csv(self, tmp_path):
        segments_xlsx = tmp_path / "segments.xlsx"
        snippet_text = "Governments should strengthen forest monitoring systems nationally."
        _write_manual_excel(segments_xlsx, [
            {"Document name": "my_doc.pdf", "Solution type": "regulation", "Segment": snippet_text},
        ])

        analyzer = SnippetAnalyzer(llm_client=FakeLLMSnippetRec(), config={"min_confidence": 0.6})
        result = analyzer.analyze(segments_xlsx, tmp_path, input_source="manual")

        out_path = tmp_path / "recommendations_snippet_manual.csv"
        assert out_path.exists()
        df = pd.read_csv(out_path)
        assert len(df) >= 1
        assert df.iloc[0]["input_source"] == "manual"

    def test_manual_doc_name_mapped_to_doc_id(self, tmp_path):
        segments_xlsx = tmp_path / "segments.xlsx"
        _write_manual_excel(segments_xlsx, [
            {"Document name": "My Policy Brief.pdf", "Solution type": "subsidy",
             "Segment": "Governments should strengthen forest monitoring systems nationally."},
        ])

        analyzer = SnippetAnalyzer(llm_client=FakeLLMSnippetRec(), config={"min_confidence": 0.6})
        result = analyzer.analyze(segments_xlsx, tmp_path, input_source="manual")

        # create_document_id lowercases and replaces spaces with underscores
        assert len(result) >= 1
        assert "my" in result.iloc[0]["doc_id"].lower()

    def test_manual_missing_segment_column_raises(self, tmp_path):
        xlsx = tmp_path / "bad.xlsx"
        pd.DataFrame({"Document name": ["doc1.pdf"]}).to_excel(xlsx, index=False)

        analyzer = SnippetAnalyzer(llm_client=FakeLLMEmpty(), config={})
        with pytest.raises(ValueError, match="Segment"):
            analyzer.analyze(xlsx, tmp_path, input_source="manual")

    def test_manual_empty_segment_skipped(self, tmp_path):
        segments_xlsx = tmp_path / "segments.xlsx"
        _write_manual_excel(segments_xlsx, [
            {"Document name": "doc1.pdf", "Solution type": "regulation", "Segment": ""},
            {"Document name": "doc1.pdf", "Solution type": "regulation", "Segment": None},
        ])

        analyzer = SnippetAnalyzer(llm_client=FakeLLMEmpty(), config={})
        result = analyzer.analyze(segments_xlsx, tmp_path, input_source="manual")
        assert len(result) == 0

    def test_manual_multiple_docs(self, tmp_path):
        segments_xlsx = tmp_path / "segments.xlsx"
        snippet = "Governments should strengthen forest monitoring systems nationally."
        _write_manual_excel(segments_xlsx, [
            {"Document name": "doc_a.pdf", "Solution type": "regulation", "Segment": snippet},
            {"Document name": "doc_b.pdf", "Solution type": "subsidy", "Segment": snippet},
        ])

        analyzer = SnippetAnalyzer(llm_client=FakeLLMSnippetRec(), config={"min_confidence": 0.6})
        result = analyzer.analyze(segments_xlsx, tmp_path, input_source="manual")
        assert len(set(result["doc_id"])) == 2

