"""Tests for cross-validation source provenance contracts."""

import pandas as pd
import pytest

from src.policybrief.cross_validator import CrossValidator


def _write_ai_snippets(path):
    pd.DataFrame([
        {
            "doc_id": "brief",
            "rec_id": "brief_snip_001",
            "extraction_type": "recommendation",
            "source_text_raw": "Government should strengthen forest monitoring.",
            "source_provenance": "ai_extracted_recommendation_segment",
            "analysis_level": "recommendation_segment",
        }
    ]).to_csv(path, index=False)


def _write_manual_segments(path):
    pd.DataFrame([
        {
            "Document name": "brief.pdf",
            "Solution type": "Recommendation",
            "Segment": "Government should strengthen forest monitoring.",
        }
    ]).to_excel(path, index=False)


def test_compare_records_explicit_validation_sources(tmp_path):
    ai_path = tmp_path / "recommendations_snippet_ai.csv"
    manual_path = tmp_path / "manual_segments.xlsx"
    _write_ai_snippets(ai_path)
    _write_manual_segments(manual_path)

    metrics = CrossValidator().compare(ai_path, manual_path, tmp_path / "crossval")

    assert metrics["validation_inputs"]["ai"]["provenance"] == "ai_extracted_recommendation_segment"
    assert metrics["validation_inputs"]["manual"]["provenance"] == "manually_coded_recommendation_segment"
    comparison = pd.read_csv(tmp_path / "crossval" / "comparison.csv")
    assert comparison.iloc[0]["ai_source_provenance"] == "ai_extracted_recommendation_segment"
    assert comparison.iloc[0]["manual_source_provenance"] == "manually_coded_recommendation_segment"


def test_compare_rejects_manual_snippet_csv(tmp_path):
    ai_path = tmp_path / "recommendations_snippet_ai.csv"
    manual_path = tmp_path / "recommendations_snippet_manual.csv"
    _write_ai_snippets(ai_path)
    pd.DataFrame().to_csv(manual_path, index=False)

    with pytest.raises(ValueError, match="raw manually coded"):
        CrossValidator().compare(ai_path, manual_path, tmp_path / "crossval")


def test_compare_rejects_ai_input_without_provenance(tmp_path):
    ai_path = tmp_path / "recommendations_snippet_ai.csv"
    manual_path = tmp_path / "manual_segments.xlsx"
    _write_ai_snippets(ai_path)
    _write_manual_segments(manual_path)
    ai_df = pd.read_csv(ai_path).drop(columns=["source_provenance"])
    ai_df.to_csv(ai_path, index=False)

    with pytest.raises(ValueError, match="provenance column"):
        CrossValidator().compare(ai_path, manual_path, tmp_path / "crossval")