from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from src.policybrief.models import (
    Evidence,
    FrameDecision,
    FrameExtractionResponse,
    SingleFrameResult,
)
from src.policybrief.solution_frame_analysis import SolutionFrameAnalyzer


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def structured_completion(
        self,
        _messages: list[dict[str, str]],
        _response_model: type[BaseModel],
    ) -> FrameExtractionResponse:
        self.calls += 1
        return FrameExtractionResponse(frames=[
            SingleFrameResult(
                frame_id="command",
                decision=FrameDecision.PRESENT,
                confidence=0.9,
                evidence=[Evidence(page=1, quote="A concrete policy action")],
                rationale="Explicit instrument.",
            ),
            SingleFrameResult(
                frame_id="other",
                decision=FrameDecision.PRESENT,
                confidence=0.9,
                evidence=[Evidence(page=1, quote="This quote is not in passage.")],
                rationale="Concrete instrument of another type.",
            ),
        ])


def test_analyzer_writes_separate_resumable_solution_frame_data(
    tmp_path: Path,
) -> None:
    source = tmp_path / "solutions.csv"
    pd.DataFrame([
        {
            "doc_id": "brief",
            "rec_id": "solution-1",
            "input_source": "ai",
            "source_provenance": "ai_extracted_recommendation_segment",
            "analysis_level": "recommendation_segment",
            "extraction_type": "recommendation",
            "source_text_raw": "A concrete policy action",
            "page": 3,
            "instrument_type": "economic",
        },
        {
            "doc_id": "brief",
            "rec_id": "not-solution",
            "extraction_type": "expected_outcome",
            "source_text_raw": "An outcome",
        },
    ]).to_csv(source, index=False)
    llm = FakeLLM()
    analyzer = SolutionFrameAnalyzer(llm, [
        {"id": "command", "label": "Command", "short_definition": "Binding rules."},
        {"id": "other", "label": "Other", "short_definition": "Other instruments."},
    ])

    first = analyzer.analyze(source, tmp_path / "out", "ai")
    second = analyzer.analyze(source, tmp_path / "out", "ai")

    assert len(first) == len(analyzer.frames)
    assert len(second) == len(analyzer.frames)
    assert llm.calls == 1
    assert set(first["analysis_level"]) == {"solution_level_instrument_frame"}
    assert set(first["input_source"]) == {"ai"}
    assert set(first["source_provenance"]) == {
        "ai_extracted_recommendation_segment"
    }
    assert set(first["source_analysis_level"]) == {"recommendation_segment"}
    assert set(first["source_text_raw"]) == {"A concrete policy action"}
    assert set(first["page"]) == {3}
    assert set(first["instrument_type"]) == {"economic"}
    assert set(first["decision"]) == {"present", "insufficient_evidence"}
    assert first.loc[
        first["frame_id"].eq("other"), "rationale"
    ].str.contains("downgraded", case=False).all()
    assert (tmp_path / "out" / "solution_frames_ai.csv").exists()
