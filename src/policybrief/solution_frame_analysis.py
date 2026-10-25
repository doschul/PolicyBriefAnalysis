"""Classify recommendation segments against the policy-instrument frames."""

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from .llm_client import LLMClient
from .models import FrameExtractionResponse
from .utils import save_dataframe

logger = logging.getLogger(__name__)

_PROMPT = """Classify this single policy recommendation or policy option against
each supplied policy-instrument frame. Determine which instrument mechanisms
are explicitly present in the proposed solution, not merely mentioned as
background. Assess frames independently; multiple frames may be present.
Use absent when unsupported and insufficient_evidence when the passage is too
ambiguous. Use other only for a concrete instrument that fits none of the
defined frames; do not use it for missing or vague information. Evidence quotes
must be verbatim from the passage and at least 10 characters. Return a result
for every supplied frame."""

_OUTPUT_COLUMNS = [
    "solution_key",
    "input_source",
    "doc_id",
    "parent_id",
    "source_provenance",
    "source_analysis_level",
    "analysis_level",
    "solution_type",
    "source_text",
    "frame_id",
    "frame_label",
    "decision",
    "confidence",
    "evidence",
    "rationale",
    "classification_status",
    "error",
]


class SolutionFrameAnalyzer:
    """Classify solution-level passages independently of document-level frames."""

    def __init__(
        self, llm_client: LLMClient, frames: list[dict[str, Any]]
    ) -> None:
        self.llm = llm_client
        self.frames = frames
        self.frame_ids = {frame["id"] for frame in frames}

    def analyze(
        self,
        source_path: Path,
        output_dir: Path,
        input_source: str,
        max_workers: int = 1,
        checkpoint_every: int = 25,
    ) -> pd.DataFrame:
        if input_source not in {"ai", "manual"}:
            message = "input_source must be 'ai' or 'manual'"
            raise ValueError(message)
        if max_workers < 1 or checkpoint_every < 1:
            message = "max_workers and checkpoint_every must be positive"
            raise ValueError(message)

        source_path, output_dir = Path(source_path), Path(output_dir)
        inputs = self._load_inputs(source_path, input_source)
        output_path = output_dir / f"solution_frames_{input_source}.csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        source_columns = list(
            dict.fromkeys(
                column
                for item in inputs
                for column in item["source_fields"]
            )
        )
        output_columns = list(dict.fromkeys([*_OUTPUT_COLUMNS, *source_columns]))

        existing = (
            pd.read_csv(output_path).reindex(columns=output_columns)
            if output_path.exists()
            else pd.DataFrame(columns=output_columns)
        )
        completed_keys = {
            key
            for key, group in existing.groupby("solution_key")
            if set(
                group.loc[
                    group["classification_status"] == "classified", "frame_id"
                ]
            )
            == self.frame_ids
        }
        pending = [
            item for item in inputs if item["solution_key"] not in completed_keys
        ]
        logger.info(
            "Solution-frame classification (%s): %d passages (%d already complete)",
            input_source,
            len(pending),
            len(completed_keys),
        )

        batch: list[dict[str, Any]] = []

        def checkpoint() -> None:
            nonlocal existing, batch
            if not batch:
                return
            retry_keys = {row["solution_key"] for row in batch}
            existing = existing[~existing["solution_key"].isin(retry_keys)]
            updated = pd.concat(
                [existing, pd.DataFrame(batch).reindex(columns=output_columns)],
                ignore_index=True,
            )
            temp_path = output_path.with_suffix(".tmp")
            save_dataframe(updated, temp_path)
            temp_path.replace(output_path)
            existing, batch = updated, []

        if max_workers == 1:
            results = map(lambda item: self._classify(item, input_source), pending)
            for result in results:
                batch.extend(result)
                if len(batch) >= checkpoint_every * len(self.frames):
                    checkpoint()
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._classify, item, input_source)
                    for item in pending
                ]
                for future in as_completed(futures):
                    batch.extend(future.result())
                    if len(batch) >= checkpoint_every * len(self.frames):
                        checkpoint()
        checkpoint()
        return existing

    @staticmethod
    def _load_inputs(
        source_path: Path, input_source: str
    ) -> list[dict[str, Any]]:
        df = pd.read_csv(source_path)
        text_col = (
            "source_text_raw"
            if "source_text_raw" in df.columns
            else "source_text"
        )
        required = {"doc_id", "extraction_type", text_col}
        missing = required - set(df.columns)
        if missing:
            message = (
                f"{source_path} is missing required columns: {sorted(missing)}"
            )
            raise ValueError(message)

        df = df[
            df["extraction_type"]
            .astype(str)
            .str.lower()
            .isin({"recommendation", "policy_option"})
        ]
        result = []
        for idx, row in df.iterrows():
            text = str(row[text_col]).strip()
            if not text or text.lower() == "nan":
                continue
            parent_value = row.get("rec_id")
            if pd.isna(parent_value) or not str(parent_value).strip():
                parent_value = row.get("parent_id")
            if pd.isna(parent_value) or not str(parent_value).strip():
                parent_value = idx
            parent_id = str(parent_value)
            provenance_value = row.get("source_provenance")
            if pd.isna(provenance_value) or not str(provenance_value).strip():
                provenance_value = (
                    "full_policy_brief_text" if input_source == "ai"
                    else "manually_coded_recommendation_segment"
                )
            doc_id = str(row["doc_id"])
            solution_key = hashlib.sha256(
                f"{input_source}\0{doc_id}\0{parent_id}\0{text}".encode()
            ).hexdigest()
            source_fields = {}
            for original_column, original_value in row.items():
                value = None if pd.isna(original_value) else original_value
                if original_column == "analysis_level":
                    output_column = "source_analysis_level"
                elif original_column == "confidence":
                    output_column = "source_confidence"
                elif original_column in _OUTPUT_COLUMNS and original_column not in {
                    "input_source",
                    "doc_id",
                    "parent_id",
                    "source_provenance",
                }:
                    output_column = f"source_{original_column}"
                else:
                    output_column = original_column
                source_fields[output_column] = value
            source_fields["input_source"] = input_source
            source_fields["source_analysis_level"] = (
                source_fields.get("source_analysis_level")
            )
            result.append({
                "solution_key": solution_key,
                "input_source": input_source,
                "doc_id": doc_id,
                "parent_id": parent_id,
                "source_provenance": str(provenance_value),
                "source_analysis_level": source_fields.get(
                    "source_analysis_level"
                ),
                "solution_type": str(row["extraction_type"]).lower(),
                "source_text": text,
                "source_fields": source_fields,
            })
        return result

    def _classify(
        self, item: dict[str, Any], input_source: str
    ) -> list[dict[str, Any]]:
        item_data = {
            key: value for key, value in item.items() if key != "source_fields"
        }
        item_data.update(item["source_fields"])
        definitions = []
        for frame in self.frames:
            definition = [
                f"{frame['id']}: {frame['label']}",
                f"Definition: {frame['short_definition']}",
            ]
            if frame.get("analytical_notes"):
                definition.append(f"Guidance: {frame['analytical_notes']}")
            if frame.get("false_positive_notes"):
                definition.append(f"False positives: {frame['false_positive_notes']}")
            definitions.append("\n".join(definition))
        messages = [
            {"role": "system", "content": _PROMPT},
            {
                "role": "user",
                "content": (
                    "Frames:\n" + "\n\n".join(definitions)
                    + "\n\nSolution passage:\n" + item["source_text"]
                ),
            },
        ]
        try:
            response: FrameExtractionResponse = self.llm.structured_completion(
                messages, FrameExtractionResponse
            )
            by_id = {result.frame_id: result for result in response.frames}
            if len(by_id) != len(self.frames) or set(by_id) != self.frame_ids:
                message = "LLM response did not return exactly one result per frame"
                raise ValueError(message)
            rows = []
            for frame in self.frames:
                result = by_id[frame["id"]]
                evidence = [
                    {"page": quote.page, "quote": quote.quote}
                    for quote in result.evidence
                    if quote.quote and quote.quote in item["source_text"]
                ]
                decision = result.decision.value
                rationale = result.rationale
                if result.decision.value == "present" and not evidence:
                    decision = "insufficient_evidence"
                    rationale = (
                        f"{rationale} Present classification downgraded because "
                        "no evidence quote matched the passage verbatim."
                    )
                rows.append({
                    **item_data,
                    "analysis_level": "solution_level_instrument_frame",
                    "frame_id": frame["id"],
                    "frame_label": frame["label"],
                    "decision": decision,
                    "confidence": result.confidence,
                    "evidence": json.dumps(evidence, ensure_ascii=False),
                    "rationale": rationale,
                    "classification_status": "classified",
                    "error": "",
                })
            return rows
        except Exception as exc:
            logger.warning(
                "Solution-frame classification failed (%s, %s): %s",
                input_source,
                item["parent_id"],
                exc,
            )
            return [{
                **item_data,
                "analysis_level": "solution_level_instrument_frame",
                "frame_id": "",
                "frame_label": "",
                "decision": "",
                "confidence": None,
                "evidence": "[]",
                "rationale": "",
                "classification_status": "error",
                "error": str(exc),
            }]
