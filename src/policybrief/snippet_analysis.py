"""
Snippet-level (second-pass and third-pass) recommendation extraction.

Re-runs the standard RECOMMENDATION_PROMPT on pre-extracted solution
sentences with ``input_mode="snippet"``.  Supports two input sources:

- ``"ai"``: AI-extracted solution sentences from ``recommendations.csv``
  (``source_text_raw`` column).  Produces ``recommendations_snippet_ai.csv``.

- ``"manual"``: Manually coded solution segments from
  ``PBs solutions coded segments.xlsx`` (``Segment`` column).
  Produces ``recommendations_snippet_manual.csv``.

Data flow:
    Pass 1 (pipeline): full PDF → recommendations.csv
    Pass 2 (ai):       source_text_raw   → recommendations_snippet_ai.csv
    Pass 3 (manual):   Segment column    → recommendations_snippet_manual.csv
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from .llm_client import LLMClient
from .models import PageText, PolicyExtraction
from .recommendation_extractor import RecommendationExtractor
from .utils import create_document_id, save_dataframe

logger = logging.getLogger(__name__)

# Column names for the manual segments Excel
_MANUAL_DOC_COL = "Document name"
_MANUAL_TEXT_COL = "Segment"
_MANUAL_TYPE_COL = "Solution type"


class SnippetAnalyzer:
    """Re-run recommendation extraction on individual solution sentences."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[Dict[str, Any]] = None,
    ):
        cfg = config or {}
        self.extractor = RecommendationExtractor(
            llm_client=llm_client,
            config=cfg,
        )

    def analyze(
        self,
        source_path: Path,
        output_dir: Path,
        input_source: Literal["ai", "manual"] = "ai",
    ) -> pd.DataFrame:
        """Run snippet-level extraction and write output CSV.

        Parameters
        ----------
        source_path:
            - ``input_source="ai"``: path to ``recommendations.csv`` (must
              contain ``doc_id`` and ``source_text_raw`` columns).
            - ``input_source="manual"``: path to the manually coded segments
              Excel (must contain ``Document name`` and ``Segment`` columns).
        output_dir:
            Directory where the output CSV is written:
            ``recommendations_snippet_ai.csv`` or
            ``recommendations_snippet_manual.csv``.
        input_source:
            ``"ai"`` (default) or ``"manual"``.

        Returns
        -------
        pd.DataFrame
            The snippet-level extractions.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        inputs = self._load_inputs(Path(source_path), input_source)

        logger.info(
            f"Running snippet extraction ({input_source}) on {len(inputs)} "
            f"source sentences from {source_path}"
        )

        all_rows: List[Dict[str, Any]] = []
        for item in inputs:
            doc_id = item["doc_id"]
            source_text = item["source_text"]
            parent_id = item["parent_id"]
            parent_page = item["page"]

            page = PageText(
                page_num=parent_page,
                text=source_text,
                char_count=len(source_text),
                word_count=len(source_text.split()),
            )

            try:
                extractions: List[PolicyExtraction] = self.extractor.extract_recommendations(
                    [page], doc_id=doc_id, input_mode="snippet"
                )
            except Exception as exc:
                logger.warning(
                    f"[{doc_id}] Snippet extraction failed for {parent_id}: {exc}"
                )
                continue

            for ext in extractions:
                all_rows.append({
                    "doc_id": doc_id,
                    "parent_id": parent_id,
                    "input_source": input_source,
                    "rec_id": ext.rec_id,
                    "extraction_type": ext.extraction_type.value,
                    "confidence": ext.confidence,
                    "source_text_raw": ext.source_text_raw,
                    "page": ext.page,
                    "actor_text_raw": ext.actor_text_raw,
                    "actor_type_normalized": (
                        ext.actor_type_normalized.value
                        if ext.actor_type_normalized
                        else None
                    ),
                    "action_text_raw": ext.action_text_raw,
                    "instrument_type": (
                        ext.instrument_type.value if ext.instrument_type else None
                    ),
                    "policy_domain": ext.policy_domain,
                    "geographic_scope": (
                        ext.geographic_scope.value if ext.geographic_scope else None
                    ),
                    "strength": ext.strength.value if ext.strength else None,
                })

        result_df = pd.DataFrame(all_rows)
        out_name = (
            "recommendations_snippet_ai.csv"
            if input_source == "ai"
            else "recommendations_snippet_manual.csv"
        )
        out_path = output_dir / out_name
        save_dataframe(result_df, out_path)
        logger.info(
            f"Snippet extraction complete ({input_source}): "
            f"{len(all_rows)} extractions written to {out_path}"
        )
        return result_df

    # ── Input loading ─────────────────────────────────────────────────

    def _load_inputs(
        self,
        source_path: Path,
        input_source: Literal["ai", "manual"],
    ) -> List[Dict[str, Any]]:
        """Load and normalise input rows into a uniform list of dicts.

        Each dict has keys: ``doc_id``, ``source_text``, ``parent_id``, ``page``.
        """
        if input_source == "ai":
            return self._load_ai_inputs(source_path)
        return self._load_manual_inputs(source_path)

    def _load_ai_inputs(self, path: Path) -> List[Dict[str, Any]]:
        """Load from recommendations.csv.

        Accepts either ``source_text_raw`` (legacy) or ``source_text`` as the
        text column, whichever is present.
        """
        df = pd.read_csv(path)
        # Accept both column name conventions
        text_col = "source_text_raw" if "source_text_raw" in df.columns else "source_text"
        required = {"doc_id", text_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"recommendations.csv is missing required column(s): {missing}"
            )
        df = df[df[text_col].notna()].copy()
        df = df[df[text_col].astype(str).str.strip() != ""]

        result = []
        for _, row in df.iterrows():
            result.append({
                "doc_id": str(row["doc_id"]),
                "source_text": str(row[text_col]).strip(),
                "parent_id": str(row.get("rec_id", "")),
                "page": int(row["page"]) if pd.notna(row.get("page")) else 1,
            })
        return result

    def _load_manual_inputs(self, path: Path) -> List[Dict[str, Any]]:
        """Load from PBs solutions coded segments Excel (Segment column).

        Maps ``Document name`` → ``doc_id`` via ``create_document_id()``,
        which lowercases the string, replaces spaces with ``_``, and strips
        non-alphanumeric characters — identical to the mapping used in the
        main pipeline.
        """
        suffix = path.suffix.lower()
        df = pd.read_excel(path) if suffix in (".xlsx", ".xls") else pd.read_csv(path)

        for col in (_MANUAL_DOC_COL, _MANUAL_TEXT_COL):
            if col not in df.columns:
                raise ValueError(
                    f"Manual segments file is missing required column '{col}'. "
                    f"Available columns: {list(df.columns)}"
                )

        df = df[df[_MANUAL_TEXT_COL].notna()].copy()
        df = df[df[_MANUAL_TEXT_COL].astype(str).str.strip() != ""]
        df = df[df[_MANUAL_DOC_COL].notna()].copy()

        result = []
        for idx, row in df.iterrows():
            doc_name = str(row[_MANUAL_DOC_COL]).strip()
            doc_id = create_document_id(Path(doc_name))
            solution_type = str(row.get(_MANUAL_TYPE_COL, "") or "").strip()
            result.append({
                "doc_id": doc_id,
                "source_text": str(row[_MANUAL_TEXT_COL]).strip(),
                # Use solution_type as parent identifier where available
                "parent_id": f"{doc_id}_manual_{idx}" + (
                    f"_{solution_type}" if solution_type else ""
                ),
                "page": 1,  # page not available in manual segments
            })
        return result

