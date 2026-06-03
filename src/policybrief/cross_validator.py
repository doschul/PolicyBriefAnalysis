"""
Cross-validation: compare snippet-level AI and manual extraction outputs.

Primary comparison:
    ``recommendations_snippet_ai.csv`` vs ``recommendations_snippet_manual.csv``
    — document-level presence, extraction counts, extraction_type distributions.

Secondary comparison (optional):
    Both snippet outputs vs PB-level indicators from
    ``Policy Briefs corpus.csv.xlsx``.

Column schema of ``Policy Briefs corpus.csv.xlsx``:

    Related File Pdf Title | Policy brief title | Project Acronym | Date |
    Funding Program | Policy Brief URL | Focus Area | Thematic Category |
    (empty) | Solutions component present | Solutions component labeling (keyword) |
    Solution type (cf. analytical framework) | Notes |
    Chart | Table | Figure | Photo
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import re
from difflib import SequenceMatcher

import pandas as pd

from .utils import create_document_id, save_dataframe, save_json

logger = logging.getLogger(__name__)

# Default column names from the PB-level corpus (override via column_map)
_DEFAULT_CORPUS_COLUMN_MAP: Dict[str, str] = {
    "pdf_title": "Related File Pdf Title",
    "solutions_present": "Solutions component present",
    "solution_type": "Solution type (cf. analytical framework)",
    "has_chart": "Chart",
    "has_table": "Table",
    "has_figure": "Figure",
    "has_photo": "Photo",
}


class CrossValidator:
    """Compare snippet-level AI and manual extraction outputs.

    Parameters
    ----------
    corpus_path:
        Optional path to the PB-level indicators corpus
        (``Policy Briefs corpus.csv.xlsx``).  Required only for the secondary
        comparison.
    column_map:
        Override default column names for the PB-level corpus.
    """

    def __init__(
        self,
        corpus_path: Optional[Path] = None,
        column_map: Optional[Dict[str, str]] = None,
    ):
        self.corpus_path = Path(corpus_path) if corpus_path else None
        self.column_map = {**_DEFAULT_CORPUS_COLUMN_MAP, **(column_map or {})}

    # ── Public entry point ────────────────────────────────────────────

    def compare(
        self,
        ai_snippet_path: Path,
        manual_snippet_path: Path,
        output_dir: Path,
        ai_docs_path: Optional[Path] = None,
        sim_threshold: float = 0.7,
        full_recs_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run cross-validation and write comparison.csv, crosstab.csv + metrics.json.

        Parameters
        ----------
        ai_snippet_path:
            Path to ``recommendations_snippet_ai.csv`` (AI snippet pass output).
        manual_snippet_path:
            Path to the manual data file.  Accepted formats:

            - ``.xlsx`` / ``.xls``: spreadsheet with columns ``Document name``,
              ``Solution type``, ``Segment`` (e.g. ``PBs solutions coded
              segments.xlsx``).
            - ``.csv``: ``recommendations_snippet_manual.csv`` in the same schema
              as the AI snippet output.
        output_dir:
            Directory where ``comparison.csv``, ``crosstab.csv``, and
            ``metrics.json`` are written.
        ai_docs_path:
            Optional path to ``documents.csv`` from the pipeline.  When
            provided, ``has_visual_elements`` is included in the comparison
            (secondary comparison, requires ``corpus_path`` to be set).
        full_recs_path:
            Optional path to ``recommendations.csv`` (Pass 1 full-text output).
            When provided, a Pass 1 vs Pass 2 field-quality comparison is run
            and written to ``context_comparison.csv``; results are added to
            ``metrics.json`` under the ``context_comparison`` key.

        Returns
        -------
        dict
            Aggregate metrics dict with ``primary``, optional ``secondary``,
            and optional ``context_comparison`` keys.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ai_df = pd.read_csv(ai_snippet_path)

        # Load manual data — xlsx (Document name / Solution type / Segment)
        # or a CSV in the same schema as the AI snippet output
        manual_path = Path(manual_snippet_path)
        if manual_path.suffix.lower() in (".xlsx", ".xls"):
            manual_df = self._load_manual_xlsx(manual_path)
        else:
            manual_df = pd.read_csv(manual_path)

        ai_agg = self._aggregate_snippets(ai_df, prefix="ai")
        manual_agg = self._aggregate_snippets(manual_df, prefix="manual")

        # Outer join — keep all docs that appear in either pass
        merged = ai_agg.merge(manual_agg, on="doc_id", how="outer")
        for col in ("ai_count", "manual_count"):
            merged[col] = merged[col].fillna(0).astype(int)
        merged["ai_has_solutions"] = merged["ai_count"] > 0
        merged["manual_has_solutions"] = merged["manual_count"] > 0
        merged["presence_agree"] = (
            merged["ai_has_solutions"] == merged["manual_has_solutions"]
        )
        merged["type_agree"] = merged.apply(
            lambda r: (
                str(r["ai_dominant_type"]).lower() == str(r["manual_dominant_type"]).lower()
                if pd.notna(r.get("ai_dominant_type")) and pd.notna(r.get("manual_dominant_type"))
                else None
            ),
            axis=1,
        )

        # ── Content overlap ───────────────────────────────────────────────
        overlap_per_doc = self._compute_overlap(ai_df, manual_df, sim_threshold=sim_threshold)
        if not overlap_per_doc.empty:
            merged = merged.merge(overlap_per_doc, on="doc_id", how="left")

        # ── Crosstab: manual_count × ai_count ────────────────────────────
        crosstab = pd.crosstab(
            merged["manual_count"],
            merged["ai_count"],
            rownames=["manual_count"],
            colnames=["ai_count"],
        )
        crosstab.to_csv(output_dir / "crosstab.csv")
        logger.info(f"Crosstab written to {output_dir / 'crosstab.csv'}")

        # Optional secondary: compare against PB-level corpus indicators
        if self.corpus_path is not None:
            merged = self._attach_corpus(merged, ai_docs_path)

        save_dataframe(merged, output_dir / "comparison.csv")

        metrics = self._compute_metrics(merged, ai_df=ai_df, manual_df=manual_df, sim_threshold=sim_threshold)

        # Optional Pass 1 vs Pass 2 context comparison
        if full_recs_path is not None:
            ctx = self._compare_passes(
                full_recs_path=Path(full_recs_path),
                snippet_ai_path=Path(ai_snippet_path),
                output_dir=output_dir,
            )
            if ctx:
                metrics["context_comparison"] = ctx

        save_json(metrics, output_dir / "metrics.json")

        logger.info(
            f"Cross-validation complete: {len(merged)} documents compared. "
            f"Results written to {output_dir}"
        )
        return metrics

    # ── Internal helpers ──────────────────────────────────────────────

    def _aggregate_snippets(
        self, df: pd.DataFrame, prefix: str
    ) -> pd.DataFrame:
        """Aggregate per-document from a snippet CSV."""
        if df.empty or "doc_id" not in df.columns:
            return pd.DataFrame(columns=[
                "doc_id",
                f"{prefix}_count",
                f"{prefix}_dominant_type",
            ])
        agg = (
            df.groupby("doc_id")
            .agg(
                **{
                    f"{prefix}_count": ("rec_id", "count"),
                    f"{prefix}_dominant_type": (
                        "extraction_type",
                        lambda s: s.mode().iloc[0] if not s.mode().empty else None,
                    ),
                }
            )
            .reset_index()
        )
        return agg

    def _attach_corpus(
        self, merged: pd.DataFrame, ai_docs_path: Optional[Path]
    ) -> pd.DataFrame:
        """Attach PB-level corpus columns for secondary comparison."""
        suffix = self.corpus_path.suffix.lower()  # type: ignore[union-attr]
        corpus = (
            pd.read_excel(self.corpus_path)
            if suffix in (".xlsx", ".xls")
            else pd.read_csv(self.corpus_path)
        )
        pdf_col = self.column_map["pdf_title"]
        if pdf_col not in corpus.columns:
            logger.warning(
                f"Corpus column '{pdf_col}' not found; skipping secondary comparison. "
                f"Available: {list(corpus.columns)}"
            )
            return merged

        corpus["doc_id"] = corpus[pdf_col].apply(
            lambda x: create_document_id(Path(str(x))) if pd.notna(x) else None
        )
        corpus = corpus.dropna(subset=["doc_id"])

        keep_cols = ["doc_id", self.column_map["solutions_present"], self.column_map["solution_type"]]
        corpus_sub = corpus[[c for c in keep_cols if c in corpus.columns]].copy()
        corpus_sub = corpus_sub.rename(columns={
            self.column_map["solutions_present"]: "corpus_solutions_present",
            self.column_map["solution_type"]: "corpus_solution_type",
        })
        merged = merged.merge(corpus_sub, on="doc_id", how="left")

        # Compare both AI and manual against corpus presence
        for prefix in ("ai", "manual"):
            col = f"{prefix}_vs_corpus"
            if "corpus_solutions_present" in merged.columns:
                merged[col] = merged.apply(
                    lambda r, p=prefix: (
                        self._parse_bool(r.get("corpus_solutions_present"))
                        == r.get(f"{p}_has_solutions")
                        if self._parse_bool(r.get("corpus_solutions_present")) is not None
                        else None
                    ),
                    axis=1,
                )

        # Visual elements (requires ai_docs_path)
        if ai_docs_path is not None:
            ai_docs = pd.read_csv(ai_docs_path)
            if "has_visual_elements" in ai_docs.columns:
                merged = merged.merge(
                    ai_docs[["doc_id", "has_visual_elements"]], on="doc_id", how="left"
                )
                # Join visual columns from corpus
                visual_corpus = corpus[["doc_id"] + [
                    self.column_map[k]
                    for k in ("has_chart", "has_table", "has_figure", "has_photo")
                    if self.column_map[k] in corpus.columns
                ]].copy()
                merged = merged.merge(visual_corpus, on="doc_id", how="left")

        return merged

    def _parse_bool(self, value) -> Optional[bool]:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        s = str(value).strip().lower()
        if s in ("yes", "true", "1", "present", "x", "y"):
            return True
        if s in ("no", "false", "0", "absent", ""):
            return False
        return None

    def _load_manual_xlsx(self, path: Path) -> pd.DataFrame:
        """Load manual solution segments from xlsx.

        Expected columns (case-insensitive): ``Document name``, ``Solution type``,
        ``Segment``.  Returns a DataFrame with ``doc_id``, ``rec_id``,
        ``extraction_type``, ``source_text`` — compatible with
        ``_aggregate_snippets()``.
        """
        raw = pd.read_excel(path)
        col_lower = {c.lower().strip(): c for c in raw.columns}

        def _find(*candidates: str) -> Optional[str]:
            for c in candidates:
                if c.lower() in col_lower:
                    return col_lower[c.lower()]
            return None

        doc_col = _find("document name", "document_name")
        seg_col = _find("segment")
        type_col = _find("solution type", "solution_type")

        if doc_col is None or seg_col is None:
            raise ValueError(
                f"Manual xlsx must contain 'Document name' and 'Segment' columns. "
                f"Found: {list(raw.columns)}"
            )

        out = pd.DataFrame()
        out["doc_id"] = raw[doc_col].apply(
            lambda x: create_document_id(Path(str(x).strip())) if pd.notna(x) else None
        )
        out["source_text"] = raw[seg_col]
        out["extraction_type"] = raw[type_col] if type_col else "solution"
        out = out.dropna(subset=["doc_id", "source_text"]).reset_index(drop=True)
        out["rec_id"] = [f"{row.doc_id}_m{i}" for i, row in out.iterrows()]

        logger.info(
            f"Loaded {len(out)} manual segments from {path.name} "
            f"({out['doc_id'].nunique()} documents)"
        )
        return out

    def _compute_overlap(
        self, ai_df: pd.DataFrame, manual_df: pd.DataFrame, sim_threshold: float = 0.7
    ) -> pd.DataFrame:
        """Per-document content overlap: fraction of manual segments found in AI output.

        Two match modes are computed for every manual segment:

        **Exact** — normalised substring containment (lowercase + whitespace collapse).
        Checks whether the manual segment is contained in any AI text *or* any AI text
        is contained in the manual segment.  No tolerance for character-level differences.

        **Fuzzy** — ``difflib.SequenceMatcher`` ratio against every AI text in the same
        document; a segment is fuzzy-matched when ``max(ratio) >= sim_threshold``.
        The ratio is ``2 * matching_chars / total_chars`` so it is sensitive to both
        length differences and character-level deviations.

        Returns a DataFrame with columns:
            ``doc_id``, ``manual_total``,
            ``manual_matched``, ``overlap_rate``,
            ``manual_fuzzy_matched``, ``fuzzy_overlap_rate``
        """
        if manual_df.empty or ai_df.empty:
            return pd.DataFrame(columns=[
                "doc_id", "manual_total",
                "manual_matched", "overlap_rate",
                "manual_fuzzy_matched", "fuzzy_overlap_rate",
            ])

        def _norm(text) -> str:
            if not isinstance(text, str):
                return ""
            return re.sub(r"\s+", " ", text).strip().lower()

        ai_text_col = "source_text_raw" if "source_text_raw" in ai_df.columns else "source_text"
        manual_text_col = "source_text" if "source_text" in manual_df.columns else "source_text_raw"

        # Build per-doc list of normalised AI texts
        ai_by_doc: Dict[str, list] = {}
        for doc_id, grp in ai_df.groupby("doc_id"):
            ai_by_doc[str(doc_id)] = [
                _norm(t) for t in grp[ai_text_col] if pd.notna(t)
            ]

        results = []
        for _, row in manual_df.iterrows():
            doc_id = str(row["doc_id"])
            seg = _norm(row.get(manual_text_col, ""))
            ai_texts = ai_by_doc.get(doc_id, [])

            # Exact: substring containment in either direction
            exact = bool(seg) and any(
                (seg in ai) or (ai in seg)
                for ai in ai_texts
                if ai
            )

            # Fuzzy: best SequenceMatcher ratio across all AI texts
            fuzzy = False
            if bool(seg) and ai_texts:
                best = max(
                    (SequenceMatcher(None, seg, ai).ratio() for ai in ai_texts if ai),
                    default=0.0,
                )
                fuzzy = best >= sim_threshold

            results.append({"doc_id": doc_id, "exact": exact, "fuzzy": fuzzy})

        seg_df = pd.DataFrame(results)
        overlap = (
            seg_df.groupby("doc_id")
            .agg(
                manual_total=("exact", "count"),
                manual_matched=("exact", "sum"),
                manual_fuzzy_matched=("fuzzy", "sum"),
            )
            .reset_index()
        )
        overlap["overlap_rate"] = overlap["manual_matched"] / overlap["manual_total"]
        overlap["fuzzy_overlap_rate"] = overlap["manual_fuzzy_matched"] / overlap["manual_total"]
        return overlap

    def _compute_metrics(
        self,
        df: pd.DataFrame,
        ai_df: Optional[pd.DataFrame] = None,
        manual_df: Optional[pd.DataFrame] = None,
        sim_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        total = len(df)
        presence_valid = df[df["presence_agree"].notna()]
        presence_agree_count = int(presence_valid["presence_agree"].sum()) if not presence_valid.empty else 0

        type_valid = df[df["type_agree"].notna()]
        type_agree_count = int(type_valid["type_agree"].sum()) if not type_valid.empty else 0

        metrics: Dict[str, Any] = {
            "total_documents": total,
            "primary": {
                "overview": {
                    "ai_total_solutions": int(len(ai_df)) if ai_df is not None else int(df["ai_count"].sum()),
                    "manual_total_solutions": int(len(manual_df)) if manual_df is not None else int(df["manual_count"].sum()),
                    "ai_solution_docs": int(df["ai_has_solutions"].sum()),
                    "manual_solution_docs": int(df["manual_has_solutions"].sum()),
                },
                "presence_agreement": {
                    "agreed": presence_agree_count,
                    "total": len(presence_valid),
                    "rate": presence_agree_count / len(presence_valid) if len(presence_valid) > 0 else None,
                },
                "type_agreement": {
                    "agreed": type_agree_count,
                    "total": len(type_valid),
                    "rate": type_agree_count / len(type_valid) if len(type_valid) > 0 else None,
                },
            },
        }

        # Content overlap aggregate (populated when _compute_overlap ran)
        if "manual_total" in df.columns and "manual_matched" in df.columns:
            total_manual = int(df["manual_total"].sum())
            total_matched = int(df["manual_matched"].sum())
            metrics["primary"]["overlap_exact"] = {
                "manual_total": total_manual,
                "manual_matched": total_matched,
                "overlap_rate": total_matched / total_manual if total_manual > 0 else None,
            }
            if "manual_fuzzy_matched" in df.columns:
                total_fuzzy = int(df["manual_fuzzy_matched"].sum())
                metrics["primary"]["overlap_fuzzy"] = {
                    "manual_total": total_manual,
                    "manual_matched": total_fuzzy,
                    "overlap_rate": total_fuzzy / total_manual if total_manual > 0 else None,
                    "sim_threshold": sim_threshold,
                }

        # Secondary metrics (only when corpus columns present)
        secondary: Dict[str, Any] = {}
        for prefix in ("ai", "manual"):
            col = f"{prefix}_vs_corpus"
            if col in df.columns:
                valid = df[df[col].notna()]
                agree = int(valid[col].sum()) if not valid.empty else 0
                secondary[f"{prefix}_vs_corpus"] = {
                    "agreed": agree,
                    "total": len(valid),
                    "rate": agree / len(valid) if len(valid) > 0 else None,
                }
        if secondary:
            metrics["secondary"] = secondary

        return metrics

    def _compare_passes(
        self,
        full_recs_path: Path,
        snippet_ai_path: Path,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Compare Pass 1 (full-text) vs Pass 2 (AI-snippet) field populations.

        Joins Pass 1 ``recommendations.csv`` to Pass 2
        ``recommendations_snippet_ai.csv`` on ``rec_id`` (P1) = ``parent_id``
        (P2).  Measures:

        - Fill rate per field in each pass (fraction of non-null, non-empty values)
        - Agreement rate for shared fields (restricted to pairs where both passes
          produced a value)
        - Median confidence delta (Pass 2 − Pass 1)

        Writes ``context_comparison.csv`` (one row per matched extraction pair)
        to *output_dir*.
        """
        p1 = pd.read_csv(full_recs_path)
        p2 = pd.read_csv(snippet_ai_path)

        if "rec_id" not in p1.columns or "parent_id" not in p2.columns:
            logger.warning(
                "_compare_passes: missing join columns (rec_id in Pass 1, "
                "parent_id in Pass 2); skipping context comparison"
            )
            return {}

        # Align actor-type column name before merge
        p2 = p2.copy()
        if "actor_type_normalized" in p2.columns:
            p2 = p2.rename(columns={"actor_type_normalized": "actor_type"})

        merged = p1.merge(
            p2,
            left_on="rec_id",
            right_on="parent_id",
            suffixes=("_p1", "_p2"),
        )

        if merged.empty:
            logger.warning(
                "_compare_passes: inner join produced no rows; "
                "check that rec_id / parent_id values align"
            )
            return {}

        doc_id_col = "doc_id_p1" if "doc_id_p1" in merged.columns else "doc_id"
        n_pairs = len(merged)
        n_docs = merged[doc_id_col].nunique()
        logger.info(f"_compare_passes: {n_pairs} matched pairs from {n_docs} documents")

        # ── Field definitions ──────────────────────────────────────────────
        # (label, p1_col_after_merge, p2_col_after_merge)
        SHARED_FIELDS = [
            ("extraction_type", "extraction_type_p1", "extraction_type_p2"),
            ("actor_type",      "actor_type_p1",      "actor_type_p2"),
            ("instrument_type", "instrument_type_p1", "instrument_type_p2"),
            ("geographic_scope","geographic_scope_p1","geographic_scope_p2"),
            ("strength",        "strength_p1",        "strength_p2"),
            ("policy_domain",   "policy_domain_p1",   "policy_domain_p2"),
        ]
        # Present only in Pass 1 — measure fill rate only
        P1_ONLY_FIELDS = ["target_raw", "timeframe"]

        def _fill_rate(col: str) -> Optional[float]:
            if col not in merged.columns:
                return None
            s = merged[col].dropna()
            s = s[s.astype(str).str.strip() != ""]
            return round(len(s) / len(merged), 4) if len(merged) > 0 else 0.0

        # ── Fill rates ─────────────────────────────────────────────────────
        fill_rates: Dict[str, Any] = {}
        for label, p1_col, p2_col in SHARED_FIELDS:
            fill_rates[label] = {"pass1": _fill_rate(p1_col), "pass2": _fill_rate(p2_col)}
        for col in P1_ONLY_FIELDS:
            fill_rates[col] = {"pass1": _fill_rate(col), "pass2": None}

        # ── Agreement rates ────────────────────────────────────────────────
        agreement_rates: Dict[str, Any] = {}
        for label, p1_col, p2_col in SHARED_FIELDS:
            if p1_col not in merged.columns or p2_col not in merged.columns:
                continue
            both = (
                merged[p1_col].notna()
                & merged[p2_col].notna()
                & (merged[p1_col].astype(str).str.strip() != "")
                & (merged[p2_col].astype(str).str.strip() != "")
            )
            sub = merged[both]
            if len(sub) == 0:
                agreement_rates[label] = {"agreed": 0, "total": 0, "rate": None}
                continue
            agree = (
                sub[p1_col].astype(str).str.lower().str.strip()
                == sub[p2_col].astype(str).str.lower().str.strip()
            )
            agreement_rates[label] = {
                "agreed": int(agree.sum()),
                "total": len(sub),
                "rate": round(float(agree.mean()), 4),
            }

        # ── Confidence delta ───────────────────────────────────────────────
        conf_delta: Optional[float] = None
        if "confidence_p1" in merged.columns and "confidence_p2" in merged.columns:
            valid_conf = merged[["confidence_p1", "confidence_p2"]].dropna()
            if not valid_conf.empty:
                conf_delta = round(
                    float((valid_conf["confidence_p2"] - valid_conf["confidence_p1"]).median()), 4
                )

        # ── Per-row output CSV ─────────────────────────────────────────────
        out: Dict[str, Any] = {}
        out["doc_id"] = merged[doc_id_col]
        rec_p1 = "rec_id_p1" if "rec_id_p1" in merged.columns else "rec_id"
        rec_p2 = "rec_id_p2" if "rec_id_p2" in merged.columns else None
        out["rec_id_p1"] = merged[rec_p1]
        if rec_p2 and rec_p2 in merged.columns:
            out["rec_id_p2"] = merged[rec_p2]

        for label, p1_col, p2_col in SHARED_FIELDS:
            if p1_col in merged.columns:
                out[f"{label}_p1"] = merged[p1_col]
            if p2_col in merged.columns:
                out[f"{label}_p2"] = merged[p2_col]
            if p1_col in merged.columns and p2_col in merged.columns:
                both = merged[p1_col].notna() & merged[p2_col].notna()
                out[f"{label}_agree"] = both & (
                    merged[p1_col].astype(str).str.lower().str.strip()
                    == merged[p2_col].astype(str).str.lower().str.strip()
                )

        for col in P1_ONLY_FIELDS:
            if col in merged.columns:
                out[f"{col}_p1"] = merged[col]

        if "confidence_p1" in merged.columns:
            out["confidence_p1"] = merged["confidence_p1"]
        if "confidence_p2" in merged.columns:
            out["confidence_p2"] = merged["confidence_p2"]
        if "confidence_p1" in merged.columns and "confidence_p2" in merged.columns:
            out["confidence_delta"] = merged["confidence_p2"] - merged["confidence_p1"]

        out_df = pd.DataFrame(out)
        save_dataframe(out_df, output_dir / "context_comparison.csv")
        logger.info(
            f"Context comparison written: {len(out_df)} rows -> "
            f"{output_dir / 'context_comparison.csv'}"
        )

        return {
            "matched_pairs": n_pairs,
            "documents": n_docs,
            "fill_rates": fill_rates,
            "agreement_rates": agreement_rates,
            "confidence_delta_median": conf_delta,
        }

