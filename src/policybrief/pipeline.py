"""
Policy Brief Analysis Pipeline — broad-content, multi-pass LLM orchestrator.

All semantic extraction (front-matter, structural core, frames,
recommendations) is handled by broad-content LLM passes. Deterministic
rails: PDF extraction, metrics, reference detection, evidence verification,
normalization, output generation.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .frame_detector import FrameDetector
from .llm_client import LLMClient
from .metrics_calculator import MetricsCalculator
from .models import (
    DocumentFrontMatter,
    DocumentMetrics,
    FrameAssessment,
    PageText,
    PDFMetadata,
    PerDocumentExtraction,
    PolicyExtraction,
    ProcessingStatus,
    StructuralCoreResult,
)
from .pdf_extractor import PDFExtractor
from .recommendation_extractor import DocumentContent, RecommendationExtractor
from .utils import (
    clean_text_for_csv,
    create_document_id,
    ensure_output_directories,
    get_env_var,
    load_yaml_config,
    save_dataframe,
    save_json,
)

logger = logging.getLogger(__name__)


# ── LLM prompts ──────────────────────────────────────────────────────────

FRONT_MATTER_PROMPT = """Extract front-matter metadata from this policy document.

Return JSON with exactly these fields:
- title: the document title (null if not found)
- authors: list of author names (empty list if none)
- affiliations: list of institutional affiliations (empty list if none)
- emails: list of email addresses found (empty list if none)
- urls: list of URLs found (empty list if none)
- funding_statements: list of funding acknowledgements (empty list if none)
- linked_studies: list of referenced study names/IDs (empty list if none)

Rules:
- Extract information from the actual document text, not from PDF metadata.
- Only extract information explicitly stated in the text.
- Do not infer or guess. If unsure, leave as null or empty list.
- The title should be the document's main title, not a section heading.
- Authors should be individual person names, not organizations.
- Search primarily in likely front-matter and end-matter locations such as title page, author blocks, acknowledgements, "about", "imprint", "contact", or closing sections, but use the full provided text if needed.
- funding_statements should include only explicit acknowledgements of funding, sponsorship, grants, or supporting institutions. Do NOT include general mentions of organizations unless they are clearly stated as funders.
- linked_studies should include only explicit references to a fuller underlying study, companion report, working paper, journal article, or technical report that this document points to for more detail.
- Do NOT include ordinary bibliography entries, generic citations, or literature references as linked_studies.
- Emails and URLs must be explicitly present in the text; do not construct or infer them."""

STRUCTURAL_CORE_PROMPT = """Analyze the structural components of this policy document.

Assess whether the document contains:

1. **Problem identification**: Does the document explicitly identify and frame a policy problem?
   - "present": clear, explicit problem framing with evidence or description
   - "weak": problem is implied but not explicitly stated
   - "absent": no problem identification
   - A problem is "present" only if the document clearly describes a real-world issue, challenge, or policy-relevant situation that requires intervention. General background context or topic description alone is not sufficient.

2. **Solutions**: How many distinct policy solutions or options are proposed?
   - Count only concrete, actionable solutions (not vague aspirations)
   - A solution must represent a distinct course of action, intervention, or policy approach proposed by the document itself. Do NOT count repeated mentions of the same idea.
   - Are solutions explicitly linked to the identified problem?
   - Solutions should ideally address the identified problem. If solutions are presented but not clearly connected to a problem, treat them as weaker.

3. **Implementation considerations**: Does the document discuss how to implement its proposals?
   - "present": concrete implementation steps, timelines, or actors
   - "weak": brief mention without detail
   - "absent": no implementation discussion
   - Implementation considerations include any discussion of how policies would be carried out in practice, such as: who is responsible (actors or institutions), how actions would be executed (procedures or steps), when actions occur (timelines or sequencing), feasibility, barriers, facilitators, or required resources, monitoring, enforcement, or evaluation.

4. **Narrative hook**: Does the document use a compelling opening device?
   - Examples: case study, statistic, anecdote, provocative question, vivid example
   - A narrative hook should function as an engaging entry point (e.g. concrete example, striking statistic, or vivid scenario), not just general introductory text.

5. **Explicit heading labels**: For each of the following structural components, assess whether they are clearly labelled with a section heading or subheading in the document. A component is "explicitly labelled" only if there is a visible heading (e.g. "The Problem", "Challenges", "Recommendations", "Policy Options", "Implementation", "Next Steps") that clearly signals the section's purpose. Implicit structure without labelled headings does not count.
   - problem_explicitly_labelled: Is the problem/background/motivation section labelled with a heading?
   - solutions_explicitly_labelled: Are solutions or recommendations labelled with a heading?
   - implementation_explicitly_labelled: Are implementation considerations labelled with a heading?

6. **Procedural clarity**: Does the document provide concrete guidance on HOW actions should be carried out?
   - "present": the document specifies named steps, procedures, sequencing, timelines, operational instructions, or clearly describes who does what and how
   - "weak": some procedural language exists but is vague, incomplete, or only aspirational
   - "absent": the document recommends actions but provides no concrete guidance on execution
   - Procedural clarity is distinct from merely mentioning implementation. A document can mention implementation considerations (e.g. "implementation will require resources") without providing concrete procedural guidance.
   - General aspirations like "governments should act" do NOT count as procedural clarity.

Return JSON with:
- problem_status: "present"/"absent"/"weak"
- problem_summary: brief description of the problem (null if absent)
- solutions_count: integer count of distinct solutions
- solutions_explicit: boolean, are solutions explicitly proposed?
- implementation_status: "present"/"absent"/"weak"
- implementation_count: number of implementation considerations
- narrative_hook_present: boolean
- narrative_hook_type: type of hook used (null if none)
- problem_explicitly_labelled: boolean
- solutions_explicitly_labelled: boolean
- implementation_explicitly_labelled: boolean
- procedural_clarity_status: "present"/"absent"/"weak"

Be conservative. Only mark as "present" when evidence is clear.
Look at the document broadly and consider how elements function in the document, not just whether certain words appear.
Do NOT rely on section headings alone for problem_status, solutions, or implementation_status; a document may have implicit structure without labeled sections.
Do NOT treat general aspirations, high-level goals, or problem descriptions as solutions."""


class PolicyBriefPipeline:
    """Main pipeline: PDF → text → metrics → LLM analysis → output tables."""

    def __init__(
        self,
        config_dir: Path,
        output_dir: Path,
        max_workers: int = 4,
        force_reprocess: bool = False,
    ):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.force_reprocess = force_reprocess

        # Load configs
        self.config = load_yaml_config(self.config_dir / "config.yaml")
        self.frames = load_yaml_config(self.config_dir / "frames.yaml").get("frames", [])

        # Module switches
        modules = self.config.get("modules", {})
        self.enable_front_matter = modules.get("front_matter", True)
        self.enable_structural_core = modules.get("structural_core", True)
        self.enable_frames = modules.get("frames", True)
        self.enable_recommendations = modules.get("recommendations", True)

        # Initialize components
        pdf_cfg = self.config.get("pdf", {})
        self.pdf_extractor = PDFExtractor(
            method=pdf_cfg.get("extract_method", "pymupdf"),
            preserve_layout=pdf_cfg.get("preserve_layout", True),
            max_pages=pdf_cfg.get("max_pages", 0),
            max_file_size_mb=pdf_cfg.get("max_file_size_mb", 50),
        )
        self.metrics_calculator = MetricsCalculator()

        # LLM client (optional, created on first use)
        self._llm_client: Optional[LLMClient] = None
        self._frame_detector: Optional[FrameDetector] = None
        self._recommendation_extractor: Optional[RecommendationExtractor] = None

        # Content-hash cache
        self._hash_cache: Dict[str, str] = {}

        ensure_output_directories(self.output_dir)

    # ── Lazy LLM initialization ───────────────────────────────────────

    @property
    def llm_client(self) -> LLMClient:
        if self._llm_client is None:
            api_key = get_env_var("OPENAI_API_KEY", required=True)
            oai_cfg = self.config.get("openai", {})
            self._llm_client = LLMClient(
                api_key=api_key,
                model=oai_cfg.get("model", "gpt-4o-mini"),
                temperature=oai_cfg.get("temperature", 0.1),
                max_tokens=oai_cfg.get("max_tokens", 4000),
                timeout=oai_cfg.get("timeout", 60),
                max_retries=oai_cfg.get("max_retries", 5),
                retry_delay=oai_cfg.get("retry_delay", 2.0),
            )
        return self._llm_client

    @property
    def frame_detector(self) -> FrameDetector:
        if self._frame_detector is None:
            frames_cfg = self.config.get("frames", {})
            self._frame_detector = FrameDetector(
                llm_client=self.llm_client,
                frames_config=self.frames,
                min_confidence=frames_cfg.get("min_confidence", 0.7),
                max_spans_per_frame=frames_cfg.get("max_spans_per_frame", 5),
                context_window=frames_cfg.get("context_window", 500),
                min_evidence_quotes=frames_cfg.get("min_evidence_quotes", 1),
                max_evidence_quotes=frames_cfg.get("max_evidence_quotes", 3),
                max_chars_per_chunk=frames_cfg.get("max_chars_per_chunk", 100000),
            )
        return self._frame_detector

    @property
    def recommendation_extractor(self) -> RecommendationExtractor:
        if self._recommendation_extractor is None:
            rec_cfg = self.config.get("recommendations", {})
            self._recommendation_extractor = RecommendationExtractor(
                llm_client=self.llm_client,
                config=rec_cfg,
            )
        return self._recommendation_extractor

    # ── Main entry point ──────────────────────────────────────────────

    def process_documents(
        self,
        file_paths: List[Path],
    ) -> Dict[str, List]:
        """Process multiple PDFs and write output tables."""
        results: Dict[str, List] = {
            "processed": [],
            "skipped": [],
            "errors": [],
        }

        for fp in file_paths:
            try:
                doc_id = create_document_id(fp)
                file_hash = self.pdf_extractor.compute_file_hash(fp)

                if not self.force_reprocess and self._is_cached(doc_id, file_hash):
                    results["skipped"].append(doc_id)
                    continue

                extraction = self._process_single(fp, doc_id, file_hash)
                results["processed"].append(extraction)

                # Write audit file
                audit_path = self.output_dir / "audit" / f"{doc_id}.json"
                save_json(extraction.model_dump(), audit_path)

            except Exception as exc:
                logger.error(f"Failed to process {fp}: {exc}")
                results["errors"].append(f"{fp}: {exc}")

        # Generate output tables from all processed results
        if results["processed"]:
            self._generate_output_files(results["processed"])

        return results

    def _is_cached(self, doc_id: str, file_hash: str) -> bool:
        cached = self._hash_cache.get(doc_id)
        if cached == file_hash:
            logger.info(f"[{doc_id}] Unchanged (hash match), skipping")
            return True
        return False

    # ── Single-document processing ────────────────────────────────────

    def _process_single(
        self,
        file_path: Path,
        doc_id: str,
        file_hash: str,
    ) -> PerDocumentExtraction:
        start = time.time()
        warnings: List[str] = []

        # 1. PDF extraction
        pages, metadata = self.pdf_extractor.extract(file_path)
        likely_scanned, quality = self.pdf_extractor.detect_scanned(pages)
        if likely_scanned:
            warnings.append("Document appears to be scanned / image-only")

        # 2. Metrics
        metrics = self.metrics_calculator.calculate_metrics(pages)

        # 3. Front matter (LLM)
        front_matter: Optional[DocumentFrontMatter] = None
        if self.enable_front_matter and not likely_scanned:
            try:
                front_matter = self._extract_front_matter(pages)
            except Exception as exc:
                logger.warning(f"[{doc_id}] Front-matter extraction failed: {exc}")
                warnings.append(f"Front-matter failed: {exc}")

        # 4. Structural core (LLM)
        structural_core: Optional[StructuralCoreResult] = None
        if self.enable_structural_core and not likely_scanned:
            try:
                structural_core = self._extract_structural_core(pages)
            except Exception as exc:
                logger.warning(f"[{doc_id}] Structural-core extraction failed: {exc}")
                warnings.append(f"Structural-core failed: {exc}")

        # 5. Frame detection
        frame_assessments: List[FrameAssessment] = []
        policy_mix = False
        if self.enable_frames and not likely_scanned:
            try:
                frame_assessments = self.frame_detector.detect_frames(pages)
                policy_mix = self.frame_detector.detect_policy_mix(frame_assessments)
            except Exception as exc:
                logger.warning(f"[{doc_id}] Frame detection failed: {exc}")
                warnings.append(f"Frame detection failed: {exc}")

        # 6. Recommendation extraction
        policy_extractions: List[PolicyExtraction] = []
        if self.enable_recommendations and not likely_scanned:
            try:
                policy_extractions = self.recommendation_extractor.extract_recommendations(
                    pages, doc_id
                )
            except Exception as exc:
                logger.warning(f"[{doc_id}] Recommendation extraction failed: {exc}")
                warnings.append(f"Recommendation extraction failed: {exc}")

        duration = time.time() - start

        status = ProcessingStatus(
            doc_id=doc_id,
            file_path=str(file_path),
            file_hash=file_hash,
            file_size_bytes=os.path.getsize(file_path),
            processing_timestamp=datetime.now(),
            processing_duration_seconds=round(duration, 2),
            parser_used=self.pdf_extractor.method,
            likely_scanned=likely_scanned,
            text_extraction_quality=quality,
            pages_processed=len(pages),
            frames_processed=len(frame_assessments),
            recommendations_extracted=len(policy_extractions),
            warnings=warnings,
        )

        self._hash_cache[doc_id] = file_hash

        return PerDocumentExtraction(
            doc_id=doc_id,
            pages=pages,
            metadata=metadata,
            front_matter=front_matter,
            metrics=metrics,
            structural_core=structural_core,
            frame_assessments=frame_assessments,
            policy_mix_present=policy_mix,
            policy_extractions=policy_extractions,
            processing_status=status,
        )

    # ── LLM-based front-matter extraction ─────────────────────────────

    def _extract_front_matter(self, pages: List[PageText]) -> DocumentFrontMatter:
        """Extract front matter from broad document content via LLM."""
        # Use first 3 pages and last page (covers most formats)
        sample_pages = list(pages[:3])
        if len(pages) > 3:
            sample_pages.append(pages[-1])
        content = DocumentContent(sample_pages)
        text = content.full_text_with_markers()
        # Truncate to avoid token limits
        if len(text) > 8000:
            text = text[:8000]

        messages = [
            {"role": "system", "content": FRONT_MATTER_PROMPT},
            {"role": "user", "content": text},
        ]
        return self.llm_client.structured_completion(messages, DocumentFrontMatter)

    # ── LLM-based structural core extraction ──────────────────────────

    def _extract_structural_core(self, pages: List[PageText]) -> StructuralCoreResult:
        """Analyse structural core via broad document content + LLM."""
        content = DocumentContent(pages)
        # For short docs, send everything; for longer, sample broadly
        if content.total_chars <= 12000:
            text = content.full_text_with_markers()
        else:
            # Sample: first 5 pages, middle pages, last 3 pages
            n = len(pages)
            indices = set(range(min(5, n)))
            if n > 10:
                mid = n // 2
                indices.update(range(max(0, mid - 1), min(n, mid + 2)))
            if n > 5:
                indices.update(range(max(0, n - 3), n))
            sample = [pages[i] for i in sorted(indices)]
            sample_content = DocumentContent(sample)
            text = sample_content.full_text_with_markers()
        if len(text) > 15000:
            text = text[:15000]

        messages = [
            {"role": "system", "content": STRUCTURAL_CORE_PROMPT},
            {"role": "user", "content": text},
        ]
        return self.llm_client.structured_completion(messages, StructuralCoreResult)

    # ── Output generation ─────────────────────────────────────────────

    def _generate_output_files(self, results: List[PerDocumentExtraction]) -> None:
        """Write CSV output tables from extraction results."""
        self._write_documents_csv(results)
        self._write_frames_csv(results)
        self._write_recommendations_csv(results)
        self._write_structural_core_csv(results)

    def _write_documents_csv(self, results: List[PerDocumentExtraction]) -> None:
        rows: List[Dict[str, Any]] = []
        for r in results:
            row: Dict[str, Any] = {"doc_id": r.doc_id}
            # Metadata
            row["title"] = r.metadata.title
            row["author"] = r.metadata.author
            # Front matter
            if r.front_matter:
                row["fm_title"] = r.front_matter.title
                row["fm_authors"] = "; ".join(r.front_matter.authors)
                row["fm_affiliations"] = "; ".join(r.front_matter.affiliations)
                row["fm_emails"] = "; ".join(r.front_matter.emails)
                row["fm_urls"] = "; ".join(r.front_matter.urls)
                row["funding_statement_present"] = len(r.front_matter.funding_statements) > 0
                row["funding_statements_raw"] = "; ".join(r.front_matter.funding_statements) if r.front_matter.funding_statements else None
            else:
                row["funding_statement_present"] = None
                row["funding_statements_raw"] = None
            # Metrics
            for k, v in r.metrics.model_dump().items():
                row[k] = v
            # Status
            row["parser_used"] = r.processing_status.parser_used
            row["likely_scanned"] = r.processing_status.likely_scanned
            row["text_extraction_quality"] = r.processing_status.text_extraction_quality
            row["processing_duration_seconds"] = r.processing_status.processing_duration_seconds
            row["frames_processed"] = r.processing_status.frames_processed
            row["recommendations_extracted"] = r.processing_status.recommendations_extracted
            row["policy_mix_present"] = r.policy_mix_present
            row["warnings"] = "; ".join(r.processing_status.warnings)
            rows.append(row)
        save_dataframe(pd.DataFrame(rows), self.output_dir / "documents.csv")

    def _write_frames_csv(self, results: List[PerDocumentExtraction]) -> None:
        rows: List[Dict[str, Any]] = []
        for r in results:
            for fa in r.frame_assessments:
                row = {
                    "doc_id": r.doc_id,
                    "frame_id": fa.frame_id,
                    "frame_label": fa.frame_label,
                    "decision": fa.decision.value,
                    "confidence": fa.confidence,
                    "evidence_count": len(fa.evidence),
                    "rationale": clean_text_for_csv(fa.rationale),
                }
                if fa.evidence:
                    row["evidence_1_page"] = fa.evidence[0].page
                    row["evidence_1_quote"] = clean_text_for_csv(
                        fa.evidence[0].quote, 500
                    )
                rows.append(row)
        save_dataframe(pd.DataFrame(rows), self.output_dir / "frames.csv")

    def _write_recommendations_csv(self, results: List[PerDocumentExtraction]) -> None:
        rows: List[Dict[str, Any]] = []
        for r in results:
            for pe in r.policy_extractions:
                row = {
                    "doc_id": r.doc_id,
                    "rec_id": pe.rec_id,
                    "extraction_type": pe.extraction_type.value,
                    "confidence": pe.confidence,
                    "source_text": clean_text_for_csv(pe.source_text_raw, 500),
                    "page": pe.page,
                    "actor_raw": pe.actor_text_raw,
                    "actor_type": pe.actor_type_normalized.value if pe.actor_type_normalized else None,
                    "action_raw": clean_text_for_csv(pe.action_text_raw or "", 300),
                    "target_raw": clean_text_for_csv(pe.target_text_raw or "", 300),
                    "instrument_type": pe.instrument_type.value if pe.instrument_type else None,
                    "strength": pe.strength.value if pe.strength else None,
                    "geographic_scope": pe.geographic_scope.value if pe.geographic_scope else None,
                    "timeframe": pe.timeframe.value if pe.timeframe else None,
                    "policy_domain": pe.policy_domain,
                }
                rows.append(row)
        save_dataframe(pd.DataFrame(rows), self.output_dir / "recommendations.csv")

    def _write_structural_core_csv(self, results: List[PerDocumentExtraction]) -> None:
        rows: List[Dict[str, Any]] = []
        for r in results:
            if r.structural_core:
                row = {"doc_id": r.doc_id}
                row.update(r.structural_core.model_dump())
                rows.append(row)
        if rows:
            save_dataframe(pd.DataFrame(rows), self.output_dir / "structural_core.csv")

    # ── Summary ───────────────────────────────────────────────────────

    def compute_extraction_summary(
        self,
        results: List[PerDocumentExtraction],
    ) -> Dict[str, Any]:
        """Compute a summary across processed documents."""
        total_frames_present = sum(
            1
            for r in results
            for fa in r.frame_assessments
            if fa.decision.value == "present"
        )
        total_extractions = sum(len(r.policy_extractions) for r in results)
        total_pages = sum(r.metrics.page_count for r in results)
        warnings_list: List[str] = []
        for r in results:
            warnings_list.extend(r.processing_status.warnings)

        return {
            "documents_processed": len(results),
            "total_pages": total_pages,
            "total_frames_present": total_frames_present,
            "total_extractions": total_extractions,
            "policy_mix_documents": sum(1 for r in results if r.policy_mix_present),
            "warnings": warnings_list,
        }
