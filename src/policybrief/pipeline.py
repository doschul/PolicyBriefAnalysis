"""
Main pipeline orchestrator for policy brief analysis.

Coordinates PDF extraction, metrics calculation, frame detection, and recommendation extraction.
"""

import concurrent.futures
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .frame_detector import FrameDetector
from .llm_client import LLMClient
from .metrics_calculator import MetricsCalculator
from .models import (
    PerDocumentExtraction,
    ProcessingStatus,
    FrameAssessment,
    PolicyRecommendation
)
from .pdf_extractor import PDFExtractor
from .recommendation_extractor import RecommendationExtractor
from .utils import (
    create_document_id,
    ensure_output_directories,
    get_env_var,
    load_yaml_config,
    save_dataframe,
    save_json,
    validate_file_paths,
    clean_text_for_csv,
    ProgressTracker
)


logger = logging.getLogger(__name__)


class PolicyBriefPipeline:
    """Main pipeline for policy brief analysis."""
    
    def __init__(
        self,
        config_dir: Path,
        output_dir: Path,
        max_workers: int = 4,
        force_reprocess: bool = False
    ):
        """
        Initialize the policy brief analysis pipeline.
        
        Args:
            config_dir: Directory containing configuration files
            output_dir: Directory for output files
            max_workers: Maximum concurrent processing threads  
            force_reprocess: Skip hash-based change detection
        """
        self.config_dir = config_dir
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.force_reprocess = force_reprocess
        
        logger.info(f"Initializing pipeline with config from {config_dir}")
        
        # Load configuration
        self.config = self._load_configuration()
        self.frames = self._load_frames_config()
        self.enums = self._load_enums_config()
        
        # Initialize components
        self.llm_client = self._initialize_llm_client()
        self.pdf_extractor = self._initialize_pdf_extractor()
        self.metrics_calculator = MetricsCalculator()
        self.frame_detector = self._initialize_frame_detector()
        self.recommendation_extractor = self._initialize_recommendation_extractor()
        
        # Ensure output directories exist
        ensure_output_directories(output_dir)
        
        # Track processing cache
        self.processing_cache = self._load_processing_cache()
        
        logger.info("Pipeline initialization complete")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load main configuration file."""
        config_file = self.config_dir / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Main config file not found: {config_file}")
        
        return load_yaml_config(config_file)
    
    def _load_frames_config(self) -> List[Dict[str, Any]]:
        """Load frames configuration."""
        frames_file = self.config_dir / "frames.yaml"
        if not frames_file.exists():
            raise FileNotFoundError(f"Frames config file not found: {frames_file}")
        
        frames_config = load_yaml_config(frames_file)
        frames = frames_config.get("frames", [])
        
        if not frames:
            raise ValueError("No frames defined in frames.yaml")
        
        logger.info(f"Loaded {len(frames)} theoretical frames")
        return frames
    
    def _load_enums_config(self) -> Dict[str, List[str]]:
        """Load enums configuration."""
        enums_file = self.config_dir / "enums.yaml"
        if not enums_file.exists():
            raise FileNotFoundError(f"Enums config file not found: {enums_file}")
        
        return load_yaml_config(enums_file)
    
    def _initialize_llm_client(self) -> LLMClient:
        """Initialize OpenAI LLM client."""
        openai_config = self.config.get("openai", {})
        
        # Get API key from environment
        api_key = get_env_var("OPENAI_API_KEY", required=True)
        
        return LLMClient(
            api_key=api_key,
            model=openai_config.get("model", "gpt-4o-2024-08-06"),
            temperature=openai_config.get("temperature", 0.1),
            max_tokens=openai_config.get("max_tokens", 4000),
            timeout=openai_config.get("timeout", 60),
            max_retries=openai_config.get("max_retries", 3),
            retry_delay=openai_config.get("retry_delay", 1.0)
        )
    
    def _initialize_pdf_extractor(self) -> PDFExtractor:
        """Initialize PDF extractor."""
        pdf_config = self.config.get("pdf", {})
        
        return PDFExtractor(
            extract_method=pdf_config.get("extract_method", "pymupdf"),
            preserve_layout=pdf_config.get("preserve_layout", True),
            max_pages=pdf_config.get("max_pages", 0),
            max_file_size_mb=pdf_config.get("max_file_size_mb", 50)
        )
    
    def _initialize_frame_detector(self) -> FrameDetector:
        """Initialize frame detector."""
        frames_config = self.config.get("frames", {})
        
        return FrameDetector(
            llm_client=self.llm_client,
            frames_config=self.frames,
            min_confidence=frames_config.get("min_confidence", 0.7),
            max_spans_per_frame=frames_config.get("max_spans_per_frame", 5),
            context_window=frames_config.get("context_window", 500),
            min_evidence_quotes=frames_config.get("min_evidence_quotes", 1),
            max_evidence_quotes=frames_config.get("max_evidence_quotes", 3)
        )
    
    def _initialize_recommendation_extractor(self) -> RecommendationExtractor:
        """Initialize recommendation extractor."""
        rec_config = self.config.get("recommendations", {})
        
        return RecommendationExtractor(
            llm_client=self.llm_client,
            enums_config=self.enums,
            min_confidence=rec_config.get("min_confidence", 0.6),
            max_recommendations=rec_config.get("max_recommendations", 10),
            recommendation_signals=rec_config.get("recommendation_signals"),
            target_sections=rec_config.get("target_sections")
        )
    
    def _load_processing_cache(self) -> Dict[str, str]:
        """Load cache of processed files and their hashes."""
        cache_file = self.output_dir / ".processing_cache.json"
        
        if cache_file.exists() and not self.force_reprocess:
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load processing cache: {e}")
        
        return {}
    
    def _save_processing_cache(self) -> None:
        """Save processing cache to disk."""
        cache_file = self.output_dir / ".processing_cache.json" 
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.processing_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save processing cache: {e}")
    
    def process_documents(self, pdf_files: List[Path]) -> Dict[str, Any]:
        """
        Process a list of PDF documents.
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            Processing results summary
        """
        logger.info(f"Starting processing of {len(pdf_files)} documents")
        
        # Validate file paths
        valid_files = validate_file_paths(pdf_files)
        if len(valid_files) != len(pdf_files):
            logger.warning(f"Filtered {len(pdf_files) - len(valid_files)} invalid files")
        
        # Check which files need processing
        files_to_process = []
        skipped_files = []
        
        for file_path in valid_files:
            if self._needs_processing(file_path):
                files_to_process.append(file_path)
            else:
                skipped_files.append(file_path)
                logger.debug(f"Skipping unchanged file: {file_path}")
        
        logger.info(f"Processing {len(files_to_process)} files, skipping {len(skipped_files)} unchanged")
        
        if not files_to_process:
            logger.info("No files to process")
            return {
                "processed": [],
                "skipped": skipped_files,
                "errors": []
            }
        
        # Process files with concurrency
        processed_results = []
        errors = []
        
        progress = ProgressTracker(len(files_to_process), "Processing documents")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all processing tasks
            future_to_file = {
                executor.submit(self._process_single_document, file_path): file_path
                for file_path in files_to_process
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    if result:
                        processed_results.append(result)
                        # Update cache
                        file_hash = PDFExtractor.compute_file_hash(file_path)
                        self.processing_cache[str(file_path)] = file_hash
                    else:
                        errors.append(f"Processing failed for {file_path}")
                        
                except Exception as e:
                    logger.error(f"Processing failed for {file_path}: {e}")
                    errors.append(f"{file_path}: {str(e)}")
                
                progress.update()
        
        progress.finish()
        
        # Save processing cache
        self._save_processing_cache()
        
        # Generate output files
        if processed_results:
            self._generate_output_files(processed_results)
        
        return {
            "processed": processed_results,
            "skipped": skipped_files,
            "errors": errors
        }
    
    def _needs_processing(self, file_path: Path) -> bool:
        """Check if file needs processing based on content hash."""
        if self.force_reprocess:
            return True
        
        try:
            current_hash = PDFExtractor.compute_file_hash(file_path)
            cached_hash = self.processing_cache.get(str(file_path))
            
            return current_hash != cached_hash
            
        except Exception as e:
            logger.warning(f"Hash computation failed for {file_path}: {e}")
            return True  # Process if we can't determine
    
    def _process_single_document(self, file_path: Path) -> Optional[PerDocumentExtraction]:
        """
        Process a single PDF document.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Document extraction results or None if failed
        """
        start_time = time.time()
        doc_id = create_document_id(file_path)
        
        logger.info(f"Processing document: {doc_id} ({file_path.name})")
        
        try:
            # Extract PDF content
            pages, metadata, extraction_info = self.pdf_extractor.extract_document(file_path)
            
            if not pages or not any(page.text.strip() for page in pages):
                logger.warning(f"No text content extracted from {file_path}")
                return None
            
            # Extract headings
            headings = self.pdf_extractor.extract_headings(pages)
            
            # Calculate metrics
            full_text = "\n".join(page.text for page in pages)
            metrics = self.metrics_calculator.calculate_metrics(pages, headings, full_text)
            
            # Detect theoretical frames
            frame_assessments = self.frame_detector.detect_frames(pages)
            
            # Extract recommendations
            recommendations = self.recommendation_extractor.extract_recommendations(pages)
            
            # Generate recommendation IDs
            for i, rec in enumerate(recommendations):
                rec.rec_id = f"{doc_id}_rec_{i+1:02d}"
            
            # Create processing status
            processing_duration = time.time() - start_time
            file_hash = PDFExtractor.compute_file_hash(file_path)
            
            processing_status = ProcessingStatus(
                doc_id=doc_id,
                file_path=str(file_path),
                file_hash=file_hash,
                file_size_bytes=file_path.stat().st_size,
                processing_timestamp=datetime.now(),
                processing_duration_seconds=processing_duration, 
                parser_used=self.pdf_extractor.extract_method,
                likely_scanned=extraction_info.get("likely_scanned", False),
                text_extraction_quality=extraction_info.get("text_extraction_quality", 1.0),
                pages_processed=len(pages),
                frames_processed=len(frame_assessments),
                recommendations_extracted=len(recommendations),
                warnings=extraction_info.get("warnings", [])
            )
            
            # Create complete extraction result
            extraction = PerDocumentExtraction(
                doc_id=doc_id,
                pages=pages,
                headings=headings,
                metadata=metadata,
                metrics=metrics,
                frame_assessments=frame_assessments,
                recommendations=recommendations,
                processing_status=processing_status
            )
            
            # Save audit file
            self._save_audit_file(extraction)
            
            logger.info(f"Completed processing {doc_id}: {len(frame_assessments)} frames, {len(recommendations)} recommendations")
            
            return extraction
            
        except Exception as e:
            logger.error(f"Document processing failed for {file_path}: {e}")
            return None
    
    def _save_audit_file(self, extraction: PerDocumentExtraction) -> None:
        """Save per-document audit file."""
        audit_file = self.output_dir / "audit" / f"{extraction.doc_id}.json"
        
        try:
            # Convert to dict for JSON serialization
            audit_data = extraction.model_dump()
            
            # Include raw text in audit if configured
            include_raw_text = self.config.get("output", {}).get("include_raw_text", False)
            if not include_raw_text:
                # Remove text content to save space
                for page in audit_data["pages"]:
                    page["text"] = f"[Text content omitted - {page['char_count']} chars]"
            
            save_json(audit_data, audit_file, 
                     compress=self.config.get("output", {}).get("compress_json", True))
            
        except Exception as e:
            logger.error(f"Failed to save audit file for {extraction.doc_id}: {e}")
    
    def _generate_output_files(self, results: List[PerDocumentExtraction]) -> None:
        """Generate tabular output files from processing results."""
        logger.info("Generating output files...")
        
        output_config = self.config.get("output", {})
        formats = output_config.get("formats", ["csv"])
        
        try:
            # Generate documents table
            documents_df = self._create_documents_dataframe(results)
            
            for fmt in formats:
                if fmt in ["csv", "parquet"]:
                    save_dataframe(documents_df, self.output_dir / f"documents.{fmt}", fmt)
            
            # Generate frames table
            frames_df = self._create_frames_dataframe(results)
            
            for fmt in formats:
                if fmt in ["csv", "parquet"]:
                    save_dataframe(frames_df, self.output_dir / f"frames.{fmt}", fmt)
            
            # Generate recommendations table
            recommendations_df = self._create_recommendations_dataframe(results)
            
            for fmt in formats:
                if fmt in ["csv", "parquet"]:
                    save_dataframe(recommendations_df, self.output_dir / f"recommendations.{fmt}", fmt)
            
            logger.info("Output files generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate output files: {e}")
            raise
    
    def _create_documents_dataframe(self, results: List[PerDocumentExtraction]) -> pd.DataFrame:
        """Create documents dataframe from processing results."""
        rows = []
        
        for result in results:
            row = {
                # Identification
                "doc_id": result.doc_id,
                "file_path": result.processing_status.file_path,
                "file_name": Path(result.processing_status.file_path).name,
                
                # PDF metadata
                "title": result.metadata.title or "",
                "author": result.metadata.author or "",
                "creation_date": result.metadata.creation_date,
                "subject": result.metadata.subject or "",
                
                # Processing info
                "processing_timestamp": result.processing_status.processing_timestamp,
                "processing_duration_seconds": result.processing_status.processing_duration_seconds,
                "likely_scanned": result.processing_status.likely_scanned,
                "text_extraction_quality": result.processing_status.text_extraction_quality,
                
                # Document metrics
                **result.metrics.model_dump(),
                
                # Summary counts
                "frames_present": len([f for f in result.frame_assessments if f.decision == "present"]),
                "frames_absent": len([f for f in result.frame_assessments if f.decision == "absent"]),
                "recommendations_count": len(result.recommendations),
            }
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_frames_dataframe(self, results: List[PerDocumentExtraction]) -> pd.DataFrame:
        """Create frames dataframe from processing results."""
        rows = []
        
        for result in results:
            for assessment in result.frame_assessments:
                # Prepare evidence text
                evidence_quotes = [clean_text_for_csv(ev.quote, 200) for ev in assessment.evidence]
                evidence_pages = [str(ev.page) for ev in assessment.evidence]
                
                row = {
                    "doc_id": result.doc_id,
                    "frame_id": assessment.frame_id,
                    "frame_label": assessment.frame_label,
                    "decision": assessment.decision,
                    "confidence": assessment.confidence,
                    "evidence_count": len(assessment.evidence),
                    "evidence_quotes": " | ".join(evidence_quotes),
                    "evidence_pages": ",".join(evidence_pages),
                    "rationale": clean_text_for_csv(assessment.rationale, 300),
                    "counterevidence_count": len(assessment.counterevidence)
                }
                
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_recommendations_dataframe(self, results: List[PerDocumentExtraction]) -> pd.DataFrame:
        """Create recommendations dataframe from processing results."""
        rows = []
        
        for result in results:
            for rec in result.recommendations:
                # Prepare evidence text
                evidence_quotes = [clean_text_for_csv(ev.quote, 200) for ev in rec.evidence]
                evidence_pages = [str(ev.page) for ev in rec.evidence]
                
                row = {
                    "doc_id": result.doc_id,
                    "rec_id": rec.rec_id,
                    "actor": rec.actor,
                    "action": clean_text_for_csv(rec.action, 200),
                    "target": clean_text_for_csv(rec.target, 200),
                    "instrument_type": rec.instrument_type,
                    "policy_domain": rec.policy_domain,
                    "geographic_scope": rec.geographic_scope,
                    "timeframe": rec.timeframe,
                    "strength": rec.strength,
                    "evidence_count": len(rec.evidence),
                    "evidence_quotes": " | ".join(evidence_quotes),
                    "evidence_pages": ",".join(evidence_pages)
                }
                
                rows.append(row)
        
        return pd.DataFrame(rows)