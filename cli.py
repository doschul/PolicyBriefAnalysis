#!/usr/bin/env python3
"""
Policy Brief Analysis Pipeline CLI

Usage:
    python cli.py extract --input_dir ./pdfs --output_dir ./out --config ./config
"""
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If dotenv not available, try manual loading
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

from src.policybrief.pipeline import PolicyBriefPipeline
from src.policybrief.snippet_analysis import SnippetAnalyzer
from src.policybrief.cross_validator import CrossValidator


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log')
        ]
    )


@click.group()
def cli() -> None:
    """Policy Brief Analysis Pipeline CLI."""
    pass


@cli.command()
@click.option(
    '--input_dir', 
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help='Directory containing PDF files to process'
)
@click.option(
    '--output_dir',
    required=True, 
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help='Directory to write output files'
)
@click.option(
    '--config',
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help='Configuration directory containing frames.yaml and other config files'
)
@click.option(
    '--max_workers',
    default=4,
    help='Maximum number of concurrent workers for processing'
)
@click.option(
    '--force_reprocess',
    is_flag=True,
    help='Force reprocessing of all files, ignoring content hashes'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
@click.option(
    '--dry_run',
    is_flag=True,
    help='Show what would be processed without actually running'
)
def extract(
    input_dir: Path,
    output_dir: Path, 
    config: Path,
    max_workers: int,
    force_reprocess: bool,
    verbose: bool,
    dry_run: bool
) -> None:
    """Extract policy data from PDF files."""
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting policy brief extraction pipeline")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config directory: {config}")
    
    # Validate directories
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
        
    if not config.exists():
        logger.error(f"Config directory does not exist: {config}")
        sys.exit(1)
        
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDF files
    pdf_files = list(input_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
        
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    if dry_run:
        logger.info("DRY RUN - would process:")
        for pdf_file in pdf_files:
            logger.info(f"  {pdf_file}")
        return
    
    # Initialize pipeline
    try:
        pipeline = PolicyBriefPipeline(
            config_dir=config,
            output_dir=output_dir,
            max_workers=max_workers,
            force_reprocess=force_reprocess
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Process files
    try:
        results = pipeline.process_documents(pdf_files)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Processed {len(results['processed'])} documents")
        logger.info(f"Skipped {len(results['skipped'])} unchanged documents")
        
        if results['processed']:
            summary = pipeline.compute_extraction_summary(results['processed'])
            logger.info(f"Extraction summary: {summary['total_extractions']} extractions, "
                        f"{summary['total_frames_present']} frames present")
            if summary.get('warnings'):
                for w in summary['warnings']:
                    logger.warning(f"  [WARN] {w}")
        
        if results['errors']:
            logger.warning(f"Encountered {len(results['errors'])} errors:")
            for error in results['errors']:
                logger.warning(f"  {error}")
                
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


@cli.command()
@click.option(
    '--config',
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help='Configuration directory'
)
def validate_config(config: Path) -> None:
    """Validate configuration files."""
    setup_logging(True)
    logger = logging.getLogger(__name__)
    
    logger.info("Validating configuration...")
    
    try:
        # This will validate the config during initialization
        pipeline = PolicyBriefPipeline(
            config_dir=config,
            output_dir=Path("temp"),  # Won't be used for validation
            max_workers=1,
            force_reprocess=False
        )
        
        logger.info("✓ Configuration is valid")
        logger.info(f"✓ Found {len(pipeline.frames)} theoretical frames")
        
    except Exception as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        sys.exit(1)


@cli.command()
def version() -> None:
    """Show version information."""
    click.echo("Policy Brief Analysis Pipeline v0.1.0")


@cli.command(name="run-snippet-analysis")
@click.option(
    "--source-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help=(
        "For --input-source=ai: path to recommendations.csv. "
        "For --input-source=manual: path to PBs solutions coded segments.xlsx"
    ),
)
@click.option(
    "--input-source",
    type=click.Choice(["ai", "manual"], case_sensitive=False),
    default="ai",
    show_default=True,
    help="Input source: 'ai' (recommendations.csv) or 'manual' (coded segments Excel)",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to write recommendations_snippet_ai.csv or recommendations_snippet_manual.csv",
)
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Configuration directory (for OpenAI settings)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--max-workers",
    default=1,
    show_default=True,
    help="Number of parallel LLM workers (1 = sequential)",
)
def run_snippet_analysis(
    source_file: Path,
    input_source: str,
    output_dir: Path,
    config: Path,
    verbose: bool,
    max_workers: int,
) -> None:
    """Re-run extraction on individual solution sentences.

    Uses the same RECOMMENDATION_PROMPT as the document-level pipeline
    (input_mode='snippet').  Supports two input sources:

    \b
    ai:     reads source_text_raw from recommendations.csv
            → writes recommendations_snippet_ai.csv
    manual: reads Segment column from PBs solutions coded segments.xlsx
            → writes recommendations_snippet_manual.csv
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    from src.policybrief.llm_client import LLMClient
    from src.policybrief.utils import get_env_var, load_yaml_config

    cfg = load_yaml_config(config / "config.yaml")
    api_key = get_env_var("OPENAI_API_KEY", required=True)
    oai_cfg = cfg.get("openai", {})
    llm = LLMClient(
        api_key=api_key,
        model=oai_cfg.get("model", "gpt-4o-mini"),
        temperature=oai_cfg.get("temperature", 0.1),
        max_tokens=oai_cfg.get("max_tokens", 4000),
        timeout=oai_cfg.get("timeout", 60),
        max_retries=oai_cfg.get("max_retries", 5),
        retry_delay=oai_cfg.get("retry_delay", 2.0),
    )

    analyzer = SnippetAnalyzer(llm_client=llm, config=cfg.get("recommendation_extraction", {}))
    result_df = analyzer.analyze(
        source_path=source_file,
        output_dir=output_dir,
        input_source=input_source,
        max_workers=max_workers,
    )
    out_name = (
        "recommendations_snippet_ai.csv"
        if input_source == "ai"
        else "recommendations_snippet_manual.csv"
    )
    logger.info(f"Done. {len(result_df)} snippet extractions written to {output_dir / out_name}")


@cli.command()
@click.option(
    "--ai-snippet-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to recommendations_snippet_ai.csv",
)
@click.option(
    "--manual-snippet-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Path to recommendations_snippet_manual.csv",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory to write comparison.csv and metrics.json",
)
@click.option(
    "--corpus-file",
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Optional: path to Policy Briefs corpus.csv.xlsx for secondary comparison",
)
@click.option(
    "--ai-docs-file",
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Optional: path to documents.csv for visual-element comparison (requires --corpus-file)",
)
@click.option(
    "--sim-threshold",
    default=0.7,
    show_default=True,
    type=float,
    help="SequenceMatcher ratio threshold for fuzzy segment matching (0–1)",
)
@click.option(
    "--full-recs-file",
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Optional: path to recommendations.csv (Pass 1) for context comparison vs AI-snippet pass",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def validate(
    ai_snippet_file: Path,
    manual_snippet_file: Path,
    output_dir: Path,
    corpus_file: Optional[Path],
    ai_docs_file: Optional[Path],
    sim_threshold: float,
    full_recs_file: Optional[Path],
    verbose: bool,
) -> None:
    """Cross-validate AI snippet extractions against manual snippet extractions.

    Primary comparison: recommendations_snippet_ai.csv vs
    recommendations_snippet_manual.csv (presence, counts, extraction types).

    Optional secondary comparison against PB-level corpus indicators when
    --corpus-file is provided.

    Optional Pass 1 vs Pass 2 field-quality comparison when --full-recs-file
    is provided; writes context_comparison.csv and adds context_comparison
    section to metrics.json.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    validator = CrossValidator(corpus_path=corpus_file)

    metrics = validator.compare(
        ai_snippet_path=ai_snippet_file,
        manual_snippet_path=manual_snippet_file,
        output_dir=output_dir,
        ai_docs_path=ai_docs_file,
        sim_threshold=sim_threshold,
        full_recs_path=full_recs_file,
    )

    logger.info("Cross-validation complete.")
    primary = metrics.get("primary", {})
    pres = primary.get("presence_agreement", {})
    if pres.get("rate") is not None:
        logger.info(
            f"Presence agreement: {pres['agreed']}/{pres['total']} "
            f"({pres['rate']:.1%})"
        )
    type_agr = primary.get("type_agreement", {})
    if type_agr.get("rate") is not None:
        logger.info(
            f"Dominant type agreement: {type_agr['agreed']}/{type_agr['total']} "
            f"({type_agr['rate']:.1%})"
        )


if __name__ == '__main__':
    cli()