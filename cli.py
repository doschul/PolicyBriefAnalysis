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
                    logger.warning(f"  ⚠ {w}")
        
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


if __name__ == '__main__':
    cli()