"""
Utility functions for the policy brief analysis pipeline.

Common functions for file handling, configuration loading, and data processing.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


logger = logging.getLogger(__name__)


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Parsed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        logger.debug(f"Loaded config from {config_path}")
        return config or {}
        
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML config {config_path}: {e}")
        raise


def save_json(data: Any, file_path: Path, compress: bool = False) -> None:
    """
    Save data as JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        file_path: Output file path
        compress: Whether to compress the file
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if compress:
            import gzip
            with gzip.open(f"{file_path}.gz", 'wt', encoding='utf-8') as file:
                json.dump(data, file, indent=2, default=str, ensure_ascii=False)
        else:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, default=str, ensure_ascii=False)
        
        logger.debug(f"Saved JSON to {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise


def load_json(file_path: Path, compressed: bool = False) -> Any:
    """
    Load data from JSON file.
    
    Args:
        file_path: JSON file path
        compressed: Whether file is gzipped
        
    Returns:
        Loaded data
    """
    try:
        if compressed:
            import gzip
            with gzip.open(file_path, 'rt', encoding='utf-8') as file:
                data = json.load(file)
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        
        logger.debug(f"Loaded JSON from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        raise


def save_dataframe(df: pd.DataFrame, file_path: Path, format: str = "csv") -> None:
    """
    Save DataFrame in specified format.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        format: Output format ("csv" or "parquet")
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format.lower() == "parquet":
            df.to_parquet(file_path, index=False)
        elif format.lower() == "csv":
            df.to_csv(file_path, index=False, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug(f"Saved {format.upper()} to {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to save {format} to {file_path}: {e}")
        raise


def create_document_id(file_path: Path) -> str:
    """
    Create stable document ID from file path.
    
    Args:
        file_path: Source file path
        
    Returns:
        Document ID string
    """
    # Use filename without extension as base
    base_name = file_path.stem
    
    # Clean up name for use as ID
    doc_id = base_name.lower().replace(' ', '_')
    doc_id = ''.join(c for c in doc_id if c.isalnum() or c in '_-')
    
    # Ensure it starts with letter
    if not doc_id[0].isalpha():
        doc_id = 'doc_' + doc_id
    
    return doc_id


def flatten_evidence_list(evidence_list: List[Dict]) -> List[Dict]:
    """
    Flatten evidence list for tabular output.
    
    Args:
        evidence_list: List of evidence dictionaries
        
    Returns:
        Flattened list with one row per evidence item
    """
    flattened = []
    
    for item in evidence_list:
        if isinstance(item, dict):
            flattened.append(item)
        else:
            # Handle Pydantic models
            flattened.append(item.model_dump())
    
    return flattened


def ensure_output_directories(output_dir: Path) -> None:
    """
    Ensure output directory structure exists.
    
    Args:
        output_dir: Base output directory
    """
    directories = [
        output_dir,
        output_dir / "audit"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")


def get_env_var(var_name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Get environment variable with validation.
    
    Args:
        var_name: Environment variable name
        default: Default value if not set
        required: Whether variable is required
        
    Returns:
        Variable value or None
        
    Raises:
        ValueError: If required variable is missing
    """
    value = os.getenv(var_name, default)
    
    if required and not value:
        raise ValueError(f"Required environment variable not set: {var_name}")
    
    return value


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    
    return f"{size_bytes:.1f} TB"


def validate_file_paths(file_paths: List[Path]) -> List[Path]:
    """
    Validate and filter file paths.
    
    Args:
        file_paths: List of file paths to validate
        
    Returns:
        List of valid file paths
    """
    valid_paths = []
    
    for path in file_paths:
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue
            
        if not path.is_file():
            logger.warning(f"Not a file: {path}")
            continue
            
        if path.suffix.lower() != '.pdf':
            logger.warning(f"Not a PDF file: {path}")
            continue
            
        valid_paths.append(path)
    
    return valid_paths


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration
    """
    merged = {}
    
    for config in configs:
        if config:
            merged = _deep_merge(merged, config)
    
    return merged


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if (key in result and 
            isinstance(result[key], dict) and 
            isinstance(value, dict)):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Optional log format string
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_string))
    logging.getLogger().addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)


class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total: int, description: str = "Processing"):
        """Initialize progress tracker."""
        self.total = total
        self.current = 0
        self.description = description
        self.logger = logging.getLogger(__name__)
    
    def update(self, increment: int = 1) -> None:
        """Update progress."""
        self.current += increment
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        self.logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def finish(self) -> None:
        """Mark as finished."""
        self.logger.info(f"{self.description}: Complete ({self.total} items)")


def clean_text_for_csv(text: str, max_length: int = 1000) -> str:
    """
    Clean text for CSV output.
    
    Args:
        text: Text to clean
        max_length: Maximum length to keep
        
    Returns:
        Cleaned and truncated text
    """
    if not text:
        return ""
    
    # Remove problematic characters
    cleaned = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Normalize whitespace
    cleaned = ' '.join(cleaned.split())
    
    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length-3] + "..."
    
    return cleaned