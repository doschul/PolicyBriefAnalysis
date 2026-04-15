"""Utility functions for the policy brief analysis pipeline."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(data: Any, file_path: Path, compress: bool = False) -> None:
    """Save data as JSON (optionally gzipped)."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        import gzip
        with gzip.open(f"{file_path}.gz", "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)


def load_json(file_path: Path, compressed: bool = False) -> Any:
    """Load data from JSON file."""
    if compressed:
        import gzip
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe(df: pd.DataFrame, file_path: Path, fmt: str = "csv") -> None:
    """Save DataFrame as CSV or Parquet."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(file_path, index=False)
    else:
        df.to_csv(file_path, index=False, encoding="utf-8")


def create_document_id(file_path: Path) -> str:
    """Create a stable document ID from a file path."""
    doc_id = file_path.stem.lower().replace(" ", "_")
    doc_id = "".join(c for c in doc_id if c.isalnum() or c in "_-")
    if not doc_id or not doc_id[0].isalpha():
        doc_id = "doc_" + doc_id
    return doc_id


def ensure_output_directories(output_dir: Path) -> None:
    """Ensure the output directory structure exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit").mkdir(parents=True, exist_ok=True)


def get_env_var(name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get environment variable with optional required check."""
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Required environment variable not set: {name}")
    return value


def validate_file_paths(file_paths: List[Path]) -> List[Path]:
    """Return only valid, existing PDF paths."""
    valid = []
    for p in file_paths:
        if not p.exists() or not p.is_file():
            logger.warning(f"Skipping invalid path: {p}")
            continue
        if p.suffix.lower() != ".pdf":
            logger.warning(f"Not a PDF: {p}")
            continue
        valid.append(p)
    return valid


def clean_text_for_csv(text: str, max_length: int = 1000) -> str:
    """Clean text for CSV output: collapse whitespace and truncate."""
    if not text:
        return ""
    cleaned = " ".join(text.replace("\n", " ").replace("\r", " ").split())
    if len(cleaned) > max_length:
        cleaned = cleaned[: max_length - 3] + "..."
    return cleaned


class ProgressTracker:
    """Simple progress logging helper."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description

    def update(self, increment: int = 1) -> None:
        self.current += increment
        pct = (self.current / self.total * 100) if self.total else 0
        logger.info(f"{self.description}: {self.current}/{self.total} ({pct:.0f}%)")

    def finish(self) -> None:
        logger.info(f"{self.description}: Complete ({self.total} items)")
