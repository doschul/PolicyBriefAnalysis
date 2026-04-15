"""PDF text extraction with PyMuPDF (primary) and pypdf (fallback)."""

import hashlib
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import PDFMetadata, PageText

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text and metadata from PDF files."""

    def __init__(
        self,
        method: str = "pymupdf",
        preserve_layout: bool = True,
        max_pages: int = 0,
        max_file_size_mb: int = 50,
    ):
        self.method = method
        self.preserve_layout = preserve_layout
        self.max_pages = max_pages
        self.max_file_size_mb = max_file_size_mb

    # ── Public API ────────────────────────────────────────────────────

    def extract(self, file_path: Path) -> Tuple[List[PageText], PDFMetadata]:
        """Extract pages and metadata from a PDF file."""
        self._validate_file(file_path)

        if self.method == "pymupdf":
            try:
                return self._extract_with_fitz(file_path)
            except ImportError:
                logger.warning("PyMuPDF not available, falling back to pypdf")
                return self._extract_with_pypdf(file_path)
        return self._extract_with_pypdf(file_path)

    def compute_file_hash(self, file_path: Path) -> str:
        """SHA-256 hash of the file (for caching / dedup)."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def detect_scanned(self, pages: List[PageText]) -> Tuple[bool, float]:
        """Heuristic: is this likely a scanned / image-only PDF?"""
        if not pages:
            return True, 0.0
        total_chars = sum(p.char_count for p in pages)
        avg_chars = total_chars / len(pages)
        # Scanned PDFs typically yield < 50 chars per page
        if avg_chars < 50:
            return True, 0.0
        quality = min(1.0, avg_chars / 500)
        return False, round(quality, 3)

    # ── PyMuPDF backend ───────────────────────────────────────────────

    def _extract_with_fitz(self, file_path: Path) -> Tuple[List[PageText], PDFMetadata]:
        import fitz

        doc = fitz.open(str(file_path))
        try:
            metadata = self._parse_fitz_metadata(doc.metadata or {})
            pages: List[PageText] = []
            page_limit = self.max_pages or len(doc)

            for idx in range(min(len(doc), page_limit)):
                page = doc[idx]
                if self.preserve_layout:
                    text = self._extract_layout_text(page)
                else:
                    text = page.get_text("text") or ""
                words = text.split()
                pages.append(PageText(
                    page_num=idx + 1,
                    text=text,
                    char_count=len(text),
                    word_count=len(words),
                ))
            return pages, metadata
        finally:
            doc.close()

    def _extract_layout_text(self, page: Any) -> str:
        """Extract text preserving layout via text blocks."""
        blocks = page.get_text("blocks") or []
        text_blocks = sorted(
            (b for b in blocks if b[6] == 0),  # type 0 = text
            key=lambda b: (b[1], b[0]),  # sort by y then x
        )
        lines: List[str] = []
        for b in text_blocks:
            block_text = b[4].strip() if b[4] else ""
            if block_text:
                lines.append(block_text)
        return "\n".join(lines)

    @staticmethod
    def _parse_fitz_metadata(raw: Dict[str, Any]) -> PDFMetadata:
        def _parse_date(val: Optional[str]) -> Optional[datetime]:
            if not val:
                return None
            for fmt in ("%Y%m%d%H%M%S", "%Y-%m-%d", "%Y%m%d"):
                cleaned = re.sub(r"^D:", "", val).split("+")[0].split("-")[0]
                try:
                    return datetime.strptime(cleaned[:len(fmt.replace("%", ""))], fmt)
                except (ValueError, IndexError):
                    continue
            return None

        return PDFMetadata(
            title=raw.get("title") or None,
            author=raw.get("author") or None,
            subject=raw.get("subject") or None,
            creator=raw.get("creator") or None,
            producer=raw.get("producer") or None,
            creation_date=_parse_date(raw.get("creationDate")),
            modification_date=_parse_date(raw.get("modDate")),
        )

    # ── pypdf backend ─────────────────────────────────────────────────

    def _extract_with_pypdf(self, file_path: Path) -> Tuple[List[PageText], PDFMetadata]:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        metadata = self._parse_pypdf_metadata(reader.metadata)
        pages: List[PageText] = []
        page_limit = self.max_pages or len(reader.pages)

        for idx in range(min(len(reader.pages), page_limit)):
            text = reader.pages[idx].extract_text() or ""
            words = text.split()
            pages.append(PageText(
                page_num=idx + 1,
                text=text,
                char_count=len(text),
                word_count=len(words),
            ))
        return pages, metadata

    @staticmethod
    def _parse_pypdf_metadata(meta: Any) -> PDFMetadata:
        if not meta:
            return PDFMetadata()
        return PDFMetadata(
            title=getattr(meta, "title", None),
            author=getattr(meta, "author", None),
            subject=getattr(meta, "subject", None),
            creator=getattr(meta, "creator", None),
            producer=getattr(meta, "producer", None),
        )

    # ── Validation ────────────────────────────────────────────────────

    def _validate_file(self, file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        if not file_path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if self.max_file_size_mb and size_mb > self.max_file_size_mb:
            raise ValueError(
                f"File too large: {size_mb:.1f} MB (limit {self.max_file_size_mb} MB)"
            )
