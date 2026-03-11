"""
PDF text extraction with layout preservation and metadata extraction.

Supports both PyPDF and PyMuPDF (fitz) for robust extraction across different PDF types.
"""

import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pypdf
from pypdf import PdfReader

from .models import PDFMetadata, PageText


logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text and metadata from PDF files."""
    
    def __init__(
        self,
        extract_method: str = "pymupdf",
        preserve_layout: bool = True,
        max_pages: int = 0,
        max_file_size_mb: int = 50
    ):
        """
        Initialize PDF extractor.
        
        Args:
            extract_method: "pypdf" or "pymupdf"  
            preserve_layout: Whether to attempt layout preservation
            max_pages: Maximum pages to process (0 = no limit)
            max_file_size_mb: Skip files larger than this (MB)
        """
        self.extract_method = extract_method
        self.preserve_layout = preserve_layout
        self.max_pages = max_pages
        self.max_file_size_mb = max_file_size_mb
        
        if extract_method not in ["pypdf", "pymupdf"]:
            raise ValueError(f"Unsupported extract method: {extract_method}")
    
    def extract_document(self, file_path: Path) -> Tuple[List[PageText], PDFMetadata, Dict]:
        """
        Extract text and metadata from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (pages, metadata, extraction_info)
            
        Raises:
            ValueError: If file is too large or invalid
            Exception: If extraction fails
        """
        logger.info(f"Extracting PDF: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if self.max_file_size_mb > 0 and file_size > (self.max_file_size_mb * 1024 * 1024):
            raise ValueError(f"File too large: {file_size / (1024*1024):.1f}MB > {self.max_file_size_mb}MB")
        
        # Extract based on method
        if self.extract_method == "pymupdf":
            return self._extract_with_fitz(file_path)
        else:
            return self._extract_with_pypdf(file_path)
    
    def _extract_with_fitz(self, file_path: Path) -> Tuple[List[PageText], PDFMetadata, Dict]:
        """Extract using PyMuPDF (fitz) - better layout preservation."""
        try:
            doc = fitz.open(file_path)
            
            # Extract metadata
            metadata = self._extract_fitz_metadata(doc)
            
            # Extract pages
            pages = []
            total_pages = len(doc)
            max_pages = self.max_pages if self.max_pages > 0 else total_pages
            
            extraction_info = {
                "total_pages": total_pages,
                "pages_processed": min(total_pages, max_pages),
                "likely_scanned": False,
                "text_extraction_quality": 1.0
            }
            
            # Track scanned document detection
            text_lengths = []
            
            for page_num in range(min(total_pages, max_pages)):
                page = doc[page_num]
                
                if self.preserve_layout:
                    # Get text with layout info
                    text = page.get_text("dict")
                    page_text = self._process_fitz_layout(text)
                else:
                    # Simple text extraction
                    page_text = page.get_text()
                
                # Clean text
                page_text = self._clean_text(page_text)
                
                # Count stats
                char_count = len(page_text)
                word_count = len(page_text.split()) if page_text else 0
                text_lengths.append(char_count)
                
                pages.append(PageText(
                    page_num=page_num + 1,  # 1-based
                    text=page_text,
                    char_count=char_count,
                    word_count=word_count
                ))
            
            doc.close()
            
            # Detect likely scanned documents
            if text_lengths:
                avg_text_length = sum(text_lengths) / len(text_lengths)
                extraction_info["likely_scanned"] = avg_text_length < 100
                
                # Quality estimate based on text density
                expected_chars_per_page = 2000  # Rough estimate
                quality = min(1.0, avg_text_length / expected_chars_per_page)
                extraction_info["text_extraction_quality"] = max(0.1, quality)
            
            return pages, metadata, extraction_info
            
        except Exception as e:
            logger.error(f"Fitz extraction failed for {file_path}: {e}")
            raise
    
    def _extract_with_pypdf(self, file_path: Path) -> Tuple[List[PageText], PDFMetadata, Dict]:
        """Extract using PyPDF - fallback option."""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                
                # Extract metadata
                metadata = self._extract_pypdf_metadata(reader)
                
                # Extract pages
                pages = []
                total_pages = len(reader.pages)
                max_pages = self.max_pages if self.max_pages > 0 else total_pages
                
                extraction_info = {
                    "total_pages": total_pages,
                    "pages_processed": min(total_pages, max_pages),
                    "likely_scanned": False,
                    "text_extraction_quality": 0.8  # Lower than fitz
                }
                
                text_lengths = []
                
                for page_num in range(min(total_pages, max_pages)):
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    # Clean text
                    page_text = self._clean_text(page_text)
                    
                    # Count stats
                    char_count = len(page_text)
                    word_count = len(page_text.split()) if page_text else 0
                    text_lengths.append(char_count)
                    
                    pages.append(PageText(
                        page_num=page_num + 1,  # 1-based
                        text=page_text,
                        char_count=char_count,
                        word_count=word_count
                    ))
                
                # Detect likely scanned documents
                if text_lengths:
                    avg_text_length = sum(text_lengths) / len(text_lengths)
                    extraction_info["likely_scanned"] = avg_text_length < 100
                
                return pages, metadata, extraction_info
                
        except Exception as e:
            logger.error(f"PyPDF extraction failed for {file_path}: {e}")
            raise
    
    def _process_fitz_layout(self, text_dict: Dict) -> str:
        """Process PyMuPDF text dict to preserve layout."""
        lines = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:  # Text block
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    
                    if line_text.strip():
                        lines.append(line_text.strip())
            
        return "\n".join(lines)
    
    def _extract_fitz_metadata(self, doc) -> PDFMetadata:
        """Extract metadata using PyMuPDF."""
        meta = doc.metadata
        
        def parse_date(date_str: Optional[str]) -> Optional[datetime]:
            """Parse PDF date string."""
            if not date_str:
                return None
            
            # Remove PDF date prefix if present
            if date_str.startswith("D:"):
                date_str = date_str[2:]
            
            # Try common formats
            for fmt in ["%Y%m%d%H%M%S", "%Y%m%d", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(date_str[:len(fmt)], fmt)
                except ValueError:
                    continue
            
            return None
        
        return PDFMetadata(
            title=meta.get("title"),
            author=meta.get("author"), 
            subject=meta.get("subject"),
            creator=meta.get("creator"),
            producer=meta.get("producer"),
            creation_date=parse_date(meta.get("creationDate")),
            modification_date=parse_date(meta.get("modDate"))
        )
    
    def _extract_pypdf_metadata(self, reader: PdfReader) -> PDFMetadata:
        """Extract metadata using PyPDF."""
        meta = reader.metadata
        
        if not meta:
            return PDFMetadata()
        
        def safe_get(key: str) -> Optional[str]:
            """Safely get metadata field."""
            try:
                value = meta.get(key)
                return str(value) if value else None
            except:
                return None
        
        return PDFMetadata(
            title=safe_get("/Title"),
            author=safe_get("/Author"),
            subject=safe_get("/Subject"),
            creator=safe_get("/Creator"),
            producer=safe_get("/Producer"),
            # PyPDF date handling is complex, skip for now
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/\@\#\$\%\&\*\+\=\<\>\{\}\|\\\~\`]', '', text)
        
        # Remove excessive dots (from OCR artifacts)
        text = re.sub(r'\.{4,}', '...', text)
        
        return text.strip()
    
    def extract_headings(self, pages: List[PageText]) -> List[str]:
        """Extract potential document headings."""
        headings = []
        
        for page in pages:
            lines = page.text.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Simple heading detection heuristics
                if (line and 
                    len(line) < 100 and  # Not too long
                    len(line.split()) >= 2 and  # At least 2 words
                    not line.endswith('.') and  # No sentence endings
                    (line.isupper() or  # All caps
                     line.istitle() or  # Title case
                     re.match(r'^\d+\.?\s+[A-Z]', line))):  # Numbered heading
                    
                    headings.append(line)
        
        return headings
    
    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Compute SHA256 hash of file for change detection."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()