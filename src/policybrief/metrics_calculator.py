"""
Document metrics calculation including structural and linguistic analysis.

Computes comprehensive document-level metrics for policy brief analysis.
"""

import logging
import re
from typing import List, Optional

import nltk
import textstat

from .models import DocumentMetrics, PageText


logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate comprehensive document metrics."""
    
    def __init__(self):
        """Initialize metrics calculator with required NLTK data."""
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self) -> None:
        """Download required NLTK data if not present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
    
    def calculate_metrics(
        self, 
        pages: List[PageText], 
        headings: List[str],
        full_text: Optional[str] = None
    ) -> DocumentMetrics:
        """
        Calculate comprehensive document metrics.
        
        Args:
            pages: List of page text objects
            headings: Detected document headings
            full_text: Optional pre-computed full text (for efficiency)
            
        Returns:
            DocumentMetrics object with all computed metrics
        """
        logger.debug(f"Calculating metrics for {len(pages)} pages")
        
        # Get full text if not provided
        if full_text is None:
            full_text = "\n".join(page.text for page in pages)
        
        # Basic structure metrics
        page_count = len(pages)
        word_count = sum(page.word_count for page in pages)
        char_count = sum(page.char_count for page in pages)
        
        # Text structure analysis
        structure_metrics = self._analyze_text_structure(full_text)
        
        # Linguistic metrics
        linguistic_metrics = self._analyze_linguistics(full_text, word_count)
        
        # Readability metrics
        readability_metrics = self._calculate_readability(full_text)
        
        # Content analysis
        content_metrics = self._analyze_content_features(full_text)
        
        return DocumentMetrics(
            # Basic structure
            page_count=page_count,
            word_count=word_count,
            char_count=char_count,
            
            # Text structure
            heading_count=len(headings),
            paragraph_count=structure_metrics["paragraph_count"],
            sentence_count=structure_metrics["sentence_count"],
            list_item_count=structure_metrics["list_item_count"],
            
            # Linguistic metrics
            avg_sentence_length=linguistic_metrics["avg_sentence_length"],
            lexical_diversity=linguistic_metrics["lexical_diversity"],
            avg_word_length=linguistic_metrics["avg_word_length"],
            
            # Readability
            flesch_kincaid_grade=readability_metrics["flesch_kincaid_grade"],
            flesch_reading_ease=readability_metrics["flesch_reading_ease"],
            
            # Content features
            table_count=content_metrics["table_count"],
            figure_count=content_metrics["figure_count"],
            reference_count=content_metrics["reference_count"],
            url_count=content_metrics["url_count"],
            
            # Style analysis
            passive_voice_percent=linguistic_metrics.get("passive_voice_percent")
        )
    
    def _analyze_text_structure(self, text: str) -> dict:
        """Analyze text structure (paragraphs, sentences, lists)."""
        if not text:
            return {
                "paragraph_count": 0,
                "sentence_count": 0,
                "list_item_count": 0
            }
        
        # Count paragraphs (double newlines or distinct lines)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Count sentences using NLTK
        try:
            sentences = nltk.sent_tokenize(text)
            sentence_count = len([s for s in sentences if len(s.strip()) > 10])
        except Exception as e:
            logger.warning(f"Sentence tokenization failed: {e}")
            # Fallback: count by punctuation
            sentence_count = len(re.findall(r'[.!?]+', text))
        
        # Count list items
        list_patterns = [
            r'^\s*[•·▪▫‣⁃]\s',  # Bullet points
            r'^\s*[\-\*]\s',     # Dash/asterisk bullets  
            r'^\s*\d+[\.\)]\s',  # Numbered lists
            r'^\s*[a-z][\.\)]\s', # Lettered lists
        ]
        
        list_item_count = 0
        for line in text.split('\n'):
            line = line.strip()
            if any(re.match(pattern, line) for pattern in list_patterns):
                list_item_count += 1
        
        return {
            "paragraph_count": paragraph_count,
            "sentence_count": sentence_count,
            "list_item_count": list_item_count
        }
    
    def _analyze_linguistics(self, text: str, word_count: int) -> dict:
        """Analyze linguistic characteristics."""
        if not text or word_count == 0:
            return {
                "avg_sentence_length": 0.0,
                "lexical_diversity": 0.0,
                "avg_word_length": 0.0,
                "passive_voice_percent": None
            }
        
        # Tokenize words and sentences
        try:
            words = nltk.word_tokenize(text.lower())
            sentences = nltk.sent_tokenize(text)
            
            # Filter to actual words (remove punctuation)
            actual_words = [w for w in words if w.isalpha()]
            
        except Exception as e:
            logger.warning(f"Linguistic analysis failed: {e}")
            # Fallback to simple splitting
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            actual_words = words
            sentences = re.split(r'[.!?]+', text)
        
        # Average sentence length
        valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if valid_sentences:
            total_words_in_sentences = sum(len(s.split()) for s in valid_sentences)
            avg_sentence_length = total_words_in_sentences / len(valid_sentences)
        else:
            avg_sentence_length = 0.0
        
        # Lexical diversity (Type-Token Ratio)
        if actual_words:
            unique_words = set(actual_words)
            lexical_diversity = len(unique_words) / len(actual_words)
        else:
            lexical_diversity = 0.0
        
        # Average word length
        if actual_words:
            avg_word_length = sum(len(word) for word in actual_words) / len(actual_words)
        else:
            avg_word_length = 0.0
        
        # Passive voice estimation (simple heuristic)
        passive_voice_percent = self._estimate_passive_voice(text)
        
        return {
            "avg_sentence_length": avg_sentence_length,
            "lexical_diversity": lexical_diversity,
            "avg_word_length": avg_word_length,
            "passive_voice_percent": passive_voice_percent
        }
    
    def _calculate_readability(self, text: str) -> dict:
        """Calculate readability metrics using textstat."""
        if not text or len(text.strip()) < 100:
            return {
                "flesch_kincaid_grade": None,
                "flesch_reading_ease": None
            }
        
        try:
            # Textstat requires reasonable length text
            fk_grade = textstat.flesch_kincaid_grade(text)
            fk_ease = textstat.flesch_reading_ease(text)
            
            # Validate results (textstat can return weird values)
            if fk_grade < 0 or fk_grade > 25:
                fk_grade = None
            if fk_ease < 0 or fk_ease > 100:
                fk_ease = None
                
        except Exception as e:
            logger.warning(f"Readability calculation failed: {e}")
            fk_grade = None
            fk_ease = None
        
        return {
            "flesch_kincaid_grade": fk_grade,
            "flesch_reading_ease": fk_ease
        }
    
    def _analyze_content_features(self, text: str) -> dict:
        """Analyze content features like tables, figures, references."""
        if not text:
            return {
                "table_count": 0,
                "figure_count": 0, 
                "reference_count": 0,
                "url_count": 0
            }
        
        # Table detection (heuristic)
        table_indicators = [
            r'\btable\s+\d+',
            r'^\s*\|.*\|.*\|',  # Markdown-style tables
            r'\t.*\t.*\t',      # Tab-separated columns
        ]
        table_count = sum(len(re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)) 
                         for pattern in table_indicators)
        
        # Figure detection
        figure_indicators = [
            r'\bfigure\s+\d+',
            r'\bfig\.?\s+\d+',
            r'\b(chart|graph|diagram)\s+\d+',
        ]
        figure_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                          for pattern in figure_indicators)
        
        # Reference detection  
        reference_patterns = [
            r'\[\d+\]',                    # [1]
            r'\(\w+\s+\d{4}\)',          # (Author 2020)
            r'\b\w+\s+et\s+al\.?\s+\d{4}', # Author et al. 2020
            r'\bdoi:|DOI:',               # DOI references
        ]
        reference_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                             for pattern in reference_patterns)
        
        # URL detection
        url_patterns = [
            r'https?://[^\s]+',
            r'www\.[^\s]+\.[a-z]{2,}',
        ]
        url_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                       for pattern in url_patterns)
        
        return {
            "table_count": table_count,
            "figure_count": figure_count,
            "reference_count": reference_count, 
            "url_count": url_count
        }
    
    def _estimate_passive_voice(self, text: str) -> Optional[float]:
        """Estimate percentage of passive voice constructions."""
        if not text or len(text.strip()) < 100:
            return None
        
        try:
            sentences = nltk.sent_tokenize(text)
            if not sentences:
                return None
            
            # Simple passive voice patterns
            passive_patterns = [
                r'\b(is|are|was|were|being|been)\s+\w+ed\b',
                r'\b(is|are|was|were|being|been)\s+\w+en\b',
                r'\bby\s+\w+\b'  # "by X" construction
            ]
            
            passive_count = 0
            for sentence in sentences:
                if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in passive_patterns):
                    passive_count += 1
            
            return (passive_count / len(sentences)) * 100 if sentences else 0.0
            
        except Exception as e:
            logger.warning(f"Passive voice estimation failed: {e}")
            return None