"""Cheap, reliable document metrics."""

import logging
import re
from typing import List, Optional

from .models import DocumentMetrics, PageText

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


class MetricsCalculator:
    """Compute basic document metrics from extracted page text."""

    def calculate_metrics(self, pages: List[PageText]) -> DocumentMetrics:
        full_text = "\n\n".join(p.text for p in pages)
        words = full_text.split()
        word_count = len(words)
        char_count = sum(p.char_count for p in pages)
        sentences = self._split_sentences(full_text)
        sentence_count = len(sentences)
        paragraphs = [p for p in full_text.split("\n\n") if p.strip()]
        paragraph_count = len(paragraphs)

        avg_sentence_length = (
            word_count / sentence_count if sentence_count else 0.0
        )
        unique_words = set(w.lower() for w in words)
        lexical_diversity = (
            len(unique_words) / word_count if word_count else 0.0
        )
        total_word_len = sum(len(w) for w in words)
        avg_word_length = total_word_len / word_count if word_count else 0.0

        fk_grade = self._flesch_kincaid_grade(full_text)
        fk_ease = self._flesch_reading_ease(full_text)

        return DocumentMetrics(
            page_count=len(pages),
            word_count=word_count,
            char_count=char_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            avg_sentence_length=round(avg_sentence_length, 2),
            lexical_diversity=round(lexical_diversity, 4),
            avg_word_length=round(avg_word_length, 2),
            flesch_kincaid_grade=fk_grade,
            flesch_reading_ease=fk_ease,
            url_count=len(_URL_RE.findall(full_text)),
            email_count=len(_EMAIL_RE.findall(full_text)),
        )

    # ── Internals ─────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Cheap sentence splitter (regex-based)."""
        raw = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw if len(s.strip()) > 5]

    @staticmethod
    def _flesch_kincaid_grade(text: str) -> Optional[float]:
        try:
            import textstat
            score = textstat.flesch_kincaid_grade(text)
            return round(score, 2) if score else None
        except Exception:
            return None

    @staticmethod
    def _flesch_reading_ease(text: str) -> Optional[float]:
        try:
            import textstat
            score = textstat.flesch_reading_ease(text)
            return round(score, 2) if score else None
        except Exception:
            return None
