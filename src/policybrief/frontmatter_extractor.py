"""
Content-first front-matter extraction for policy briefs.

Extracts author, affiliation, contact, funding info, and linked studies
from actual document content rather than relying solely on PDF metadata.
Uses deterministic text pattern matching and layout heuristics.
"""

import re
from typing import List, Optional, Set, Tuple

from .models import PageText, DocumentFrontMatter


class FrontMatterExtractor:
    """Extract front matter from document content using pattern matching."""
    
    # Regex patterns for content extraction
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    # More conservative URL pattern - requires http/https or www
    URL_PATTERN = re.compile(r'(?:https?://|www\.)[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?')
    
    # Title extraction patterns - look for titles at document start
    TITLE_PATTERNS = [
        re.compile(r'^(.{1,150})$', re.MULTILINE),  # Short lines at start
        re.compile(r'(?:^|\n)([A-Z][A-Z\s:]{10,100}[A-Z])(?:\n|$)'),  # ALL CAPS titles
        re.compile(r'(?:^|\n)([A-Z][a-z\s:]{15,120}[a-zA-Z])(?:\n|$)'),  # Title case
    ]
    
    # Section header patterns for targeted extraction
    AUTHOR_SECTION_PATTERNS = [
        re.compile(r'(?:^|\n)(Authors?:?\s*\n?(.{1,500}?))\n\n', re.IGNORECASE | re.DOTALL),
        re.compile(r'(?:^|\n)(Written by:?\s*\n?(.{1,500}?))\n\n', re.IGNORECASE | re.DOTALL),
        re.compile(r'(?:^|\n)(By:?\s+(.{1,200}?))\n', re.IGNORECASE),
        re.compile(r'(?:^|\n)(About the authors?\s*\n?(.{1,800}?))\n\n', re.IGNORECASE | re.DOTALL),
    ]
    
    AFFILIATION_PATTERNS = [
        re.compile(r'(?:^|\n)(.*(?:University|Institute|Foundation|Center|Centre|College|School|Department).*)', re.IGNORECASE),
        re.compile(r'(?:^|\n)(.*(?:Research|Policy|Think Tank|Organization|Organisation).*)', re.IGNORECASE),
    ]
    
    FUNDING_PATTERNS = [
        re.compile(r'(?:^|\n)((?:Funding|Supported by|Grant|This work|This research|Financial support).{1,400}?)(?:\n\n|\n[A-Z])', re.IGNORECASE | re.DOTALL),
        re.compile(r'(?:^|\n)((?:Acknowledgments?|Acknowledgements?)\s*\n?(.{1,600}?))\n\n', re.IGNORECASE | re.DOTALL),
    ]
    
    LINKED_STUDY_PATTERNS = [
        re.compile(r'(?:This is (?:a |an )?(?:summary|brief|excerpt|companion piece) (?:of|to|from)\s+)([^\.]+)', re.IGNORECASE),
        re.compile(r'(?:Based on|Derived from|Full report|Complete study|Working paper)\s*:?\s*([^\.]+)', re.IGNORECASE),
        re.compile(r'(?:See also|For more details|Full version available)\s*:?\s*([^\.]+)', re.IGNORECASE),
    ]
    
    def __init__(self):
        """Initialize the extractor."""
        pass
    
    def extract_front_matter(self, pages: List[PageText]) -> DocumentFrontMatter:
        """
        Extract front matter from document pages.
        
        Args:
            pages: List of page text objects
            
        Returns:
            DocumentFrontMatter object with extracted information
        """
        if not pages:
            return DocumentFrontMatter()
        
        # Focus on first few pages and last page for front matter
        front_pages = pages[:3]
        last_page = pages[-1:] if len(pages) > 3 else []
        relevant_pages = front_pages + last_page
        
        # Combine text for analysis
        combined_text = '\n\n'.join(page.text for page in relevant_pages)
        
        # Extract different types of information
        title = self._extract_title(front_pages)
        authors = self._extract_authors(combined_text)
        affiliations = self._extract_affiliations(combined_text)
        emails = self._extract_emails(combined_text)
        urls = self._extract_urls(combined_text)
        funding_statements = self._extract_funding(combined_text)
        linked_studies = self._extract_linked_studies(combined_text)
        
        return DocumentFrontMatter(
            title=title,
            authors=authors,
            affiliations=affiliations,
            emails=emails,
            urls=urls,
            funding_statements=funding_statements,
            linked_studies=linked_studies
        )
    
    def _extract_title(self, front_pages: List[PageText]) -> Optional[str]:
        """Extract document title from first page."""
        if not front_pages:
            return None
        
        first_page_text = front_pages[0].text.strip()
        if not first_page_text:
            return None
        
        # Look for title in first few lines
        lines = first_page_text.split('\n')[:10]  # First 10 lines only
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip very short lines (likely metadata)
            if len(line) < 10:
                continue
                
            # Skip lines that look like metadata
            if any(keyword in line.lower() for keyword in ['page', 'file', 'pdf', '©', 'copyright']):
                continue
                
            # Title should be substantial but not too long
            if 10 <= len(line) <= 150:
                # Clean up the title
                title = line.strip('.,;:')
                
                # Basic validity check
                if re.match(r'^[A-Za-z]', title) and not title.endswith(' pdf'):
                    return title
        
        return None
    
    def _extract_authors(self, text: str) -> List[str]:
        """Extract author names from text."""
        authors = []
        
        for pattern in self.AUTHOR_SECTION_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    author_text = match[1] if len(match) > 1 else match[0]
                else:
                    author_text = match
                
                # Clean and parse author text
                parsed_authors = self._parse_author_text(author_text)
                authors.extend(parsed_authors)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_authors = []
        for author in authors:
            if author not in seen:
                seen.add(author)
                unique_authors.append(author)
        
        return unique_authors[:10]  # Limit to reasonable number
    
    def _parse_author_text(self, author_text: str) -> List[str]:
        """Parse author names from author section text."""
        if not author_text:
            return []
        
        # Clean the text
        text = author_text.strip()
        
        # Split by common separators
        separators = ['\n', ' and ', ', ', ';']
        authors = [text]
        
        for sep in separators:
            new_authors = []
            for author in authors:
                new_authors.extend(author.split(sep))
            authors = new_authors
        
        # Clean individual author names
        cleaned_authors = []
        for author in authors:
            author = author.strip()
            
            # Skip empty or very short names
            if len(author) < 3:
                continue
                
            # Skip lines that don't look like names
            if any(keyword in author.lower() for keyword in [
                'university', 'institute', 'email', '@', 'phone', 'address',
                'department', 'college', 'center', 'centre', 'foundation'
            ]):
                continue
            
            # Basic name pattern (allow letters, spaces, hyphens, apostrophes)
            if re.match(r'^[A-Za-z\s\-\'\.]+$', author) and len(author) <= 50:
                cleaned_authors.append(author)
        
        return cleaned_authors
    
    def _extract_affiliations(self, text: str) -> List[str]:
        """Extract institutional affiliations."""
        affiliations = []
        
        for pattern in self.AFFILIATION_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                affiliation = match.strip()
                
                # Clean and validate
                if 10 <= len(affiliation) <= 200:
                    affiliations.append(affiliation)
        
        # Remove duplicates
        return list(set(affiliations))[:5]  # Limit to 5 affiliations
    
    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses."""
        matches = self.EMAIL_PATTERN.findall(text)
        
        # Clean and validate emails
        emails = []
        for email in matches:
            email = email.lower().strip()
            
            # Skip obviously invalid emails
            if len(email) > 6 and email.count('@') == 1:
                emails.append(email)
        
        # Remove duplicates and limit
        return list(set(emails))[:10]
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs."""
        matches = self.URL_PATTERN.findall(text)
        
        # Clean URLs
        urls = []
        for url in matches:
            url = url.strip()
            
            # Ensure http prefix
            if url.startswith('www.'):
                url = 'https://' + url
            
            # Basic validation
            if 'http' in url and len(url) <= 200:
                urls.append(url)
        
        # Remove duplicates
        return list(set(urls))[:5]
    
    def _extract_funding(self, text: str) -> List[str]:
        """Extract funding statements."""
        funding_statements = []
        
        for pattern in self.FUNDING_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    funding_text = match[1] if len(match) > 1 and match[1] else match[0]
                else:
                    funding_text = match
                
                funding_text = funding_text.strip()
                
                # Clean and validate
                if 20 <= len(funding_text) <= 800:  # Reasonable length
                    # Clean up whitespace and newlines
                    funding_text = re.sub(r'\s+', ' ', funding_text)
                    funding_statements.append(funding_text)
        
        return funding_statements[:3]  # Limit to 3 statements
    
    def _extract_linked_studies(self, text: str) -> List[str]:
        """Extract references to linked studies or reports."""  
        linked_studies = []
        
        for pattern in self.LINKED_STUDY_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                study_ref = match.strip()
                
                # Clean and validate
                if 10 <= len(study_ref) <= 300:
                    # Clean up whitespace
                    study_ref = re.sub(r'\s+', ' ', study_ref)
                    linked_studies.append(study_ref)
        
        return linked_studies[:3]  # Limit to 3 references