"""
Tests for front-matter extraction from policy brief content.
"""

import pytest
from src.policybrief.models import PageText, DocumentFrontMatter
from src.policybrief.frontmatter_extractor import FrontMatterExtractor


class TestFrontMatterExtractor:
    """Test the FrontMatterExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FrontMatterExtractor()
    
    def test_empty_pages(self):
        """Test extraction with no pages."""
        front_matter = self.extractor.extract_front_matter([])
        
        assert isinstance(front_matter, DocumentFrontMatter)
        assert front_matter.title is None
        assert front_matter.authors == []
        assert front_matter.emails == []
        assert front_matter.urls == []
    
    def test_title_extraction(self):
        """Test title extraction from first page."""
        pages = [
            PageText(
                page_num=1,
                text="Climate Policy Implementation: Challenges and Opportunities\n\nThis is the introduction...",
                char_count=100,
                word_count=20
            )
        ]
        
        front_matter = self.extractor.extract_front_matter(pages)
        
        assert front_matter.title == "Climate Policy Implementation: Challenges and Opportunities"
    
    def test_author_extraction_simple(self):
        """Test simple author extraction."""
        pages = [
            PageText(
                page_num=1,
                text="""Policy Brief: Climate Action
                
Authors:
John Smith
Jane Doe

This report examines...""",
                char_count=100,
                word_count=20
            )
        ]
        
        front_matter = self.extractor.extract_front_matter(pages)
        
        assert "John Smith" in front_matter.authors
        assert "Jane Doe" in front_matter.authors
        assert len(front_matter.authors) == 2
    
    def test_author_extraction_with_and(self):
        """Test author extraction with 'and' separator."""
        pages = [
            PageText(
                page_num=1,
                text="""Climate Policy Brief
                
Written by: John Smith and Jane Doe

Introduction...""",
                char_count=100,
                word_count=20
            )
        ]
        
        front_matter = self.extractor.extract_front_matter(pages)
        
        assert "John Smith" in front_matter.authors
        assert "Jane Doe" in front_matter.authors
    
    def test_email_extraction(self):
        """Test email address extraction."""
        pages = [
            PageText(
                page_num=1,
                text="""Policy Brief
                
Contact: john.smith@university.edu
For questions: jane.doe@institute.org

This research...""",
                char_count=100,
                word_count=20
            )
        ]
        
        front_matter = self.extractor.extract_front_matter(pages)
        
        assert "john.smith@university.edu" in front_matter.emails
        assert "jane.doe@institute.org" in front_matter.emails
        assert len(front_matter.emails) == 2
    
    def test_url_extraction(self):
        """Test URL extraction."""
        pages = [
            PageText(
                page_num=1,
                text="""Policy Brief
                
More info: https://www.example.org/study
Website: www.policyinstitute.net

This report...""",
                char_count=100,
                word_count=20
            )
        ]
        
        front_matter = self.extractor.extract_front_matter(pages)
        
        assert "https://www.example.org/study" in front_matter.urls
        assert "https://www.policyinstitute.net" in front_matter.urls
    
    def test_affiliation_extraction(self):
        """Test institutional affiliation extraction."""
        pages = [
            PageText(
                page_num=1,
                text="""Climate Policy Analysis
                
John Smith, Climate Research Institute
Jane Doe, University of Environmental Sciences

This work...""",
                char_count=100,
                word_count=20
            )
        ]
        
        front_matter = self.extractor.extract_front_matter(pages)
        
        # Should find at least one affiliation
        assert len(front_matter.affiliations) > 0
        assert any("Institute" in aff for aff in front_matter.affiliations)
    
    def test_funding_extraction(self):
        """Test funding statement extraction."""
        pages = [
            PageText(
                page_num=1,
                text="""Policy Brief
                
Main content here...

Acknowledgments

This research was supported by the National Science Foundation 
grant #12345 and the Climate Research Foundation.

References...""",
                char_count=200,
                word_count=40
            )
        ]
        
        front_matter = self.extractor.extract_front_matter(pages)
        
        assert len(front_matter.funding_statements) > 0
        funding_text = " ".join(front_matter.funding_statements)
        assert "National Science Foundation" in funding_text
    
    def test_linked_studies_extraction(self):
        """Test extraction of linked study references."""
        pages = [
            PageText(
                page_num=1,
                text="""Policy Brief Summary
                
This is a summary of the full report "Comprehensive Climate Policy Analysis 2024"
available at our website.

Based on: "Economic Impacts of Carbon Pricing" working paper.

For more details see the complete study in Climate Policy Journal.""",
                char_count=200,
                word_count=40
            )
        ]
        
        front_matter = self.extractor.extract_front_matter(pages)
        
        assert len(front_matter.linked_studies) > 0
        studies_text = " ".join(front_matter.linked_studies)
        assert "Comprehensive Climate Policy Analysis" in studies_text or "Economic Impacts" in studies_text
    
    def test_no_false_positives(self):
        """Test that extractor doesn't create false positive data."""
        pages = [
            PageText(
                page_num=1,
                text="""A simple document with no special front matter.
                
This is just regular content without authors sections,
emails, or special metadata.

Just normal text paragraphs.""",
                char_count=150,
                word_count=30
            )
        ]
        
        front_matter = self.extractor.extract_front_matter(pages)
        
        # Should have minimal or no extraction for plain text
        assert len(front_matter.authors) == 0
        assert len(front_matter.emails) == 0
        assert len(front_matter.funding_statements) == 0
        assert len(front_matter.linked_studies) == 0
    
    def test_title_cleanup(self):
        """Test title cleanup and validation."""
        pages = [
            PageText(
                page_num=1,
                text="""Climate Policy Brief:  Implementation Challenges...
                
Content starts here...""",
                char_count=100,
                word_count=20
            )
        ]
        
        front_matter = self.extractor.extract_front_matter(pages)
        
        # Title should be cleaned of trailing punctuation
        assert front_matter.title == "Climate Policy Brief:  Implementation Challenges"
    
    def test_duplicate_removal(self):
        """Test that duplicates are removed from lists."""
        pages = [
            PageText(
                page_num=1,
                text="""Policy Brief
                
Authors: John Smith, Jane Doe
Written by: John Smith and Jane Doe

Contact: john.smith@example.com
Email: john.smith@example.com""",
                char_count=150,
                word_count=30
            )
        ]
        
        front_matter = self.extractor.extract_front_matter(pages)
        
        # Should not have duplicate authors or emails
        assert len(front_matter.authors) == 2  # John Smith and Jane Doe
        assert len(front_matter.emails) == 1   # john.smith@example.com (deduplicated)
    
    def test_last_page_extraction(self):
        """Test that back matter on last page is also extracted."""
        pages = [
            PageText(page_num=1, text="Title Page", char_count=10, word_count=2),
            PageText(page_num=2, text="Content page", char_count=12, word_count=2),
            PageText(page_num=3, text="More content", char_count=12, word_count=2),
            PageText(page_num=4, text="Last page", char_count=9, word_count=2),
            PageText(
                page_num=5,
                text="""Contact Information
                
For questions: contact@institute.org
Website: https://www.institute.org""",
                char_count=80,
                word_count=15
            )
        ]
        
        front_matter = self.extractor.extract_front_matter(pages)
        
        # Should pick up contact info from last page
        assert "contact@institute.org" in front_matter.emails
        assert "https://www.institute.org" in front_matter.urls
    
    def test_content_over_metadata_preference(self):
        """Test that content extraction provides alternative to metadata."""
        # This test validates that we can extract content-based information
        # that would supplement/override PDF metadata
        pages = [
            PageText(
                page_num=1,
                text="""Comprehensive Climate Policy Assessment
                
Authors:
Dr. Sarah Johnson, Climate Policy Institute
Prof. Michael Chen, University of Environmental Studies

Contact: s.johnson@climatepolicy.org

This comprehensive study examines...""",
                char_count=200,
                word_count=40
            )
        ]
        
        front_matter = self.extractor.extract_front_matter(pages)
        
        # Should extract content-based title and authors
        assert front_matter.title == "Comprehensive Climate Policy Assessment"
        assert len(front_matter.authors) == 2
        assert "Dr. Sarah Johnson" in front_matter.authors
        assert "Prof. Michael Chen" in front_matter.authors
        assert "s.johnson@climatepolicy.org" in front_matter.emails
        assert len(front_matter.affiliations) >= 1  # Should find institute/university