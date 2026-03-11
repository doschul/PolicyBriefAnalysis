"""
OpenAI LLM client with structured outputs and retry logic.

Handles all OpenAI API interactions with proper JSON schema validation and error handling.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Type, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential
)

from .models import (
    FrameDetectionInput,
    FrameDetectionOutput, 
    RecommendationExtractionOutput
)


logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """OpenAI client with structured outputs and retry logic."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: OpenAI API key (will use env var if None)
            model: Model name (must support structured outputs)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries
        """
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Validate model supports structured outputs
        if not self._supports_structured_outputs(model):
            logger.warning(f"Model {model} may not support structured outputs")
    
    def _supports_structured_outputs(self, model: str) -> bool:
        """Check if model supports structured outputs."""
        supported_models = [
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-11-20"
        ]
        return any(supported in model for supported in supported_models)
    
    @retry(
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def _make_api_call(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make API call with retry logic for transient errors."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            
            # Use correct token parameter based on model
            if "gpt-5" in self.model.lower():
                kwargs["max_completion_tokens"] = self.max_tokens
            else:
                kwargs["max_tokens"] = self.max_tokens
            
            if response_format:
                kwargs["response_format"] = response_format
            
            response = self.client.chat.completions.create(**kwargs)
            
            return {
                "content": response.choices[0].message.content,
                "usage": response.usage.model_dump() if response.usage else {},
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
    
    def structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        max_validation_retries: int = 2
    ) -> T:
        """
        Get structured completion with JSON schema validation.
        
        Args:
            messages: Chat messages 
            response_model: Pydantic model for response validation
            max_validation_retries: Retries for validation failures
            
        Returns:
            Validated response model instance
            
        Raises:
            ValidationError: If response doesn't match schema after retries
            Exception: If API call fails after retries
        """
        # Generate JSON schema from Pydantic model
        json_schema = response_model.model_json_schema()
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": json_schema,
                "strict": True
            }
        }
        
        validation_attempts = 0
        
        while validation_attempts <= max_validation_retries:
            try:
                # Make API call
                response = self._make_api_call(messages, response_format)
                
                if not response["content"]:
                    raise ValueError("Empty response content")
                
                # Parse and validate JSON
                try:
                    json_data = json.loads(response["content"])
                except json.JSONDecodeError as e:
                    raise ValidationError(f"Invalid JSON response: {e}")
                
                # Validate against Pydantic model
                return response_model.model_validate(json_data)
                
            except ValidationError as e:
                validation_attempts += 1
                logger.warning(f"Validation failed (attempt {validation_attempts}): {e}")
                
                if validation_attempts > max_validation_retries:
                    logger.error(f"Max validation retries exceeded for {response_model.__name__}")
                    raise
                
                # Add error feedback for retry
                error_message = {
                    "role": "system",
                    "content": f"Previous response failed validation: {str(e)}. Please provide a valid JSON response matching the required schema."
                }
                messages.append(error_message)
                
                time.sleep(self.retry_delay)
        
        raise RuntimeError("Unexpected validation retry loop exit")
    
    def detect_frame(
        self,
        frame_input: FrameDetectionInput,
        frame_definition: str
    ) -> FrameDetectionOutput:
        """
        Detect theoretical frame presence in document text.
        
        Args:
            frame_input: Input data for frame detection
            frame_definition: Full frame definition from config
            
        Returns:
            Structured frame assessment
        """
        logger.debug(f"Detecting frame: {frame_input.frame_id}")
        
        # Prepare text spans for analysis
        spans_text = ""
        for i, span in enumerate(frame_input.text_spans):
            spans_text += f"\n--- Text Span {i+1} (Page {span.get('page', '?')}) ---\n"
            if span.get('section_heading'):
                spans_text += f"Section: {span['section_heading']}\n"
            spans_text += f"{span.get('text', '')}\n"
        
        system_message = """You are a policy analysis expert tasked with detecting theoretical frameworks in policy documents. 

Your job is to determine if a specific theoretical frame is present in the given text spans.

INSTRUCTIONS:
1. Read the frame definition carefully
2. Analyze the provided text spans for evidence of the frame
3. Return your assessment using ONLY the requested JSON format
4. For 'present' decisions, you MUST provide exact verbatim quotes as evidence
5. Quotes must be exact substrings from the provided text
6. Include page numbers with all evidence quotes
7. Be conservative - only mark as 'present' if there is clear, strong evidence

DECISION CRITERIA:
- present: Clear, strong evidence with multiple supporting quotes
- absent: No meaningful evidence found
- insufficient_evidence: Some weak indicators but not enough for confident assessment"""
        
        user_message = f"""FRAME TO DETECT: {frame_input.frame_id}

FRAME DEFINITION:
{frame_definition}

TEXT SPANS TO ANALYZE:
{spans_text}

Analyze these text spans and determine if the theoretical frame is present. Provide exact verbatim quotes as evidence."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return self.structured_completion(messages, FrameDetectionOutput)
    
    def extract_recommendations(
        self,
        document_text: str,
        page_info: List[Dict[str, Any]],
        max_recommendations: int = 10
    ) -> RecommendationExtractionOutput:
        """
        Extract policy recommendations from document.
        
        Args:
            document_text: Full document text
            page_info: Page number mapping for quotes
            max_recommendations: Maximum recommendations to extract
            
        Returns:
            Structured recommendations with evidence
        """
        logger.debug(f"Extracting recommendations from document ({len(document_text)} chars)")
        
        system_message = f"""You are a policy analysis expert specializing in extracting policy recommendations from documents.

Your task is to identify and structure policy recommendations with precise evidence.

INSTRUCTIONS:
1. Find clear, actionable policy recommendations in the text
2. Extract up to {max_recommendations} recommendations
3. For each recommendation, provide exact verbatim quotes as evidence
4. Include page numbers for all evidence quotes  
5. Classify recommendations using the provided enums
6. Focus on concrete, implementable actions

WHAT COUNTS AS A RECOMMENDATION:
- Clear actions that should/must/could be taken
- Specific policy instruments or measures
- Actionable steps for implementation
- Concrete proposals for change

EVIDENCE REQUIREMENTS:
- Quotes must be exact substrings from the document
- Include page number for each quote
- Provide sufficient evidence for each recommendation"""
        
        # Prepare text with page markers for reference
        pages_with_markers = []
        for i, page_data in enumerate(page_info):
            page_num = page_data.get('page_num', i + 1)
            text = page_data.get('text', '')
            pages_with_markers.append(f"\n--- PAGE {page_num} ---\n{text}")
        
        full_text_with_pages = "\n".join(pages_with_markers)
        
        user_message = f"""DOCUMENT TEXT:
{full_text_with_pages}

Extract policy recommendations from this document. Focus on clear, actionable recommendations with strong evidence."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        return self.structured_completion(messages, RecommendationExtractionOutput)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get accumulated usage statistics."""
        # This would be implemented with proper usage tracking
        # For now, return placeholder
        return {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0
        }