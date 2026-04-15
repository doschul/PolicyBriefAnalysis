"""
OpenAI LLM client with structured outputs, schema patching, and retry logic.
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
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


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
        retry_delay: float = 1.0,
    ):
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ── Schema patching ───────────────────────────────────────────────

    @staticmethod
    def _patch_schema_required(schema: Dict[str, Any]) -> None:
        """Make a Pydantic V2 JSON schema compatible with OpenAI strict mode.

        Mutates *schema* in place:
        1. All properties listed in ``required``.
        2. ``additionalProperties: false`` on every object.
        3. Sibling keywords stripped from ``$ref`` nodes.
        """
        if not isinstance(schema, dict):
            return

        for def_schema in schema.get("$defs", {}).values():
            LLMClient._patch_schema_required(def_schema)

        if "$ref" in schema:
            for key in list(schema.keys()):
                if key != "$ref":
                    del schema[key]
            return

        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
            schema["additionalProperties"] = False
            for prop_schema in schema["properties"].values():
                LLMClient._patch_schema_required(prop_schema)

        for key in ("items", "additionalProperties"):
            if isinstance(schema.get(key), dict):
                LLMClient._patch_schema_required(schema[key])

        for key in ("anyOf", "allOf", "oneOf"):
            for sub in schema.get(key, []):
                if isinstance(sub, dict):
                    LLMClient._patch_schema_required(sub)

    # ── API call with retry ───────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIConnectionError,
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def _make_api_call(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        return {
            "content": response.choices[0].message.content,
            "usage": response.usage.model_dump() if response.usage else {},
            "finish_reason": response.choices[0].finish_reason,
        }

    # ── Structured completion ─────────────────────────────────────────

    def structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        max_validation_retries: int = 2,
    ) -> T:
        """Get a structured completion validated against *response_model*."""
        json_schema = response_model.model_json_schema()
        self._patch_schema_required(json_schema)

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": json_schema,
                "strict": True,
            },
        }

        attempts = 0
        while attempts <= max_validation_retries:
            response = self._make_api_call(messages, response_format)
            if not response["content"]:
                raise ValueError("Empty response content")
            try:
                data = json.loads(response["content"])
                return response_model.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                attempts += 1
                logger.warning(f"Validation attempt {attempts} failed: {exc}")
                if attempts > max_validation_retries:
                    raise
                messages = messages + [
                    {
                        "role": "system",
                        "content": (
                            f"Previous response failed validation: {exc}. "
                            "Please provide a valid JSON response."
                        ),
                    }
                ]
                time.sleep(self.retry_delay)

        raise RuntimeError("Unexpected validation retry loop exit")
