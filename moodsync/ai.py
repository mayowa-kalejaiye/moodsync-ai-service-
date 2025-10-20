import json
import logging
from typing import Tuple, Any

import google.generativeai as genai

from ..config import GEMINI_API_KEY
from .resilience import intelligent_retry, RetryConfig, CircuitBreaker, AIServiceError

logger = logging.getLogger(__name__)

retry_config = RetryConfig()
gemini_circuit_breaker = CircuitBreaker()


@intelligent_retry(config=retry_config, circuit_breaker=gemini_circuit_breaker)
def _call_gemini_api(prompt_text: str, model_name: str = 'gemini-2.0-flash-exp') -> str:
    if not GEMINI_API_KEY:
        raise AIServiceError(
            "AI service not configured. Missing API key.",
            error_type="configuration",
            is_retryable=False
        )

    try:
        logger.debug(f"Calling Gemini API with model: {model_name}")
        model_instance = genai.GenerativeModel(model_name)
        response = model_instance.generate_content(
            prompt_text,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
                top_p=0.8,
                top_k=40
            )
        )

        if not response.parts:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                raise AIServiceError(
                    f"Content generation blocked by Gemini: {response.prompt_feedback.block_reason_message}",
                    error_type="content_filtered",
                    is_retryable=False
                )
            else:
                raise AIServiceError(
                    "AI model did not return content - response parts empty",
                    error_type="empty_response",
                    is_retryable=True
                )

        generated_text = response.text
        if not generated_text or len(generated_text.strip()) == 0:
            raise AIServiceError(
                "AI model returned empty text content",
                error_type="empty_content",
                is_retryable=True
            )

        return generated_text

    except Exception as e:
        if isinstance(e, AIServiceError):
            raise e

        error_str = str(e).lower()

        if "quota" in error_str or "rate limit" in error_str:
            raise AIServiceError(f"Rate limit or quota exceeded: {str(e)}", error_type="rate_limit", is_retryable=True)
        elif "timeout" in error_str:
            raise AIServiceError(f"Request timeout: {str(e)}", error_type="timeout", is_retryable=True)
        elif "connection" in error_str or "network" in error_str:
            raise AIServiceError(f"Network error: {str(e)}", error_type="network", is_retryable=True)
        elif "api key" in error_str or "authentication" in error_str:
            raise AIServiceError(f"Authentication error: {str(e)}", error_type="authentication", is_retryable=False)
        elif "model" in error_str and "not found" in error_str:
            raise AIServiceError(f"Model not found: {str(e)}", error_type="model_not_found", is_retryable=False)
        else:
            raise AIServiceError(f"Unknown AI service error: {str(e)}", error_type="unknown", is_retryable=True)


def generate_gemini_response(prompt_text: str, is_json_output: bool = False) -> Tuple[Any, int]:
    logger.debug(f"Attempting to generate content with Gemini. JSON output expected: {is_json_output}")
    logger.debug(f"Prompt for Gemini:\n---\n{prompt_text}\n---")

    try:
        generated_text = _call_gemini_api(prompt_text)
        logger.info(f"Successfully generated content from Gemini ({len(generated_text)} characters)")

        if is_json_output:
            try:
                parsed_json = json.loads(generated_text)
                logger.info(f"Successfully parsed Gemini JSON output.")
                return parsed_json, 200
            except json.JSONDecodeError as e:
                logger.warning(f"Expected JSON output but received non-JSON: {generated_text}. Error: {e}")
                lines = [s.strip() for s in generated_text.split('\n') if s.strip() and len(s.strip()) > 10]
                if lines:
                    logger.info("Extracted text lines as fallback for JSON parsing failure")
                    return lines[:3], 200
                else:
                    logger.warning("Unable to extract meaningful content from malformed JSON response")
                    return ["Unable to generate suggestions at this time."], 200
        else:
            return {"text": generated_text}, 200

    except AIServiceError as e:
        logger.error(f"AI Service Error: {e.error_type} - {str(e)}")
        if e.error_type == "configuration":
            return {"error": "AI service not configured. Missing API key."}, 503
        elif e.error_type == "authentication":
            return {"error": "AI service authentication failed. Check API key."}, 401
        elif e.error_type == "rate_limit":
            return {"error": "AI service rate limit exceeded. Please try again later."}, 429
        elif e.error_type == "content_filtered":
            if is_json_output:
                return ["I'm here to help with positive suggestions for your wellbeing."], 200
            else:
                return {"text": "I'm here to support your mental health journey in a positive way."}, 200
        else:
            logger.error(f"Providing fallback response due to AI service error: {str(e)}")
            if is_json_output:
                return ["Unable to generate AI suggestions at this time. Please try again later."], 200
            else:
                return {"text": "Unable to generate AI response at this time. Please try again later."}, 200

    except Exception as e:
        logger.error(f"Unexpected error in generate_gemini_response: {str(e)}", exc_info=True)
        if is_json_output:
            return ["An unexpected error occurred. Please try again later."], 200
        else:
            return {"text": "An unexpected error occurred. Please try again later."}, 200
