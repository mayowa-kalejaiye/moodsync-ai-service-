import os
import json
import logging
import random
import threading
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any, Callable, Union
from contextlib import asynccontextmanager
from functools import wraps
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import google.generativeai as genai

# Load environment variables
load_dotenv()

# --- Retry Configuration ---
class RetryConfig:
    """Configuration for retry logic"""
    def __init__(self):
        self.max_retries = int(os.environ.get("AI_MAX_RETRIES", "3"))
        self.base_delay = float(os.environ.get("AI_BASE_DELAY", "1.0"))
        self.max_delay = float(os.environ.get("AI_MAX_DELAY", "30.0"))
        self.jitter_max = float(os.environ.get("AI_JITTER_MAX", "0.5"))
        self.exponential_base = float(os.environ.get("AI_EXPONENTIAL_BASE", "2.0"))
        
class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing if service is back

class CircuitBreaker:
    """Circuit breaker pattern implementation for AI service calls"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, success_threshold: int = 2):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.logger = logging.getLogger(__name__ + '.CircuitBreaker')
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker transitioning to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - failing fast")
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
        
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.success_count = 0
                self.logger.info("Circuit breaker transitioning to CLOSED state")
                
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning("Circuit breaker transitioning back to OPEN state")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.error(f"Circuit breaker tripped! Transitioning to OPEN state after {self.failure_count} failures")

class AIServiceError(Exception):
    """Custom exception for AI service errors"""
    def __init__(self, message: str, error_type: str = "unknown", is_retryable: bool = True):
        super().__init__(message)
        self.error_type = error_type
        self.is_retryable = is_retryable

def intelligent_retry(config: RetryConfig = None, circuit_breaker: CircuitBreaker = None):
    """
    Intelligent retry decorator with exponential backoff, jitter, and circuit breaker support
    
    Args:
        config: RetryConfig instance with retry parameters
        circuit_breaker: CircuitBreaker instance for fault tolerance
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            retry_logger = logging.getLogger(f"{__name__}.retry.{func.__name__}")
            
            for attempt in range(config.max_retries + 1):  # +1 for initial attempt
                try:
                    # Use circuit breaker if provided
                    if circuit_breaker:
                        return circuit_breaker.call(func, *args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except Exception as e:
                    last_exception = e
                    
                    # Determine if error is retryable
                    is_retryable = True
                    error_type = "unknown"
                    
                    if isinstance(e, AIServiceError):
                        is_retryable = e.is_retryable
                        error_type = e.error_type
                    elif "quota" in str(e).lower() or "rate limit" in str(e).lower():
                        error_type = "rate_limit"
                        is_retryable = True
                    elif "api key" in str(e).lower() or "authentication" in str(e).lower():
                        error_type = "authentication" 
                        is_retryable = False
                    elif "timeout" in str(e).lower() or "connection" in str(e).lower():
                        error_type = "network"
                        is_retryable = True
                    elif "model" in str(e).lower() and "not found" in str(e).lower():
                        error_type = "model_not_found"
                        is_retryable = False
                    
                    # Log the attempt
                    if attempt == 0:
                        retry_logger.warning(f"Initial attempt failed: {error_type} - {str(e)}")
                    else:
                        retry_logger.warning(f"Retry attempt {attempt}/{config.max_retries} failed: {error_type} - {str(e)}")
                    
                    # Don't retry if error is not retryable or we've exhausted attempts
                    if not is_retryable or attempt >= config.max_retries:
                        retry_logger.error(f"Giving up after {attempt + 1} attempts. Last error: {str(e)}")
                        break
                    
                    # Calculate delay with exponential backoff and jitter
                    base_delay = config.base_delay * (config.exponential_base ** attempt)
                    capped_delay = min(base_delay, config.max_delay)
                    jitter = random.uniform(0, config.jitter_max)
                    total_delay = capped_delay + jitter
                    
                    retry_logger.info(f"Retrying in {total_delay:.2f}s (base: {capped_delay:.2f}s + jitter: {jitter:.2f}s)")
                    time.sleep(total_delay)
            
            # If we get here, all attempts failed
            if isinstance(last_exception, AIServiceError):
                raise last_exception
            else:
                raise AIServiceError(
                    f"AI service call failed after {config.max_retries + 1} attempts: {str(last_exception)}",
                    error_type=error_type,
                    is_retryable=False
                )
        
        return wrapper
    return decorator

# --- Setup Logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- App Configuration & Globals ---
AI_REQUEST_TIMEOUT = 30
KEEP_ALIVE_ENABLED = os.environ.get("FLASK_ENV", "production") != "development"
KEEP_ALIVE_INTERVAL = 10 * 60
KEEP_ALIVE_URL = os.environ.get("KEEP_ALIVE_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize retry configuration and circuit breaker
retry_config = RetryConfig()
gemini_circuit_breaker = CircuitBreaker(
    failure_threshold=int(os.environ.get("AI_CIRCUIT_BREAKER_THRESHOLD", "5")),
    recovery_timeout=int(os.environ.get("AI_CIRCUIT_BREAKER_TIMEOUT", "60")),
    success_threshold=int(os.environ.get("AI_CIRCUIT_BREAKER_SUCCESS_THRESHOLD", "2"))
)

# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application starting up...")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured successfully.")
    else:
        logger.error("CRITICAL: GEMINI_API_KEY not found. AI suggestions will fail.")
    
    if os.environ.get("FLASK_ENV", "production") != "development":
        threading.Timer(30.0, start_keep_alive_service).start()
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")
    keep_alive_service.stop()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="MoodSync AI Service",
    description="AI-powered service for generating motivational messages, habit suggestions, and insights.",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class MotivationRequest(BaseModel):
    user_name: str = 'User'
    mood_trend_label: str = 'neutral'
    recent_mood_texts: List[str] = []
    recent_notes_texts: List[str] = []
    user_age: Optional[int] = None

class HabitRequest(BaseModel):
    user_name: str = 'User'
    low_mood_activities: List[str] = []
    high_mood_activities: List[str] = []

class NudgeContext(BaseModel):
    user_age: str = 'unknown'
    streak_count: int = 0
    coin_balance: int = 0
    tone: str = 'professional'
    recent_moods: List[Dict[str, Any]] = []

class NudgeRequest(BaseModel):
    context: NudgeContext
    max_length: int = 150

class ChallengeContext(BaseModel):
    challenge_type: str = 'unknown'
    stake: int = 0
    duration_days: int = 0
    user_age: str = 'unknown'
    completed: bool = False
    tone: str = 'professional'

class ChallengeMessageRequest(BaseModel):
    context: ChallengeContext
    max_length: int = 200

class InsightsContext(BaseModel):
    user_age: str = 'unknown'
    analysis_period: int = 30
    mood_data: List[Dict[str, Any]] = []
    avg_rating: float = 3.0
    total_entries: int = 0
    streak_count: int = 0
    tone: str = 'professional'

class InsightsRequest(BaseModel):
    context: InsightsContext

# Response Models
class StandardResponse(BaseModel):
    text: str

class HabitResponse(BaseModel):
    suggestions: List[str]

class NudgeResponse(BaseModel):
    message: str
    tone: str

class ChallengeResponse(BaseModel):
    message: str
    tone: str
    completed: bool

class InsightsResponse(BaseModel):
    insights: List[str]
    recommendations: List[str]
    mood_trends: str

# --- Root Endpoint ---
@app.get('/', tags=["Health"])
def read_root():
    """Root endpoint providing a welcome message and documentation links."""
    return {
        "message": "Welcome to the MoodSync AI Service!",
        "version": "1.0.0",
        "framework": "FastAPI",
        "documentation": {
            "interactive_docs": "/docs",
            "openapi_schema": "/openapi.json",
            "health_check": "/health"
        },
        "endpoints": {
            "motivation": "/motivation",
            "habits": "/habits", 
            "nudges": "/generate-nudge",
            "challenges": "/generate-challenge-message",
            "insights": "/generate-insights"
        }
    }

# --- Health Check Endpoint ---
@app.get('/health', tags=["Health"])
def health_check():
    """Health check endpoint providing service status and resilience information."""
    return {
        "status": "healthy",
        "service": "MoodSync AI Service",
        "version": "1.0.0",
        "endpoints": [
            "/motivation", 
            "/habits",
            "/generate-nudge",
            "/generate-challenge-message",
            "/generate-insights"
        ],
        "gemini_configured": bool(GEMINI_API_KEY),
        "keep_alive_enabled": KEEP_ALIVE_ENABLED,
        "resilience": {
            "circuit_breaker": {
                "state": gemini_circuit_breaker.state.value,
                "failure_count": gemini_circuit_breaker.failure_count,
                "failure_threshold": gemini_circuit_breaker.failure_threshold,
                "success_count": gemini_circuit_breaker.success_count if gemini_circuit_breaker.state == CircuitBreakerState.HALF_OPEN else 0
            },
            "retry_config": {
                "max_retries": retry_config.max_retries,
                "base_delay": retry_config.base_delay,
                "max_delay": retry_config.max_delay,
                "exponential_base": retry_config.exponential_base
            }
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get('/api/health/', tags=["Health"])
def api_health_check():
    """API health check endpoint for keep-alive service."""
    return {
        "status": "healthy",
        "service": "MoodSync AI Service",
        "timestamp": datetime.now().isoformat(),
        "uptime": "running"
    }

# --- Helper function to call Gemini ---
@intelligent_retry(config=retry_config, circuit_breaker=gemini_circuit_breaker)
def _call_gemini_api(prompt_text: str, model_name: str = 'gemini-2.0-flash-exp') -> str:
    """
    Core Gemini API call with error classification
    Separated from response processing for cleaner retry logic
    """
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
            # Check for content filtering
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
        
        # Classify different types of errors
        if "quota" in error_str or "rate limit" in error_str:
            raise AIServiceError(
                f"Rate limit or quota exceeded: {str(e)}",
                error_type="rate_limit",
                is_retryable=True
            )
        elif "timeout" in error_str:
            raise AIServiceError(
                f"Request timeout: {str(e)}",
                error_type="timeout",
                is_retryable=True
            )
        elif "connection" in error_str or "network" in error_str:
            raise AIServiceError(
                f"Network error: {str(e)}",
                error_type="network",
                is_retryable=True
            )
        elif "api key" in error_str or "authentication" in error_str:
            raise AIServiceError(
                f"Authentication error: {str(e)}",
                error_type="authentication", 
                is_retryable=False
            )
        elif "model" in error_str and "not found" in error_str:
            raise AIServiceError(
                f"Model not found: {str(e)}",
                error_type="model_not_found",
                is_retryable=False
            )
        else:
            # Unknown error - treat as potentially retryable
            raise AIServiceError(
                f"Unknown AI service error: {str(e)}",
                error_type="unknown",
                is_retryable=True
            )

def generate_gemini_response(prompt_text, is_json_output=False):
    """
    Generate response from Gemini AI with intelligent retry and fallback handling
    """
    logger.debug(f"Attempting to generate content with Gemini. JSON output expected: {is_json_output}")
    logger.debug(f"Prompt for Gemini:\n---\n{prompt_text}\n---")
    
    try:
        # Call Gemini API with retry logic
        generated_text = _call_gemini_api(prompt_text)
        logger.info(f"Successfully generated content from Gemini ({len(generated_text)} characters)")
        
        # Process the response based on expected output type
        if is_json_output:
            try:
                parsed_json = json.loads(generated_text)
                logger.info(f"Successfully parsed Gemini JSON output.")
                return parsed_json, 200
            except json.JSONDecodeError as e:
                logger.warning(f"Expected JSON output but received non-JSON: {generated_text}. Error: {e}")
                # Try to extract meaningful content as fallback
                lines = [s.strip() for s in generated_text.split('\n') if s.strip() and len(s.strip()) > 10]
                if lines:
                    logger.info("Extracted text lines as fallback for JSON parsing failure")
                    return lines[:3], 200
                else:
                    logger.warning("Unable to extract meaningful content from malformed JSON response")
                    return ["Unable to generate suggestions at this time."], 200
        else:
            # Always return the proper format for non-JSON responses
            return {"text": generated_text}, 200
            
    except AIServiceError as e:
        logger.error(f"AI Service Error: {e.error_type} - {str(e)}")
        
        # Return appropriate error responses based on error type
        if e.error_type == "configuration":
            return {"error": "AI service not configured. Missing API key."}, 503
        elif e.error_type == "authentication":
            return {"error": "AI service authentication failed. Check API key."}, 401
        elif e.error_type == "rate_limit":
            return {"error": "AI service rate limit exceeded. Please try again later."}, 429
        elif e.error_type == "content_filtered":
            # For content filtering, provide a gentle fallback
            if is_json_output:
                return ["I'm here to help with positive suggestions for your wellbeing."], 200
            else:
                return {"text": "I'm here to support your mental health journey in a positive way."}, 200
        else:
            # For other errors, provide fallback responses
            logger.error(f"Providing fallback response due to AI service error: {str(e)}")
            if is_json_output:
                return ["Unable to generate AI suggestions at this time. Please try again later."], 200
            else:
                return {"text": "Unable to generate AI response at this time. Please try again later."}, 200
                
    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error in generate_gemini_response: {str(e)}", exc_info=True)
        
        if is_json_output:
            return ["An unexpected error occurred. Please try again later."], 200
        else:
            return {"text": "An unexpected error occurred. Please try again later."}, 200

# --- Motivation Endpoint ---
@app.post('/motivation', response_model=StandardResponse, tags=["AI Service"])
def get_motivation(data: MotivationRequest):
    logger.info("Received request for /motivation endpoint.")
    logger.debug(f"Motivation request payload: {data.dict()}")

    user_name = data.user_name
    mood_trend_label = data.mood_trend_label
    recent_mood_texts = data.recent_mood_texts
    recent_notes_texts = data.recent_notes_texts
    user_age = data.user_age

    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        genz_time = random.choice([
            "chill morning", "vibey morning", "soft sunrise", "slow-mo morning", "cozy morning", "no rush morning", "just-woke-up vibes"
        ])
        mature_time = random.choice([
            "peaceful morning", "productive morning", "early start", "quiet morning"
        ])
    elif 12 <= current_hour < 18:
        genz_time = random.choice([
            "mid afternoon", "pretty chill afternoon", "laid-back afternoon", "solid afternoon", "golden hour", "snack o'clock", "energy dip zone"
        ])
        mature_time = random.choice([
            "afternoon", "busy afternoon", "steady afternoon", "sunny afternoon"
        ])
    else:
        genz_time = random.choice([
            "pretty chill evening", "cool evening", "low-key evening", "mid evening", "comfy night", "Netflix o'clock", "scroll-and-chill time"
        ])
        mature_time = random.choice([
            "relaxing evening", "quiet evening", "peaceful night", "unwind time"
        ])

    if user_age is not None:
        try:
            age = int(user_age)
        except Exception:
            age = None
    else:
        age = None

    if age is not None and age >= 30:
        time_of_day_phrase = mature_time
        style_instructions = (
            "Use a warm, supportive, and mature tone. Avoid slang and keep it professional yet friendly. "
            "No emojis unless it feels natural. Focus on actionable, realistic encouragement."
        )
        prompt_examples = (
            f"- For stress: 'Hi {user_name}, evenings can be a good time to decompress. Consider a short walk or reading to unwind.'\n"
            f"- For positive: 'Glad to hear you're feeling good this {mature_time}. Keep nurturing those positive habits.'\n"
            f"- For neutral: 'It's perfectly fine to have a {mature_time}. Maybe take a moment for yourself or connect with a friend.'\n"
        )
    else:
        time_of_day_phrase = genz_time
        style_instructions = (
            "Use a friendly, modern, Gen Z style—think real words, not clinical or generic. "
            "Sometimes use emojis if it fits naturally (but don't overdo it). "
            "If it fits, use a little humor, meme reference, or pop culture nod (but don't force it)."
        )
        prompt_examples = (
            f"- For anxiety: 'Hey {user_name}, sounds like anxiety's creeping in. Try a quick breathing reset—inhale for 4, hold for 4, exhale for 4. You got this. 💪'\n"
            f"- For positive: 'Yo {user_name}, loving that {mood_trend_label} vibe! Keep the energy up and maybe treat yourself to something fun this {genz_time}. ✨'\n"
            f"- For neutral: 'Sup {user_name}, it's all good to have a {genz_time}. Maybe just chill and recharge for a bit. No pressure, just vibes. 😌'\n"
            f"- For tired: 'Hey {user_name}, if you need a nap, take it. Self-care is not a crime. 💤'\n"
            f"- For stress: 'Hey {user_name}, stress is real. Maybe put your phone down for a sec and take a walk, or just vibe to your favorite song.'\n"
            f"- For low-key: 'Hey {user_name}, it's a low-key {genz_time}. Sometimes that's exactly what you need. Treat yourself to a snack or a meme scroll.'\n"
        )

    prompt_context = f"User: {user_name}. Time of day: {time_of_day_phrase}.\n"
    prompt_context += f"Recent mood trend: {mood_trend_label}.\n"
    if recent_mood_texts:
        prompt_context += f"Recent logged moods: {', '.join(recent_mood_texts)}.\n"
    if recent_notes_texts:
        notes_str = "; ".join(filter(None, recent_notes_texts))
        if notes_str:
            prompt_context += f"Recent notes: \"{notes_str}\".\n"

    has_anxiety = any('anxious' in text.lower() or 'anxiety' in text.lower() for text in recent_mood_texts + recent_notes_texts)
    has_stress = any('stress' in text.lower() for text in recent_notes_texts)

    if has_anxiety:
        prompt_context += "The user has expressed feelings of anxiety.\n"
    if has_stress:
        prompt_context += "The user has mentioned stress.\n"

    prompt = (
        f"{prompt_context}"
        "You are a supportive, relatable, and empathetic AI companion for a mood tracking app. "
        f"{style_instructions} "
        "Based on the user's context, provide a short (2-3 sentences), personalized, and actionable motivational message. "
        "If anxiety or stress is indicated, gently offer a brief, practical coping strategy. "
        "Keep it positive and encouraging, but avoid platitudes or toxic positivity. "
        "Make it sound like a caring friend offering support, not a robot. Do not refer to yourself as an AI. "
        "Examples:\n"
        f"{prompt_examples}"
        "Focus on the provided context and use the time-of-day phrase for a natural, relatable touch."
    )

    result, status_code = generate_gemini_response(prompt)
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=result.get("error", "Failed to generate motivation"))
    
    # Ensure we always return the correct format
    if isinstance(result, dict) and "text" in result:
        logger.info(f"Sending response for /motivation. Status: {status_code}, Result: {result}")
        return result
    else:
        # If we get an unexpected format, wrap it properly
        text_content = str(result) if not isinstance(result, dict) else result.get("text", str(result))
        formatted_result = {"text": text_content}
        logger.info(f"Sending formatted response for /motivation. Status: {status_code}, Result: {formatted_result}")
        return formatted_result

# --- Habit Suggestion Endpoint ---
@app.post('/habits', response_model=HabitResponse, tags=["AI Service"])
def get_habit_suggestions(data: HabitRequest):
    logger.info("Received request for /habits endpoint.")
    logger.debug(f"Habit suggestion request payload: {data.dict()}")
    
    user_name = data.user_name
    low_mood_activities = data.low_mood_activities
    high_mood_activities = data.high_mood_activities

    prompt_context = f"User: {user_name}.\n"
    if low_mood_activities:
        prompt_context += f"Activities sometimes linked to their lower moods: {', '.join(low_mood_activities)}.\n"
    if high_mood_activities:
        prompt_context += f"Activities sometimes linked to their better moods: {', '.join(high_mood_activities)}.\n"
    
    if not low_mood_activities and not high_mood_activities:
         prompt_context += "No specific activities strongly correlated with high or low moods were found in the recent data.\n"

    prompt = (
        f"{prompt_context}"
        "You are an AI assistant for a mood tracking app, providing habit suggestions. "
        "Based on the user's context, provide 2-3 concise, actionable, and empathetic habit suggestions. "
        "If specific activities are mentioned, try to incorporate them. "
        "One suggestion could be about mindfully approaching or reducing an activity linked to lower moods, and another about encouraging an activity linked to better moods. "
        "If no specific activities are strongly correlated, offer general well-being habits. "
        "Frame suggestions gently and constructively. Do not refer to yourself as an AI. "
        "Example for correlated activities: 'It seems like \"{low_activity}\" sometimes coincides with lower moods for you, {user_name}. Perhaps explore how that activity makes you feel, or try a mindful alternative. On the other hand, \"{high_activity}\" often appears with better moods - maybe find more ways to weave that into your week?' "
        "Example for general: '{user_name}, consider setting a small, achievable goal for today, like a 10-minute walk or dedicating time to a hobby you enjoy. Consistent small actions can make a big difference.' "
        "Output should be a list of strings, where each string is a suggestion. Format as a JSON list of strings. For example: [\"Suggestion 1\", \"Suggestion 2\"]"
    )
    
    result, status_code = generate_gemini_response(prompt, is_json_output=True)
    
    if status_code == 200:
        # Improved handling for different response types
        final_suggestions = []
        
        if isinstance(result, list):
            # Direct list response
            for item in result:
                item = str(item).strip()
                # Skip empty items and formatting artifacts
                if item and len(item) > 10 and not item.startswith(('```', '[', ']', '{', '}')):
                    # Clean quotes
                    if item.startswith('"') and item.endswith('"'):
                        item = item[1:-1]
                    elif item.startswith('"') and item.endswith('",'):
                        item = item[1:-2]
                    final_suggestions.append(item)
        
        elif isinstance(result, dict) and "text" in result:
            # Text response that should contain JSON
            text_content = result["text"]
            import re
            json_match = re.search(r'\[(.*?)\]', text_content, re.DOTALL)
            if json_match:
                try:
                    json_content = '[' + json_match.group(1) + ']'
                    parsed_list = json.loads(json_content)
                    final_suggestions = [str(item).strip().strip('"') for item in parsed_list if len(str(item).strip()) > 10]
                except:
                    # Fallback to line parsing
                    final_suggestions = [s.strip() for s in text_content.split('\n') if s.strip() and len(s.strip()) > 10]
            else:
                final_suggestions = [s.strip() for s in text_content.split('\n') if s.strip() and len(s.strip()) > 10]
        
        elif isinstance(result, str):
            # String response
            final_suggestions = [s.strip() for s in result.split('\n') if s.strip() and len(s.strip()) > 10]
        
        # Ensure we have valid suggestions
        if not final_suggestions:
            final_suggestions = ["Consider what small step you can take for your wellbeing today."]
        
        # Limit to 3 suggestions
        final_suggestions = final_suggestions[:3]

        logger.info(f"Sending response for /habits. Status: {status_code}, Suggestions: {final_suggestions}")
        return {"suggestions": final_suggestions}
    else:
        logger.info(f"Sending error response for /habits. Status: {status_code}, Error: {result}")
        raise HTTPException(status_code=status_code, detail=result)

# --- Nudge Generation Endpoint ---
@app.post('/generate-nudge', response_model=NudgeResponse, tags=["AI Service"])
def generate_nudge(data: NudgeRequest):
    """Generate a personalized nudge message based on user context"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="AI service not configured. Missing API key.")
    
    try:
        logger.info("Received request for nudge generation")
        
        context = data.context
        max_length = data.max_length
        
        user_age = context.user_age
        streak_count = context.streak_count
        coin_balance = context.coin_balance
        tone = context.tone
        recent_moods = context.recent_moods
        
        mood_context = ""
        if recent_moods:
            mood_context = "Recent mood patterns: "
            for mood in recent_moods:
                mood_context += f"{mood.get('date', 'unknown date')} - {mood.get('mood', 'unknown mood')} (rating: {mood.get('rating', 'unknown')}), "
            mood_context = mood_context.rstrip(', ')
        else:
            mood_context = "No recent mood data available"
        
        if tone == 'gen_z':
            prompt = f"""You are a friendly, supportive mental health companion speaking to a young person (age {user_age}). Generate a casual, encouraging nudge message to remind them to log their mood today. 

Context:
- Current streak: {streak_count} days
- Coin balance: {coin_balance}
- {mood_context}

Requirements:
- Use Gen Z language (casual, friendly, with emojis)
- Keep it under {max_length} characters
- Be encouraging and supportive
- Include a gentle call to action
- Don't be preachy or overly clinical

Example style: "Hey! Your {streak_count}-day streak is looking great! 🔥 Quick vibe check - how are you feeling today? ✨"

Generate a fresh, personalized message:"""
        else:
            prompt = f"""You are a professional, supportive mental health companion. Generate a respectful, encouraging nudge message to remind the user to log their mood today.

Context:
- Current streak: {streak_count} days
- Coin balance: {coin_balance}
- {mood_context}

Requirements:
- Use professional, respectful language
- Keep it under {max_length} characters
- Be encouraging and supportive
- Include a gentle call to action
- Focus on the benefits of self-reflection

Example style: "You've maintained a {streak_count}-day reflection streak. Taking a moment to acknowledge your current emotional state can be valuable. How are you feeling today?"

Generate a fresh, personalized message:"""
        
        result, status_code = generate_gemini_response(prompt)
        
        if status_code == 200:
            # Ensure we extract the message properly
            if isinstance(result, dict) and "text" in result:
                nudge_message = result["text"].strip()
            elif isinstance(result, list) and len(result) > 0:
                nudge_message = str(result[0]).strip()
            else:
                nudge_message = str(result).strip()
            
            if len(nudge_message) > max_length:
                nudge_message = nudge_message[:max_length-3] + "..."
            
            logger.info(f"Generated nudge message: {nudge_message}")
            return {"message": nudge_message, "tone": tone}
        else:
            logger.error(f"AI response failed with status {status_code}: {result}")
            raise HTTPException(status_code=status_code, detail=result)
    
    except Exception as e:
        logger.error(f"Error generating nudge: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate nudge: {str(e)}")

# --- Challenge Completion Message Generation Endpoint ---
@app.post('/generate-challenge-message', response_model=ChallengeResponse, tags=["AI Service"])
def generate_challenge_message(data: ChallengeMessageRequest):
    """Generate a personalized challenge completion message"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="AI service not configured. Missing API key.")
    
    try:
        logger.info("Received request for challenge completion message generation")
        
        context = data.context
        max_length = data.max_length
        
        challenge_type = context.challenge_type
        stake = context.stake
        duration_days = context.duration_days
        user_age = context.user_age
        completed = context.completed
        tone = context.tone
        
        if completed:
            if tone == 'gen_z':
                prompt = f"""You are a friendly, supportive mental health companion celebrating a young person's achievement. They just completed a {challenge_type} challenge that lasted {duration_days} days with a {stake} coin stake.

Requirements:
- Use Gen Z language (casual, celebratory, with emojis)
- Keep it under {max_length} characters
- Be genuinely excited and congratulatory
- Mention the coin reward they earned
- Encourage continued progress

Example style: "YESSS! 🎉 You absolutely crushed that {duration_days}-day {challenge_type} challenge! Your {stake} coins are well-earned! Keep that momentum going! 🔥✨"

Generate a fresh, personalized celebration message:"""
            else:
                prompt = f"""You are a professional, supportive mental health companion congratulating the user on completing a {challenge_type} challenge that lasted {duration_days} days with a {stake} coin stake.

Requirements:
- Use professional, respectful language
- Keep it under {max_length} characters
- Be genuinely congratulatory
- Mention the coin reward and personal growth
- Encourage continued engagement

Example style: "Congratulations on successfully completing your {duration_days}-day {challenge_type} challenge! You've earned {stake} coins and demonstrated remarkable commitment to your mental health journey."

Generate a fresh, personalized message:"""
        else:
            if tone == 'gen_z':
                prompt = f"""You are a friendly, supportive mental health companion speaking to a young person who didn't complete their {challenge_type} challenge that lasted {duration_days} days with a {stake} coin stake.

Requirements:
- Use Gen Z language (casual, encouraging, with emojis)
- Keep it under {max_length} characters
- Be supportive and not judgmental
- Encourage them to try again
- Focus on learning and growth

Example style: "Hey, no worries about the {challenge_type} challenge - these things happen! 💙 The important thing is you tried. Ready to give it another shot? You've got this! ✨"

Generate a fresh, encouraging message:"""
            else:
                prompt = f"""You are a professional, supportive mental health companion speaking to a user who didn't complete their {challenge_type} challenge that lasted {duration_days} days with a {stake} coin stake.

Requirements:
- Use professional, respectful language
- Keep it under {max_length} characters
- Be supportive and encouraging
- Focus on learning from the experience
- Encourage future participation

Example style: "While you didn't complete this {challenge_type} challenge, attempting it shows commitment to your wellbeing. Consider what you learned from this experience as you plan your next challenge."

Generate a fresh, supportive message:"""
        
        result, status_code = generate_gemini_response(prompt)
        
        if status_code == 200:
            # Ensure we extract the message properly
            if isinstance(result, dict) and "text" in result:
                message = result["text"].strip()
            elif isinstance(result, list) and len(result) > 0:
                message = str(result[0]).strip()
            else:
                message = str(result).strip()
            
            if len(message) > max_length:
                message = message[:max_length-3] + "..."
            
            logger.info(f"Generated challenge message: {message}")
            return {"message": message, "tone": tone, "completed": completed}
        else:
            logger.error(f"AI response failed with status {status_code}: {result}")
            raise HTTPException(status_code=status_code, detail=result)
    
    except Exception as e:
        logger.error(f"Error generating challenge message: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate challenge message: {str(e)}")

# --- Mood Insights Generation Endpoint ---
@app.post('/generate-insights', response_model=InsightsResponse, tags=["AI Service"])
def generate_insights(data: InsightsRequest):
    """Generate personalized mood insights and recommendations"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="AI service not configured. Missing API key.")
    
    try:
        logger.info("Received request for mood insights generation")
        
        context = data.context
        
        user_age = context.user_age
        analysis_period = context.analysis_period
        mood_data = context.mood_data
        avg_rating = context.avg_rating
        total_entries = context.total_entries
        streak_count = context.streak_count
        tone = context.tone
        
        mood_patterns = {}
        activity_patterns = {}
        
        for entry in mood_data:
            mood = entry.get('mood', '').lower()
            if mood:
                mood_patterns[mood] = mood_patterns.get(mood, 0) + 1
            
            activities = entry.get('activities', '')
            if activities:
                for activity in activities.split(','):
                    activity = activity.strip().lower()
                    if activity:
                        activity_patterns[activity] = activity_patterns.get(activity, 0) + 1
        
        top_moods = sorted(mood_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        top_activities = sorted(activity_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if tone == 'gen_z':
            prompt = f"""You're a friendly, insightful mental health companion analyzing {analysis_period} days of mood data for a young person (age {user_age}). 

Data Summary:
- Total mood entries: {total_entries}
- Average mood rating: {avg_rating:.1f}/5
- Current streak: {streak_count} days
- Most common moods: {', '.join([f'{mood} ({count}x)' for mood, count in top_moods])}
- Common activities: {', '.join([f'{activity} ({count}x)' for activity, count in top_activities])}

Mood entries sample: {mood_data[:5]}

Generate a response with:
1. 3-4 personalized insights about their mood patterns (casual, Gen Z tone with emojis)
2. 3-4 actionable recommendations for improving their mental health
3. A brief mood trend analysis

Use encouraging, non-clinical language. Be specific about patterns you notice. Format as JSON with keys: "insights", "recommendations", "mood_trends"."""
        else:
            prompt = f"""You're a professional mental health companion analyzing {analysis_period} days of mood data for a user (age {user_age}).

Data Summary:
- Total mood entries: {total_entries}
- Average mood rating: {avg_rating:.1f}/5
- Current streak: {streak_count} days
- Most common moods: {', '.join([f'{mood} ({count}x)' for mood, count in top_moods])}
- Common activities: {', '.join([f'{activity} ({count}x)' for activity, count in top_activities])}

Mood entries sample: {mood_data[:5]}

Generate a response with:
1. 3-4 personalized insights about their mood patterns (professional, supportive tone)
2. 3-4 evidence-based recommendations for enhancing their wellbeing
3. A comprehensive mood trend analysis

Use professional, encouraging language. Be specific about patterns observed. Format as JSON with keys: "insights", "recommendations", "mood_trends"."""
        
        result, status_code = generate_gemini_response(prompt, is_json_output=True)
        
        if status_code == 200:
            if isinstance(result, dict):
                if "insights" in result and "recommendations" in result:
                    logger.info(f"Generated insights: {len(result.get('insights', []))} insights, {len(result.get('recommendations', []))} recommendations")
                    return result
                elif "text" in result:
                    insights_text = result["text"]
                else:
                    insights_text = str(result)
            elif isinstance(result, list):
                insights_text = "\n".join(result)
            else:
                insights_text = str(result)
            
            lines = insights_text.split('\n')
            insights = []
            recommendations = []
            mood_trends = "Analysis of your mood patterns over the specified period."
            
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if 'insight' in line.lower() or 'pattern' in line.lower():
                    current_section = 'insights'
                elif 'recommend' in line.lower() or 'suggest' in line.lower():
                    current_section = 'recommendations'
                elif 'trend' in line.lower() or 'analysis' in line.lower():
                    current_section = 'trends'
                elif line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.')):
                    if current_section == 'insights':
                        insights.append(line)
                    elif current_section == 'recommendations':
                        recommendations.append(line)
                    elif current_section == 'trends':
                        mood_trends = line
            
            final_result = {
                'insights': insights[:4] if insights else ["Your mood tracking shows dedication to self-awareness."],
                'recommendations': recommendations[:4] if recommendations else ["Continue your daily mood tracking for better insights."],
                'mood_trends': mood_trends
            }
            
            logger.info(f"Generated insights (parsed): {len(final_result['insights'])} insights, {len(final_result['recommendations'])} recommendations")
            return final_result
        else:
            logger.error(f"AI response failed with status {status_code}: {result}")
            raise HTTPException(status_code=status_code, detail=result)
    
    except Exception as e:
        logger.error(f"Error generating insights: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

# --- Keep-Alive Service ---
class KeepAliveService:
    """Built-in keep-alive service to prevent server from sleeping"""
    
    def __init__(self, base_url=None, interval=KEEP_ALIVE_INTERVAL):
        self.base_url = base_url
        self.interval = interval
        self.running = False
        self.thread = None
        self.logger = logging.getLogger(__name__ + '.KeepAlive')
        
    def start(self):
        """Start the keep-alive service"""
        if not KEEP_ALIVE_ENABLED:
            self.logger.info("Keep-alive service disabled (development mode)")
            return
            
        if not self.base_url:
            self.logger.warning("Keep-alive service: No base URL provided, cannot start")
            return
            
        if self.running:
            self.logger.warning("Keep-alive service already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._keep_alive_loop, daemon=True)
        self.thread.start()
        self.logger.info(f"Keep-alive service started - pinging {self.base_url}/api/health/ every {self.interval//60} minutes")
    
    def stop(self):
        """Stop the keep-alive service"""
        self.running = False
        if self.thread:
            self.logger.info("Keep-alive service stopped")
    
    def _keep_alive_loop(self):
        """Main keep-alive loop"""
        while self.running:
            try:
                time.sleep(self.interval)
                
                if not self.running:
                    break
                
                self._ping_health_endpoint()
                
            except Exception as e:
                self.logger.error(f"Keep-alive service error: {str(e)}")
                
    @intelligent_retry(config=RetryConfig())
    def _ping_health_endpoint_core(self, health_url: str) -> bool:
        """Core ping logic without retry - used by retry decorator"""
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                self.logger.debug(f"Keep-alive ping successful: {health_url}")
                return True
            else:
                raise AIServiceError(
                    f"Health check failed with status {response.status_code}",
                    error_type="http_error",
                    is_retryable=True
                )
        except requests.RequestException as e:
            raise AIServiceError(
                f"Network error during health check: {str(e)}",
                error_type="network",
                is_retryable=True
            )
    
    def _ping_health_endpoint(self):
        """Ping the health endpoint with intelligent retry logic"""
        health_url = f"{self.base_url}/api/health/"
        
        try:
            self._ping_health_endpoint_core(health_url)
        except AIServiceError as e:
            self.logger.error(f"Keep-alive ping failed after all retries: {e.error_type} - {str(e)}")
        except Exception as e:
            self.logger.error(f"Unexpected error in keep-alive ping: {str(e)}")

keep_alive_service = KeepAliveService()

def start_keep_alive_service():
    """Start the keep-alive service after the app starts"""
    base_url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("KEEP_ALIVE_URL")
    
    if not base_url:
        logger.warning("Keep-alive service: No base URL provided via RENDER_EXTERNAL_URL or KEEP_ALIVE_URL environment variables")
        return
    
    base_url = base_url.rstrip('/')
    
    keep_alive_service.base_url = base_url
    keep_alive_service.start()

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 5001))
    debug_mode = os.environ.get("FLASK_ENV", "production") == "development"
    
    log_level = "debug" if debug_mode else "info"
    
    logger.info(f"Starting Uvicorn server in {'DEVELOPMENT' if debug_mode else 'PRODUCTION'} mode on port {port}")
    uvicorn.run("app:app", host='0.0.0.0', port=port, reload=debug_mode, log_level=log_level)
    logger.info(f"Application stopped.")
