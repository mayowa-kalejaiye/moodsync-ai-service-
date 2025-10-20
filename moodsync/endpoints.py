import logging
import random
from datetime import datetime
from fastapi import APIRouter, HTTPException

from .models import *
from .ai import generate_gemini_response
from ..config import KEEP_ALIVE_ENABLED

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get('/', tags=["Health"])
def read_root():
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



@router.get('/health', tags=["Health"])
def health_check():
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
        "gemini_configured": bool(False),
        "keep_alive_enabled": KEEP_ALIVE_ENABLED,
        "timestamp": datetime.now().isoformat()
    }


@router.post('/motivation', response_model=StandardResponse, tags=["AI Service"])
def get_motivation(data: MotivationRequest):
    logger.info("Received request for /motivation endpoint.")
    user_name = data.user_name
    mood_trend_label = data.mood_trend_label
    recent_mood_texts = data.recent_mood_texts
    recent_notes_texts = data.recent_notes_texts
    user_age = data.user_age

    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        genz_time = random.choice(["chill morning", "vibey morning", "soft sunrise"])
        mature_time = random.choice(["peaceful morning", "productive morning"])
    elif 12 <= current_hour < 18:
        genz_time = random.choice(["mid afternoon", "pretty chill afternoon"])
        mature_time = random.choice(["afternoon", "busy afternoon"])
    else:
        genz_time = random.choice(["pretty chill evening", "cool evening"])
        mature_time = random.choice(["relaxing evening", "quiet evening"])

    age = None
    if user_age is not None:
        try:
            age = int(user_age)
        except Exception:
            age = None

    if age is not None and age >= 30:
        time_of_day_phrase = mature_time
        style_instructions = (
            "Use a warm, supportive, and mature tone. Avoid slang and keep it professional yet friendly. "
        )
    else:
        time_of_day_phrase = genz_time
        style_instructions = (
            "Use a friendly, modern, Gen Z style—think real words, not clinical or generic. "
        )

    prompt_context = f"User: {user_name}. Time of day: {time_of_day_phrase}.\n"
    prompt_context += f"Recent mood trend: {mood_trend_label}.\n"
    if recent_mood_texts:
        prompt_context += f"Recent logged moods: {', '.join(recent_mood_texts)}.\n"

    prompt = (
        f"{prompt_context}"
        "You are a supportive, relatable, and empathetic AI companion for a mood tracking app. "
        f"{style_instructions} "
        "Based on the user's context, provide a short (2-3 sentences), personalized, and actionable motivational message. "
    )

    result, status_code = generate_gemini_response(prompt)
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=result.get("error", "Failed to generate motivation"))

    if isinstance(result, dict) and "text" in result:
        return result
    else:
        text_content = str(result) if not isinstance(result, dict) else result.get("text", str(result))
        formatted_result = {"text": text_content}
        return formatted_result
