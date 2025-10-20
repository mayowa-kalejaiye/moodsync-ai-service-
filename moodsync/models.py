from typing import List, Optional, Dict, Any
from pydantic import BaseModel


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
