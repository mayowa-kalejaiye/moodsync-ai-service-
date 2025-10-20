# MoodSync AI Service

An intelligent AI-powered backend service that provides personalized mental health support for the [MoodSync](https://github.com/mayowa-kalejaiye/Moodify.git) mood tracking application. This service leverages Google's Gemini AI to deliver contextual motivation, habit suggestions, insights, and personalized user interactions.

## 🚀 Features

### Core AI Endpoints

#### 1. **Motivational Messages** (`/motivation`)
- Generates personalized motivational content based on user context
- Adapts tone and language based on user age (Gen Z vs. mature communication styles)
- Incorporates time-of-day awareness with contextual phrases
- Responds to specific mental health indicators (anxiety, stress)
- Uses recent mood patterns and user notes for personalization

#### 2. **Habit Suggestions** (`/habits`)
- Analyzes user's mood-activity correlations
- Provides 2-3 actionable, personalized habit recommendations
- Identifies activities linked to positive and negative mood patterns
- Offers constructive alternatives for mood-lowering activities
- Returns suggestions in structured JSON format

#### 3. **Smart Nudges** (`/generate-nudge`)
- Creates personalized reminder messages for mood logging
- Adapts messaging based on user's current streak and engagement
- Supports multiple communication tones (professional, Gen Z)
- Incorporates recent mood patterns for relevance
- Respects character limits for mobile notifications

#### 4. **Challenge Completion Messages** (`/generate-challenge-message`)
- Generates celebration messages for completed challenges
- Provides supportive messages for incomplete challenges
- Customizes tone based on user preferences and age
- Incorporates challenge details (type, duration, stakes)
- Encourages continued engagement and growth

#### 5. **Mood Insights & Analytics** (`/generate-insights`)
- Analyzes mood patterns over configurable time periods
- Provides 3-4 personalized insights about mood trends
- Generates evidence-based recommendations for wellbeing
- Creates comprehensive mood trend analysis
- Returns structured insights in JSON format

## 🛠️ Technical Architecture

### AI Integration
- **Primary AI Provider**: Google Gemini AI
- **Current Model**: `gemini-2.0-flash-exp` (latest experimental model)
- **Error Handling**: Comprehensive error handling with detailed logging
- **Response Parsing**: Smart JSON parsing with fallback mechanisms

### Communication Styles
- **Gen Z Mode**: Casual, emoji-rich, relatable language for younger users
- **Professional Mode**: Mature, respectful, clinical-appropriate tone for older users
- **Age-Adaptive**: Automatically selects appropriate style based on user age (30+ defaults to professional)

### Context Awareness
- **Time Intelligence**: Generates time-appropriate phrases (morning, afternoon, evening)
- **Mood Pattern Recognition**: Analyzes recent mood entries for personalized responses
- **Activity Correlation**: Identifies patterns between activities and mood states
- **Mental Health Sensitivity**: Detects anxiety, stress indicators for appropriate responses

### Framework & Architecture
- **Web Framework**: FastAPI (high-performance, modern Python web framework)
- **Async Support**: Built-in async/await support for better performance
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation at `/docs`
- **Type Safety**: Full Pydantic model validation for request/response data
- **CORS Support**: Configured for cross-origin requests from mobile apps

### Resilience & Fault Tolerance
- **Intelligent Retry Logic**: Exponential backoff with jitter for transient failures
- **Circuit Breaker Pattern**: Prevents cascading failures and enables fast recovery
- **Error Classification**: Distinguishes between retryable and non-retryable errors
- **Configurable Parameters**: Fully configurable retry delays, thresholds, and timeouts
- **Graceful Degradation**: Provides meaningful fallback responses during outages
- **Comprehensive Logging**: Detailed retry attempt and failure logging for debugging

## 📋 API Reference

### Request/Response Examples

#### Motivation Endpoint
```http
POST /motivation
Content-Type: application/json

{
  "user_name": "Alex",
  "mood_trend_label": "anxious",
  "recent_mood_texts": ["anxious", "stressed", "overwhelmed"],
  "recent_notes_texts": ["work deadline approaching"],
  "user_age": 22
}
```

#### Habit Suggestions Endpoint
```http
POST /habits
Content-Type: application/json

{
  "user_name": "Alex",
  "low_mood_activities": ["social media", "late night scrolling"],
  "high_mood_activities": ["exercise", "reading", "spending time outdoors"]
}
```

#### Insights Generation Endpoint
```http
POST /generate-insights
Content-Type: application/json

{
  "context": {
    "user_age": 25,
    "analysis_period": 30,
    "mood_data": [...],
    "avg_rating": 3.2,
    "total_entries": 28,
    "streak_count": 7,
    "tone": "gen_z"
  }
}
```

## 🚦 Setup & Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key
- FastAPI web framework

### Environment Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ai_service
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Configuration**
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
PORT=5001
FLASK_ENV=production
KEEP_ALIVE_URL=
RENDER_EXTERNAL_URL=
```

4. **Run the service**
```bash
python app.py
```

The service will be available at `http://localhost:5001`

**API Documentation**: Visit `http://localhost:5001/docs` for interactive Swagger documentation

### Deployment to Render

This service is optimized for deployment on [Render](https://render.com):

1. **Connect Repository**: Link your GitHub repository to Render
2. **Use Web Service**: Create a new Web Service (not Static Site)
3. **Build Configuration**: The included `render.yaml` will auto-configure
4. **Environment Variables**: Set `GEMINI_API_KEY` in Render dashboard
5. **Auto-Deploy**: Optional - enable for automatic deployments

**Required Environment Variables on Render:**
- `GEMINI_API_KEY`: Your Google Gemini API key

The service includes:
- `runtime.txt`: Specifies Python 3.10.14 for compatibility
- `render.yaml`: Optimized build configuration for Render
- Compatible dependency versions to avoid Rust compilation issues

### Dependencies
```txt
fastapi==0.95.2
uvicorn[standard]==0.22.0
python-dotenv==1.0.0
google-generativeai==0.3.2
pydantic==1.10.12
requests==2.31.0
```

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini AI API key (required)
- `PORT`: Service port (default: 5001)
- `FLASK_ENV`: Environment mode (development/production)
- `KEEP_ALIVE_URL`: Optional URL for keep-alive service
- `RENDER_EXTERNAL_URL`: External service URL for deployment

**Resilience Configuration:**
- `AI_MAX_RETRIES`: Maximum retry attempts for AI calls (default: 3)
- `AI_BASE_DELAY`: Base delay between retries in seconds (default: 1.0)
- `AI_MAX_DELAY`: Maximum delay between retries in seconds (default: 30.0)
- `AI_JITTER_MAX`: Maximum jitter to add to delays (default: 0.5)
- `AI_EXPONENTIAL_BASE`: Exponential backoff multiplier (default: 2.0)
- `AI_CIRCUIT_BREAKER_THRESHOLD`: Failures before circuit opens (default: 5)
- `AI_CIRCUIT_BREAKER_TIMEOUT`: Recovery timeout in seconds (default: 60)
- `AI_CIRCUIT_BREAKER_SUCCESS_THRESHOLD`: Successes needed to close circuit (default: 2)

### Logging
- Comprehensive logging with configurable levels
- Request/response tracking for debugging
- Error tracking with stack traces
- AI model selection and fallback logging

## 🔗 Integration with MoodSync

This AI service is designed to seamlessly integrate with the [MoodSync mobile application](https://github.com/mayowa-kalejaiye/Moodify.git) by providing:

- **Real-time Motivation**: Contextual encouragement based on user's current emotional state
- **Intelligent Habit Formation**: Data-driven suggestions for improving mental wellbeing
- **Personalized Insights**: Deep analysis of mood patterns and trends
- **Engagement Optimization**: Smart notifications and challenge completion feedback
- **Age-Appropriate Communication**: Adaptive messaging for different user demographics

## 📊 AI Model Management

### Model Selection Strategy

The service uses Google's latest Gemini models with a focused, resilient approach:

1. **Primary Model**: `gemini-2.0-flash-exp` (latest experimental model)
2. **Single Model Focus**: Optimized for performance and resource efficiency
3. **Intelligent Retry**: Exponential backoff with jitter for transient failures
4. **Circuit Breaker Protection**: Prevents cascading failures during outages
5. **Error Classification**: Smart distinction between retryable and permanent errors
6. **Performance Monitoring**: Response time and success rate tracking

# MoodSync AI Service

An intelligent AI-powered backend service that provides personalized mental health support for the MoodSync mobile application. It exposes FastAPI endpoints that generate personalized motivational messages, habit suggestions, nudges, and mood insights using Google Gemini models.

See the consolidated developer & deployment guide in `docs/DEVELOPER.md` for setup, deployment and troubleshooting instructions.

Key files:
- `app.py` — top-level shim that re-exports the FastAPI app from `moodsync.core`.
- `moodsync/` — application package (routes, AI glue code, resilience helpers).
- `requirements.txt`, `runtime.txt` — dependency/runtime pins.

If you're a contributor, open `docs/DEVELOPER.md` first — it contains contribution, testing, and deployment guidance.
  - Half-open state for recovery testing
