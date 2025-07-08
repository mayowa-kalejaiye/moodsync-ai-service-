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

### Utility Features

#### 6. **Model Discovery** (`/list-models`)
- Lists available Gemini AI models
- Filters models that support content generation
- Provides model metadata and capabilities
- Helps with service debugging and model selection

## 🛠️ Technical Architecture

### AI Integration
- **Primary AI Provider**: Google Gemini AI
- **Model Fallback System**: Automatically tries multiple Gemini models (`gemini-1.0-pro`, `gemini-pro`, `gemini-1.5-flash-latest`)
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
- Flask web framework

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
```

4. **Run the service**
```bash
python app.py
```

The service will be available at `http://localhost:5001`

### Dependencies
```
flask
python-dotenv
google-generativeai
```

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Your Google Gemini AI API key (required)
- `PORT`: Service port (default: 5001)

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
The service implements a robust fallback system:
1. Attempts primary model (`gemini-1.0-pro`)
2. Falls back to secondary models if primary fails
3. Provides detailed error logging for debugging
4. Graceful degradation for service reliability

### Error Handling
- **Model Not Found**: Automatically tries alternative models
- **Content Filtering**: Handles AI safety filtering gracefully
- **API Limits**: Comprehensive rate limiting and quota management
- **JSON Parsing**: Fallback mechanisms for malformed AI responses

## 🏗️ Development

### Code Structure
```
ai_service/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .env               # Environment configuration
└── README.md          # This file
```

### Key Components
- **AI Response Generator**: Core function handling Gemini AI interactions
- **Context Builders**: Functions that prepare user context for AI prompts
- **Tone Adapters**: Logic for age and preference-based communication styles
- **Pattern Analyzers**: Mood and activity correlation detection

## 🤝 Contributing

This service is part of the MoodSync ecosystem. For contributions:
1. Follow the coding standards established in the main MoodSync project
2. Ensure all AI interactions are tested with various user contexts
3. Maintain backward compatibility with the mobile app API
4. Add comprehensive logging for new features

## 📝 License

This project is part of the MoodSync application suite. Please refer to the main [MoodSync repository](https://github.com/mayowa-kalejaiye/Moodify.git) for licensing information.

## 🔍 Monitoring & Debugging

### Health Checks
- Use `/list-models` endpoint to verify AI service connectivity
- Monitor logs for model fallback patterns
- Track response times and error rates

### Common Issues
- **AI Key Missing**: Ensure `GEMINI_API_KEY` is properly set
- **Model Errors**: Check `/list-models` for available model names
- **JSON Parsing**: Review logs for malformed AI responses

---

**Built with ❤️ for mental health and wellbeing**

For questions about the MoodSync mobile app, visit the [main repository](https://github.com/mayowa-kalejaiye/Moodify.git).
