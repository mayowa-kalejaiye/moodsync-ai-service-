import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
AI_REQUEST_TIMEOUT = 30
KEEP_ALIVE_ENABLED = os.environ.get("FLASK_ENV", "production") != "development"
KEEP_ALIVE_INTERVAL = 10 * 60
KEEP_ALIVE_URL = os.environ.get("KEEP_ALIVE_URL")
