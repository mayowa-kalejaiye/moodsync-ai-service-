import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from ..config import GEMINI_API_KEY, KEEP_ALIVE_ENABLED
from .keepalive import KeepAliveService

logger = logging.getLogger(__name__)


keep_alive_service = KeepAliveService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            logger.info("Gemini API configured successfully.")
        except Exception:
            logger.exception("Failed to configure Gemini API")
    else:
        logger.error("CRITICAL: GEMINI_API_KEY not found. AI suggestions will fail.")

    if KEEP_ALIVE_ENABLED:
        # start keep alive later; endpoints will set base_url and start when appropriate
        pass

    yield

    logger.info("Application shutting down...")
    keep_alive_service.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MoodSync AI Service",
        description="AI-powered service for generating motivational messages, habit suggestions, and insights.",
        version="1.0.0",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Import endpoints to register routes (import the router to avoid circular imports)
    from .endpoints import router as endpoints_router  # noqa: F401
    app.include_router(endpoints_router)

    return app


app = create_app()
