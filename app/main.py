"""
app/main.py
-----------
FastAPI application entry-point.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import chat, evaluation, ingest
from app.core.config import get_settings
from app.core.logger import logger, setup_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logger()
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version} [{settings.app_env}]")
    logger.info(f"LLM  : {settings.ollama_llm_model}  @  {settings.ollama_base_url}")
    logger.info(f"Embed: {settings.ollama_embed_model}")
    yield
    logger.info("Shutting down …")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "RAG Chatbot powered by Ollama (local LLM) + ChromaDB + LangChain. "
            "Includes professional RAGAS-based evaluation."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(ingest.router)
    app.include_router(chat.router)
    app.include_router(evaluation.router)

    @app.get("/health", tags=["System"])
    async def health_check():
        return JSONResponse({"status": "ok", "version": settings.app_version})

    return app


app = create_app()
