"""
FastAPI Application — Main entry point with Hybrid AI Knowledge Engine.

Startup: LLM health check → Weaviate indexing → CAG preload → Agent init
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.model_factory import get_model, load_all_configs
from ..core.concurrency import shutdown_executor
from ..rag.vector_store import WeaviateVectorStore
from ..rag.indexer import index_providers
from ..state.session_store import SessionStore
from ..state.booking_store import BookingStore
from ..cag.cag_manager import CAGManager
from ..observability.tracing import init_tracing
from ..agents.orchestrator import Orchestrator
from .routes import router, set_dependencies

logger = logging.getLogger("api.app")

_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    start_time = time.time()
    logger.info("═" * 60)
    logger.info("  AI Service Orchestrator — OPEN SOURCE STACK")
    logger.info("  Gemma 4 31B (Unsloth) + Weaviate + Redis + PostgreSQL")
    logger.info("═" * 60)

    # Initialize OpenTelemetry tracing
    init_tracing()

    # Load all configs
    configs = load_all_configs()

    # Initialize LLM (Unsloth — local Gemma 4 31B, in-process)
    llm = get_model()
    provider = os.getenv("MODEL_PROVIDER", "unsloth")
    logger.info(f"LLM: {llm.model_id} via {provider} (local, in-process)")

    # Check local model health
    if hasattr(llm, "check_health"):
        if llm.check_health():
            logger.info(f"{provider} model: ✅ ready")
            if hasattr(llm, "list_models"):
                logger.info(f"Available models: {llm.list_models()}")
        else:
            logger.warning(
                f"{provider} model: ❌ failed to load — verify UNSLOTH_MODEL_ID and HF_TOKEN"
            )

    # Initialize Weaviate vector store & index
    logger.info("Initializing Weaviate vector store...")
    try:
        vector_store = WeaviateVectorStore()
        if vector_store.count == 0:
            logger.info("Weaviate empty — indexing 50K providers...")
            embedding_cfg = configs["model"].get("embedding", {})
            vector_store = index_providers(
                embedding_model=embedding_cfg.get("model_name", "BAAI/bge-large-en-v1.5"),
                batch_size=embedding_cfg.get("batch_size", 128),
                device=embedding_cfg.get("device", "cpu"),
            )
        else:
            logger.info(f"Weaviate loaded: {vector_store.count} documents")
    except Exception as e:
        logger.error(f"Weaviate connection failed: {e}. Start with: docker-compose up -d weaviate")
        vector_store = None

    # Initialize CAG (Cache-Augmented Generation)
    cag_manager = CAGManager()
    logger.info(f"CAG: {cag_manager.size} golden knowledge entries loaded")

    # Initialize state stores (Redis + PostgreSQL with fallbacks)
    session_store = SessionStore()
    booking_store = BookingStore()

    # Initialize orchestrator
    if vector_store:
        orchestrator = Orchestrator(
            llm=llm,
            vector_store=vector_store,
            session_store=session_store,
            booking_store=booking_store,
            cag_manager=cag_manager,
            configs=configs,
        )
    else:
        orchestrator = None
        logger.error("Orchestrator not initialized — Weaviate required")

    set_dependencies(orchestrator, vector_store, session_store, start_time)

    logger.info("═" * 60)
    logger.info(f"  System Ready | {vector_store.count if vector_store else 0} providers")
    logger.info(f"  Hybrid Engine: RAG + Reranking + CAG + Context Engineering")
    logger.info(f"  Startup: {time.time() - start_time:.1f}s")
    logger.info(f"  Thread Pool: MAX_WORKERS={os.getenv('MAX_WORKERS', '8')}")
    logger.info("═" * 60)

    yield

    # Cleanup
    shutdown_executor()  # Gracefully drain the shared thread pool
    if vector_store:
        vector_store.close()
    logger.info("Shutdown complete.")


# Create FastAPI app
app = FastAPI(
    title="AI Service Orchestrator for Informal Economy",
    description=(
        "Agentic AI system powered by Gemma (Unsloth, open-source) with "
        "Hybrid AI Knowledge Engine: Agentic RAG + Reranking + CAG + "
        "Context Engineering + LoRA Fine-Tuning."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "service": "AI Service Orchestrator",
        "version": "2.0.0",
        "stack": "Gemma (Unsloth) + Weaviate + Redis + PostgreSQL",
        "knowledge_engine": ["Agentic RAG", "Reranking", "CAG (Rust)", "Context Engineering", "LoRA Fine-Tuning"],
        "docs": "/docs",
        "health": "/api/v1/health",
    }
