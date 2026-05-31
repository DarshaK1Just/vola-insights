import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of launch directory
_HERE = Path(__file__).resolve().parent          # api/
_PROJECT_ROOT = _HERE.parent                     # transaction_rag/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.logging_config import configure_logging
configure_logging()

import logging
import os
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.models import RunRequest, RunResponse, UserInfo
from src.pipeline import TransactionRAGPipeline
from src.config import Config

logger = logging.getLogger(__name__)

_pipeline: TransactionRAGPipeline | None = None


def _make_demo_df() -> pd.DataFrame:
    """Built-in demo dataset used when DATA_FILE is not configured."""
    from datetime import datetime, timedelta
    rng = np.random.default_rng(42)
    users = [
        ("usr_jose001", "Jose BazBaz"),
        ("usr_ana002", "Ana Reyes"),
        ("usr_mike003", "Mike Chen"),
    ]
    categories = [
        "Food > Restaurants > Fast Food",
        "Food > Groceries",
        "Transport > Ride Share",
        "Entertainment > Streaming",
        "Utilities > Electric",
        "Shopping > Clothing",
        "Health > Pharmacy",
    ]
    rows = []
    base = datetime(2024, 1, 1)
    for uid, uname in users:
        for i in range(130):
            dt = base + timedelta(days=int(rng.integers(0, 365)))
            is_income = rng.random() < 0.15
            if is_income:
                amt = -float(int(rng.integers(2500, 5000)))
                cat = "Income > Salary"
                merchant = "Acme Corp"
            else:
                amt = float(round(float(rng.integers(5, 350)) + rng.random(), 2))
                cat = categories[int(rng.integers(0, len(categories)))]
                merchant = f"Store_{int(rng.integers(1, 25))}"
            rows.append({
                "user_id": uid,
                "user_name": uname,
                "transaction_date": dt,
                "transaction_amount": amt,
                "merchant_name": merchant,
                "transaction_category_detail": cat,
            })
    return pd.DataFrame(rows)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup sequence:
      1. Load DataFrame (~200 ms) — synchronous
      2. Create TransactionRAGPipeline (~200 ms) — synchronous
         Pipeline.__init__ is fast; it starts a background thread that
         loads guardrails + Redis + LangGraph while the API serves requests.
      3. yield — API is live immediately
    """
    global _pipeline
    data_file = Config.DATA_FILE
    try:
        if data_file and Path(data_file).exists():
            from src.data_loader import load_transactions
            df = load_transactions(data_file)
            logger.info("Loaded %d transactions from %s", len(df), data_file)
        else:
            logger.warning("DATA_FILE not set — using built-in demo data")
            df = _make_demo_df()
        _pipeline = TransactionRAGPipeline(df=df)
        logger.info(
            "API ready — %d users loaded. Guardrails + LangGraph loading in background...",
            len(_pipeline.users),
        )
    except Exception as exc:
        logger.error("Failed to initialise pipeline: %s", exc, exc_info=True)
    yield
    logger.info("Shutting down Transaction AI API")


app = FastAPI(
    title="Vola Insights",
    description="DataFrame-first financial AI pipeline — OpenRouter LLM + LangGraph + Guardrails AI.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["ops"])
def health():
    """
    Always returns 200 (API is up).
    Check 'pipeline_ready' to know if guardrails + LangGraph are fully loaded.
    """
    ready = bool(_pipeline and _pipeline.is_ready)
    return {
        "status": "ok",
        "pipeline_ready": ready,
        "data_rows": len(_pipeline._df) if _pipeline else 0,
        "data_file": Config.DATA_FILE or "demo",
        "cache": _pipeline.cache_info if _pipeline else {},
    }


@app.get("/ready", tags=["ops"])
def ready():
    """
    Returns 200 when the full pipeline (guardrails + LangGraph) is ready.
    Returns 503 while still initialising.
    Use this endpoint to poll before sending the first /run request.
    """
    if not (_pipeline and _pipeline.is_ready):
        raise HTTPException(503, detail="Pipeline still initialising — try again in a moment")
    return {"status": "ready", "users": len(_pipeline.users)}


@app.get("/cache/stats", tags=["ops"])
def cache_stats():
    if not _pipeline:
        raise HTTPException(503, "Pipeline not ready")
    return _pipeline.cache_info


@app.get("/users", tags=["data"])
def get_users():
    if _pipeline is None:
        return {"users": []}
    return {"users": _pipeline.users}


@app.post("/run", response_model=RunResponse, tags=["pipeline"])
def run(body: RunRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    try:
        result = _pipeline.run(user_id=body.user_id, prompt=body.prompt)
        return RunResponse(
            user_name=result.get("user_name", ""),
            response=result.get("response", ""),
            data_summary=result.get("data_summary", {}),
            visualizations=result.get("visualizations", []),
            cache_hit=result.get("cache_hit", False),
            latency_ms=result.get("latency_ms", 0),
            guardrail_flags=result.get("guardrail_flags", []),
            error=result.get("error"),
        )
    except Exception as exc:
        logger.exception("Pipeline error for user_id=%s", body.user_id)
        return RunResponse(response="An internal error occurred.", error=str(exc))


@app.get("/charts/{filename}", tags=["charts"])
def get_chart(filename: str):
    output_dir = Path(Config.OUTPUT_DIR).resolve()
    requested = (output_dir / filename).resolve()
    try:
        requested.relative_to(output_dir)
    except ValueError:
        raise HTTPException(status_code=404, detail="File not found")
    if not requested.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(requested), media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host=Config.API_HOST, port=Config.API_PORT, reload=True)
