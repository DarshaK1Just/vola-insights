"""
In-process backend for Streamlit Cloud (no FastAPI on localhost:8000).

Keeps the real ``httpx`` package untouched so LangGraph / LangChain can import it.
"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_pipeline = None


def get_pipeline():
    """Lazy singleton — initialised once per Streamlit process."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _load():
        from src.config import Config
        from src.pipeline import TransactionRAGPipeline

        data_file = Config.DATA_FILE
        if data_file and Path(data_file).exists():
            from src.data_loader import load_transactions
            df = load_transactions(data_file)
        else:
            from api.app import _make_demo_df
            df = _make_demo_df()

        pipe = TransactionRAGPipeline(df=df)
        pipe._ready.wait(timeout=180)
        return pipe

    _pipeline = _load()
    return _pipeline


def fetch_users() -> list[dict]:
    return get_pipeline().users


def fetch_health() -> dict:
    from src.config import Config

    pipe = get_pipeline()
    return {
        "status": "ok",
        "pipeline_ready": pipe.is_ready,
        "data_rows": len(pipe._df),
        "data_file": Config.DATA_FILE or "demo",
        "cache": pipe.cache_info,
    }


def run_pipeline(user_id: str, prompt: str) -> dict:
    return get_pipeline().run(user_id=user_id, prompt=prompt)


def fetch_chart_bytes(filename: str) -> bytes | None:
    from src.config import Config

    output_dir = Path(Config.OUTPUT_DIR).resolve()
    chart_path = (output_dir / filename).resolve()
    try:
        chart_path.relative_to(output_dir)
    except ValueError:
        return None
    if chart_path.is_file():
        return chart_path.read_bytes()
    return None
