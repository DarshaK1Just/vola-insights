"""
streamlit_app.py — Streamlit Community Cloud entry point for Vola Insights.

This file is the deployment entry point used by Streamlit Community Cloud.
It runs the full pipeline DIRECTLY inside Streamlit (no separate FastAPI server needed).

For local development with the full stack (FastAPI + Streamlit), use:
    uvicorn api.app:app --port 8000
    streamlit run frontend/app.py

For Streamlit Cloud deployment, this file handles everything in one process.

Required secrets (add via Streamlit Cloud → Settings → Secrets):
    OPENROUTER_API_KEY = "sk-or-..."
    REDIS_URL          = ""        # leave blank — Redis not available on Cloud
    DATA_FILE          = ""        # leave blank — uses built-in demo data
"""
from __future__ import annotations
import os
import sys
import time
import threading
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Embedded mode: pipeline runs in-process (no separate FastAPI on port 8000)
os.environ.setdefault("VOLA_EMBEDDED", "1")

# ── Load Streamlit secrets into os.environ ────────────────────────────────────
# Streamlit Cloud stores secrets in st.secrets — copy them to env vars so the
# pipeline's Config class picks them up via os.environ.get(...)
try:
    import streamlit as st
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
        elif isinstance(_v, dict):
            for _sk, _sv in _v.items():
                if isinstance(_sv, str):
                    os.environ.setdefault(_sk, _sv)
except Exception:
    pass  # running locally — .env already loaded

import src.logging_config  # noqa: F401

# ── Lazy pipeline init (shared across reruns via st.session_state) ────────────
import streamlit as st
import pandas as pd

@st.cache_resource(show_spinner=False)
def _get_pipeline():
    """Load data and initialise pipeline once; cache the instance."""
    from src.config import Config
    from src.pipeline import TransactionRAGPipeline

    data_file = Config.DATA_FILE
    if data_file and Path(data_file).exists():
        from src.data_loader import load_transactions
        df = load_transactions(data_file)
    else:
        # Built-in demo dataset — works without any data file
        from api.app import _make_demo_df
        df = _make_demo_df()

    pipeline = TransactionRAGPipeline(df=df)
    # Wait for full init (guardrails + LangGraph) — max 3 min
    pipeline._ready.wait(timeout=180)
    return pipeline


# ── Override the API call used by the main frontend to call pipeline directly ─
# The frontend/app.py is designed to call a FastAPI backend via HTTP.
# On Streamlit Cloud there is no backend, so we monkey-patch call_pipeline_sync
# to call the pipeline directly.
def _direct_pipeline_call(backend_url: str, user_id: str, prompt: str) -> dict:
    """Replaces HTTP call with a direct pipeline.run() call."""
    pipeline = _get_pipeline()
    return pipeline.run(user_id=user_id, prompt=prompt)

# Inject the direct caller before importing the frontend
import types

# Keep the real httpx in sys.modules for pipeline / LangChain / OpenRouter.
# Only the Streamlit frontend gets a local shim (injected via exec globals).
import httpx as _real_httpx  # noqa: F401


class _FakeConnectError(Exception):
    pass


class _FakeTimeoutException(Exception):
    pass


class _FakeHTTPStatusError(Exception):
    def __init__(self, message: str, *, response=None):
        super().__init__(message)
        self.response = response


_fake_httpx = types.ModuleType("httpx")
_fake_httpx.ConnectError = _FakeConnectError
_fake_httpx.TimeoutException = _FakeTimeoutException
_fake_httpx.HTTPStatusError = _FakeHTTPStatusError
_fake_httpx.RequestError = _FakeConnectError

class _FakeResponse:
    def __init__(self, data=None, *, content: bytes | None = None, status_code: int = 200):
        self._data = data if data is not None else {}
        self._content = content
        self.status_code = status_code

    def json(self):
        return self._data

    @property
    def content(self) -> bytes:
        return self._content or b""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _FakeHTTPStatusError(
                f"HTTP {self.status_code}",
                response=self,
            )


def _fake_http_get(url: str, **_kwargs) -> _FakeResponse:
    with _FakeClient() as client:
        return client.get(url, **_kwargs)


def _fake_http_post(url: str, **kwargs) -> _FakeResponse:
    with _FakeClient() as client:
        return client.post(url, **kwargs)


class _FakeClient:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def get(self, url, **_):
        from src.config import Config
        pipeline = _get_pipeline()
        if "/users" in url:
            return _FakeResponse({"users": pipeline.users})
        if "/health" in url:
            return _FakeResponse({
                "status": "ok",
                "pipeline_ready": pipeline.is_ready,
                "data_rows": len(pipeline._df),
                "data_file": Config.DATA_FILE or "demo",
                "cache": pipeline.cache_info,
            })
        if "/charts/" in url:
            filename = url.rsplit("/charts/", 1)[-1].split("?")[0]
            output_dir = Path(Config.OUTPUT_DIR).resolve()
            chart_path = (output_dir / filename).resolve()
            try:
                chart_path.relative_to(output_dir)
            except ValueError:
                return _FakeResponse(status_code=404)
            if chart_path.is_file():
                return _FakeResponse(content=chart_path.read_bytes())
            return _FakeResponse(status_code=404)
        return _FakeResponse({})

    def post(self, url, json=None, **_):
        payload = json or {}
        result = _direct_pipeline_call("", payload.get("user_id", ""), payload.get("prompt", ""))
        return _FakeResponse(result)


class _FakeAsyncClient:
    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def post(self, url, json=None, **_):
        payload = json or {}
        result = _direct_pipeline_call("", payload.get("user_id", ""), payload.get("prompt", ""))
        return _FakeResponse(result)


_fake_httpx.Client = _FakeClient
_fake_httpx.AsyncClient = _FakeAsyncClient
_fake_httpx.get = _fake_http_get
_fake_httpx.post = _fake_http_post
_fake_httpx.Timeout = _real_httpx.Timeout

# Patch asyncio so the frontend's async call runs synchronously
import asyncio as _asyncio

def _patched_call(backend_url: str, user_id: str, prompt: str) -> dict:
    return _direct_pipeline_call(backend_url, user_id, prompt)

# ── Run the main Streamlit frontend ──────────────────────────────────────────
# Streamlit re-executes the entire script on each interaction, so we exec the
# frontend's app.py code directly in this module's global namespace.

_frontend_path = _ROOT / "frontend" / "app.py"
_frontend_code = _frontend_path.read_text(encoding="utf-8")

# Patch the call_pipeline_sync function that frontend uses
_frontend_code = _frontend_code.replace(
    "def call_pipeline_sync(backend_url: str, user_id: str, prompt: str) -> dict:",
    "def call_pipeline_sync(backend_url: str, user_id: str, prompt: str) -> dict:\n    return _patched_pipeline_call(backend_url, user_id, prompt)\ndef _unused_call_pipeline_sync(backend_url, user_id, prompt):"
)
# Frontend must use the shim; do not import real httpx (pipeline needs the real module).
_frontend_code = _frontend_code.replace("import httpx\n", "# httpx provided by embedded entry point\n")

# Inject the patched caller
_globals = {
    "__name__": "__main__",
    "__file__": str(_frontend_path),
    "_patched_pipeline_call": _patched_call,
    "_VOLA_EXEC_FROM_EMBED": True,
    "httpx": _fake_httpx,
}

exec(compile(_frontend_code, str(_frontend_path), "exec"), _globals)
