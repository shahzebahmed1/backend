#!/usr/bin/env python3
# Copyright 2025 Google
# Apache-2.0

"""
Veyda AI Retail Agent – FastAPI backend API for Vertex AI Agent Engine.
Run locally: uvicorn app:app --reload
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import suppress
from typing import Dict, Generator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import vertexai
from vertexai import agent_engines

# ──────────────────────────── config & Vertex init ───────────────────────────

load_dotenv()  # optional .env

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION   = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
BUCKET     = os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET", f"{PROJECT_ID}-adk-staging")
RESOURCE_ID = os.getenv("REASONING_ENGINE_RESOURCE_ID")   # full projects/…/reasoningEngines/…

if not all((PROJECT_ID, LOCATION, BUCKET, RESOURCE_ID)):
    raise RuntimeError(
        "Set GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, "
        "GOOGLE_CLOUD_STORAGE_BUCKET, and REASONING_ENGINE_RESOURCE_ID"
    )

vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=f"gs://{BUCKET}")
AGENT = agent_engines.get(RESOURCE_ID)      # ← SDK, not raw REST

# ──────────────────────────── session cache ──────────────────────────────────

_SESSION_CACHE: Dict[str, str] = {}
_CACHE_LOCK = threading.Lock()


def _get_session(user_id: str) -> str:
    with _CACHE_LOCK:
        if user_id not in _SESSION_CACHE:
            session = AGENT.create_session(user_id=user_id)
            _SESSION_CACHE[user_id] = session["id"]
        return _SESSION_CACHE[user_id]


def _delete_session(user_id: str) -> None:
    with suppress(KeyError):
        with _CACHE_LOCK:
            session_id = _SESSION_CACHE.pop(user_id)
        AGENT.delete_session(user_id=user_id, session_id=session_id)

# ──────────────────────────── SSE helper ─────────────────────────────────────

def _sse_stream(user_id: str, message: str) -> Generator[bytes, None, None]:
    """Yield Server-Sent Event frames understood by the frontend."""
    try:
        session_id = _get_session(user_id)
        yield b": ping\n\n"                                  # keep-alive

        for event in AGENT.stream_query(                     # SDK call
            user_id=user_id, session_id=session_id, message=message
        ):
            for part in event.get("content", {}).get("parts", []):
                if "text" in part:
                    data = json.dumps({"reply": part["text"]})
                    print(data)
                    yield f"data: {data}\n\n".encode()

        yield b'data: {"done": true}\n\n'

    except Exception as exc:
        logging.exception("stream_query failed")
        data = json.dumps({"error": str(exc)})
        yield f"data: {data}\n\n".encode()

# ──────────────────────────── FastAPI setup ──────────────────────────────────

app = FastAPI(title="Veyda AI Retail Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in prod
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# request-body schemas --------------------------------------------------------
class ChatPayload(BaseModel):
    message: str
    user_id: str

class SessionPayload(BaseModel):
    user_id: str

# chat endpoints --------------------------------------------------------------
@app.get("/chat")
async def chat_events(request: Request, user_id: str, message: str):
    if not message.strip():
        raise HTTPException(400, "Message is empty")

    async def event_gen():
        for chunk in _sse_stream(user_id, message):
            yield chunk
            if await request.is_disconnected():
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.post("/chat")
async def chat_trigger(payload: ChatPayload):
    """Frontend fires this POST; we simply acknowledge so its fetch() resolves."""
    return JSONResponse({"status": "streaming"}, status_code=202)

@app.post("/end_session")
async def end_session(payload: SessionPayload):
    _delete_session(payload.user_id)
    return JSONResponse({"status": "ended"})

# health check endpoint -------------------------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Veyda AI Retail Agent API"}

# dev entry-point -------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), reload=True) 