from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class TopKItem(BaseModel):
    intent: str
    confidence: float


class LatencyMetrics(BaseModel):
    extraction: Optional[float] = 0.0
    inference: Optional[float] = 0.0
    total: Optional[float] = 0.0


class PredictionResponse(BaseModel):
    status: str  # ok, low_confidence, low_quality, model_unavailable
    intent: str
    gloss: str
    text_zh: str
    confidence: float
    top_k: List[TopKItem] = []
    warnings: List[str] = []
    latency_ms: LatencyMetrics
    model_version: Optional[str] = None
