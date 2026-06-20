from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException
import numpy as np

from app.backend.services import create_recognition_service
from app.backend.schemas import PredictionResponse
from cslr.features.extractor import MediaPipeHolisticExtractor

router = APIRouter()

_service = None

def get_service():
    global _service
    if _service is None:
        _service = create_recognition_service()
    return _service


@router.get("/health")
async def health_check():
    try:
        service = get_service()
        # Check if model is available
        model_ready = service.get('ctc_service') is not None
        demo_mode = service.get('demo_mode', False)
        return {
            "status": "healthy",
            "model_ready": model_ready,
            "demo_mode": demo_mode
        }
    except Exception as e:
        return {
            "status": "healthy",
            "model_ready": False,
            "demo_mode": False,
            "error": str(e)
        }


@router.post("/predict")
async def predict(video: UploadFile = File(...)):
    try:
        service = get_service()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await video.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        
        try:
            extractor = MediaPipeHolisticExtractor()
            extraction = extractor.extract(tmp_path)
            
            if not extraction.quality.accepted:
                return PredictionResponse(
                    status="low_quality",
                    intent="unknown",
                    gloss="UNKNOWN",
                    text_zh="视频质量不足，请重新录制。",
                    confidence=0.0,
                    top_k=[],
                    warnings=extraction.quality.warnings,
                    latency_ms={"extraction": 0, "inference": 0, "total": 0},
                    model_version="ctc"
                )
            
            result = service['ctc_service'].predict(extraction.features)
            gloss_sequence = result['gloss_sequence']
            gloss_list = result['gloss_list']
            
            intent_map = {
                "挂号": "REGISTRATION", "预约": "APPOINTMENT", "缴费": "PAYMENT",
                "药": "PHARMACY", "药房": "PHARMACY", "疼": "PAIN", "痛": "PAIN",
                "难受": "PAIN", "帮助": "HELP_REQUEST", "帮": "HELP_REQUEST",
                "时间": "TIME_QUERY", "哪里": "LOCATION_QUERY", "怎么": "QUESTION",
                "什么": "QUESTION", "紧急": "EMERGENCY", "救命": "EMERGENCY",
                "谢谢": "GREETING", "你好": "GREETING", "再见": "GREETING"
            }
            
            intent = "OTHER"
            for word in gloss_list:
                if word in intent_map:
                    intent = intent_map[word]
                    break
            
            catalog = service['catalog']
            template = catalog.reconstruct(intent, 0.9, 0.65)
            
            return {
                "status": "ok",
                "intent": template.intent,
                "gloss": template.gloss,
                "gloss_sequence": gloss_sequence,
                "text_zh": template.text_zh if template.intent != "unknown" else gloss_sequence,
                "confidence": 0.9,
                "top_k": [{"intent": intent, "confidence": 0.9}],
                "warnings": [],
                "latency_ms": {"extraction": 0, "inference": result['inference_time_ms'], "total": result['inference_time_ms']},
                "model_version": "ctc"
            }
            
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
                
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
