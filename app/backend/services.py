from __future__ import annotations

import os
from pathlib import Path

# Import both the original (for labels) and CTC service
from cslr.semantic import IntentCatalog
from cslr.inference.ctc_service import create_ctc_service

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_recognition_service():
    """
    Create a CTC-based recognition service for Web interface.
    Uses CTC ONNX model for Gloss sequence prediction.
    """
    labels_path = Path(
        os.getenv("CSLR_LABELS_PATH", str(PROJECT_ROOT / "configs/hospital_intents.yaml"))
    )
    
    # Create CTC service
    ctc_service = create_ctc_service()
    
    # Load intent catalog for mapping
    catalog = IntentCatalog.from_yaml(labels_path)
    
    return {
        'ctc_service': ctc_service,
        'catalog': catalog,
        'demo_mode': _as_bool(os.getenv("CSLR_DEMO_MODE", "false"))
    }
