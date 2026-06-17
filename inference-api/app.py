from __future__ import annotations

import json
import logging
import os
import sys
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from medical_methodology import (  # noqa: E402
    INPUT_FIELD_COUNT,
    RECOMMENDATIONS,
    SPECIALTIES,
    normalize_profile,
    predict_specialty,
)

app = FastAPI(
    title="Medical Student Specialization Inference API",
    version="1.0.0",
    description=(
        "API for predicting the most likely future medical specialty for a medical institute student "
        "based on 19 educational and competency indicators."
    ),
)

BASELINE_MODEL_PATH = Path(
    Path(__file__).resolve().parent / "models" / "specialty_baseline.json"
)
TRACE_LOGGING_ENABLED = os.getenv("DECISION_TRACE_LOGGING", "true").lower() == "true"
logger = logging.getLogger("medical-specialization-api")
logger.setLevel(logging.INFO)


class StudentProfilePayload(BaseModel):
    student_name: str = Field(..., min_length=3, max_length=120)
    course_year: int = Field(4, ge=1, le=6)
    academic_score: float = Field(..., ge=0, le=100)
    anatomy_score: float = Field(..., ge=0, le=100)
    physiology_score: float = Field(..., ge=0, le=100)
    surgery_interest: float = Field(..., ge=0, le=10)
    cardiology_interest: float = Field(..., ge=0, le=10)
    neurology_interest: float = Field(..., ge=0, le=10)
    ent_interest: float = Field(..., ge=0, le=10)
    manual_dexterity: float = Field(..., ge=0, le=10)
    stress_tolerance: float = Field(..., ge=0, le=10)
    empathy: float = Field(..., ge=0, le=10)
    analytical_thinking: float = Field(..., ge=0, le=10)
    communication_skill: float = Field(..., ge=0, le=10)
    research_orientation: float = Field(..., ge=0, le=10)
    night_shift_readiness: float = Field(..., ge=0, le=10)
    cardiovascular_endurance: float = Field(..., ge=0, le=10)
    auditory_attention: float = Field(..., ge=0, le=10)
    precision_focus: float = Field(..., ge=0, le=10)
    student_note: str = Field("", max_length=400)


def load_priors() -> dict[str, float]:
    if not BASELINE_MODEL_PATH.exists():
        return {key: 0.25 for key in SPECIALTIES}

    try:
        payload = json.loads(BASELINE_MODEL_PATH.read_text(encoding="utf-8"))
        priors = payload.get("class_priors", {})
        prepared = {key: float(priors.get(key, 0.25)) for key in SPECIALTIES}
        total = sum(prepared.values()) or 1.0
        return {key: round(value / total, 4) for key, value in prepared.items()}
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return {key: 0.25 for key in SPECIALTIES}


def log_prediction_trace(
    trace_id: str,
    normalized: dict,
    priors: dict[str, float],
    prediction: dict,
) -> None:
    if not TRACE_LOGGING_ENABLED:
        return

    log_payload = {
        "trace_id": trace_id,
        "event": "prediction_decision_trace",
        "student_name": normalized.get("student_name"),
        "course_year": normalized.get("course_year"),
        "normalized_profile": normalized,
        "class_priors": priors,
        "score_breakdown": prediction.get("score_breakdown", []),
        "top_specialties": prediction.get("top_specialties", []),
        "predicted_specialty_key": prediction.get("predicted_specialty_key"),
        "predicted_specialty": prediction.get("predicted_specialty"),
        "confidence": prediction.get("confidence"),
        "key_factors": prediction.get("key_factors", []),
    }
    serialized = json.dumps(log_payload, ensure_ascii=False)
    logger.info("prediction_trace=%s", serialized)
    print(f"prediction_trace={serialized}", flush=True)


@app.get("/health")
def health() -> dict:
    priors = load_priors()
    return {
        "status": "ok",
        "domain": "Лечение",
        "supported_specialties": list(SPECIALTIES.values()),
        "supported_specialty_count": len(SPECIALTIES),
        "input_field_count": INPUT_FIELD_COUNT,
        "model_loaded": BASELINE_MODEL_PATH.exists(),
        "model_source": str(BASELINE_MODEL_PATH.name),
        "model_role": "weighted_profile_assessor",
        "class_priors": priors,
    }


@app.post("/predict")
def predict(payload: StudentProfilePayload) -> dict:
    if hasattr(payload, "model_dump"):
        profile = payload.model_dump()
    else:
        profile = payload.dict()

    normalized = normalize_profile(profile)
    priors = load_priors()
    prediction = predict_specialty(normalized, priors=priors)
    trace_id = f"trace-{uuid4().hex[:10]}"
    log_prediction_trace(trace_id, normalized, priors, prediction)

    return {
        **prediction,
        "trace_id": trace_id,
        "normalized_profile": normalized,
        "model_loaded": BASELINE_MODEL_PATH.exists(),
        "model_source": BASELINE_MODEL_PATH.name,
        "model_role": "weighted_profile_assessor",
        "available_specialties": list(SPECIALTIES.values()),
        "recommendation_pool": RECOMMENDATIONS[prediction["predicted_specialty_key"]],
    }
