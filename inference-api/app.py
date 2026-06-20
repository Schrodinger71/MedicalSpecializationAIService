from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from medical_methodology import (  # noqa: E402
    FEATURE_INPUT_DIM,
    INPUT_FIELD_COUNT,
    RECOMMENDATIONS,
    SPECIALTIES,
    SPECIALTY_KEYS,
    SPECIALTY_NAMES,
    build_summary,
    feature_vector,
    heuristic_distribution,
    key_factors,
    named_distribution,
    normalize_profile,
    score_breakdown_from_distribution,
    specialty_scores,
)

try:
    from tensorflow import keras

    TENSORFLOW_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only when TF is missing
    keras = None
    TENSORFLOW_IMPORT_ERROR = str(exc)

try:
    import h5py
except Exception:  # pragma: no cover - exercised only when h5py is missing
    h5py = None


app = FastAPI(
    title="Medical Student Specialization Inference API",
    version="2.0.0",
    description=(
        "API for predicting the most likely future medical specialty for a medical institute student "
        "based on 19 educational and competency indicators. Combines a transparent weighted-profile "
        "heuristic with a trained neural network supporting classifier."
    ),
)

BASELINE_MODEL_PATH = Path(Path(__file__).resolve().parent / "models" / "specialty_baseline.json")
TRACE_LOGGING_ENABLED = os.getenv("DECISION_TRACE_LOGGING", "true").lower() == "true"
logger = logging.getLogger("medical-specialization-api")
logger.setLevel(logging.INFO)

MODEL = None
MODEL_SOURCE = "methodology_only"
MODEL_LOAD_ERRORS: list[str] = []
MODEL_USABLE = False
MODEL_DIAGNOSTIC = "not_loaded"

HEURISTIC_WEIGHT = 0.35
MODEL_WEIGHT = 0.65


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


def configured_model_paths() -> list[Path]:
    explicit_model = os.getenv("MODEL_PATH")
    base_dir = Path(__file__).resolve().parent

    paths: list[Path] = []
    if explicit_model:
        paths.append(Path(explicit_model))

    paths.extend(
        [
            base_dir / "models" / "model.keras",
            Path("/app/models/model.keras"),
            base_dir / "models" / "model.h5",
            Path("/app/models/model.h5"),
        ]
    )

    deduplicated: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(path)
    return deduplicated


def create_inference_model():
    if keras is None:
        raise ValueError("tensorflow/keras is unavailable.")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(FEATURE_INPUT_DIM,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.15),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(len(SPECIALTY_KEYS), activation="softmax"),
        ]
    )
    model(np.zeros((1, FEATURE_INPUT_DIM), dtype=np.float32))
    return model


def load_weights_only_from_h5(path: Path):
    model = create_inference_model()
    model.load_weights(path)
    return model


def load_weights_only_from_keras_archive(path: Path):
    if not zipfile.is_zipfile(path):
        raise ValueError("File is not a valid .keras archive.")

    with zipfile.ZipFile(path, "r") as archive:
        if "model.weights.h5" not in set(archive.namelist()):
            raise ValueError("The .keras archive does not contain model.weights.h5.")

        with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp:
            tmp.write(archive.read("model.weights.h5"))
            tmp_path = tmp.name

    try:
        model = create_inference_model()
        model.load_weights(tmp_path)
        return model
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def evaluate_model_health(model) -> tuple[bool, str]:
    from medical_methodology import demo_profiles

    samples = demo_profiles()[:6]
    vectors = np.stack([feature_vector(profile) for profile in samples]).astype(np.float32)

    try:
        predictions = np.array(model.predict(vectors, verbose=0), dtype=np.float32)
    except Exception as exc:
        return False, f"prediction_failed: {exc}"

    if predictions.ndim != 2 or predictions.shape[1] != len(SPECIALTY_KEYS):
        return False, "unexpected_output_shape"

    top_classes = np.argmax(predictions, axis=1)
    top_confidences = np.max(predictions, axis=1)
    if len(set(top_classes.tolist())) == 1 and float(np.min(top_confidences)) >= 0.99:
        return False, "degenerate_constant_prediction"
    return True, "ok"


def load_model() -> None:
    global MODEL, MODEL_SOURCE, MODEL_LOAD_ERRORS, MODEL_USABLE, MODEL_DIAGNOSTIC

    MODEL = None
    MODEL_LOAD_ERRORS = []
    MODEL_USABLE = False
    MODEL_DIAGNOSTIC = "not_loaded"

    if keras is None:
        MODEL_SOURCE = "tensorflow_unavailable"
        if TENSORFLOW_IMPORT_ERROR:
            MODEL_LOAD_ERRORS.append(f"tensorflow import failed: {TENSORFLOW_IMPORT_ERROR}")
        MODEL_DIAGNOSTIC = "tensorflow_unavailable"
        return

    for path in configured_model_paths():
        if not path.exists():
            continue

        try:
            MODEL = keras.models.load_model(path, compile=False)
            MODEL_SOURCE = str(path)
            MODEL_USABLE, MODEL_DIAGNOSTIC = evaluate_model_health(MODEL)
            return
        except Exception as exc:
            MODEL_LOAD_ERRORS.append(f"{path}: {exc}")
            if path.suffix == ".keras":
                try:
                    MODEL = load_weights_only_from_keras_archive(path)
                    MODEL_SOURCE = f"{path} (weights-only fallback)"
                    MODEL_USABLE, MODEL_DIAGNOSTIC = evaluate_model_health(MODEL)
                    return
                except Exception as weights_exc:
                    MODEL_LOAD_ERRORS.append(f"{path} weights-only fallback: {weights_exc}")
            elif path.suffix == ".h5":
                try:
                    MODEL = load_weights_only_from_h5(path)
                    MODEL_SOURCE = f"{path} (weights-only fallback)"
                    MODEL_USABLE, MODEL_DIAGNOSTIC = evaluate_model_health(MODEL)
                    return
                except Exception as weights_exc:
                    MODEL_LOAD_ERRORS.append(f"{path} weights-only fallback: {weights_exc}")

    if MODEL_LOAD_ERRORS:
        MODEL_SOURCE = "model_load_failed"
        MODEL_DIAGNOSTIC = "load_failed"
        return

    MODEL_SOURCE = "model_file_not_found"
    MODEL_DIAGNOSTIC = "file_not_found"


def model_distribution(features: np.ndarray) -> np.ndarray:
    uniform = np.full(len(SPECIALTY_KEYS), 1.0 / len(SPECIALTY_KEYS), dtype=np.float32)
    if MODEL is None or not MODEL_USABLE:
        return uniform

    try:
        prediction = MODEL.predict(np.array([features], dtype=np.float32), verbose=0)[0]
        prediction = np.array(prediction, dtype=np.float32)
        if prediction.shape[0] != len(SPECIALTY_KEYS):
            return uniform
        total = float(prediction.sum())
        if total <= 0:
            return uniform
        return prediction / total
    except Exception:
        return uniform


def log_prediction_trace(trace_id: str, normalized: dict, priors: dict[str, float], result: dict) -> None:
    if not TRACE_LOGGING_ENABLED:
        return

    log_payload = {
        "trace_id": trace_id,
        "event": "prediction_decision_trace",
        "student_name": normalized.get("student_name"),
        "course_year": normalized.get("course_year"),
        "normalized_profile": normalized,
        "class_priors": priors,
        "predicted_specialty_key": result.get("predicted_specialty_key"),
        "predicted_specialty": result.get("predicted_specialty"),
        "confidence": result.get("confidence"),
        "model_usable": MODEL_USABLE,
        "model_source": MODEL_SOURCE,
        "key_factors": result.get("key_factors", []),
    }
    serialized = json.dumps(log_payload, ensure_ascii=False)
    logger.info("prediction_trace=%s", serialized)
    print(f"prediction_trace={serialized}", flush=True)


@app.on_event("startup")
def startup_event() -> None:
    load_model()


@app.get("/health")
def health() -> dict:
    priors = load_priors()
    return {
        "status": "ok",
        "domain": "Лечение",
        "supported_specialties": SPECIALTY_NAMES,
        "supported_specialty_count": len(SPECIALTY_KEYS),
        "input_field_count": INPUT_FIELD_COUNT,
        "feature_input_dim": FEATURE_INPUT_DIM,
        "model_loaded": MODEL is not None,
        "model_usable": MODEL_USABLE,
        "model_source": MODEL_SOURCE,
        "model_diagnostic": MODEL_DIAGNOSTIC,
        "model_load_errors": MODEL_LOAD_ERRORS,
        "model_role": "supporting_classifier" if MODEL_USABLE else "weighted_profile_assessor",
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

    heuristic = heuristic_distribution(normalized, priors=priors)
    features = feature_vector(normalized)
    model_probs = model_distribution(features)

    combined = (heuristic * HEURISTIC_WEIGHT) + (model_probs * MODEL_WEIGHT)
    combined = combined / float(combined.sum() or 1.0)

    predicted_index = int(np.argmax(combined))
    specialty_key = SPECIALTY_KEYS[predicted_index]
    confidence = float(round(combined[predicted_index], 4))

    scores = specialty_scores(normalized)

    trace_id = f"trace-{uuid4().hex[:10]}"
    result = {
        "predicted_specialty_key": specialty_key,
        "predicted_specialty": SPECIALTIES[specialty_key],
        "confidence": confidence,
        "score_breakdown": score_breakdown_from_distribution(scores, combined),
        "probabilities": named_distribution(combined),
        "heuristic_probabilities": named_distribution(heuristic),
        "model_probabilities": named_distribution(model_probs),
        "combined_probabilities": named_distribution(combined),
        "top_specialties": [
            {
                "specialty_key": SPECIALTY_KEYS[index],
                "specialty_name": SPECIALTIES[SPECIALTY_KEYS[index]],
                "probability": float(round(combined[index], 4)),
            }
            for index in np.argsort(combined)[::-1]
        ],
        "key_factors": key_factors(normalized, specialty_key),
        "summary": build_summary(normalized, specialty_key, confidence),
        "recommendations": RECOMMENDATIONS[specialty_key],
        "input_count": INPUT_FIELD_COUNT,
        "domain": "Лечение",
    }
    log_prediction_trace(trace_id, normalized, priors, result)

    return {
        **result,
        "trace_id": trace_id,
        "normalized_profile": normalized,
        "model_loaded": MODEL is not None,
        "model_usable": MODEL_USABLE,
        "model_source": MODEL_SOURCE,
        "model_diagnostic": MODEL_DIAGNOSTIC,
        "model_role": "supporting_classifier" if MODEL_USABLE else "weighted_profile_assessor",
        "available_specialties": SPECIALTY_NAMES,
        "recommendation_pool": RECOMMENDATIONS[specialty_key],
    }
