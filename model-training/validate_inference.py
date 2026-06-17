from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_FILE = PROJECT_ROOT / "inference-api" / "app.py"


def load_app():
    spec = importlib.util.spec_from_file_location("medical_inference_app", APP_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load inference API module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


def main() -> None:
    app = load_app()
    client = TestClient(app)

    health = client.get("/health")
    print("Health:", health.status_code, health.json())

    payload = {
        "student_name": "Проверочный студент",
        "course_year": 5,
        "academic_score": 91,
        "anatomy_score": 94,
        "physiology_score": 86,
        "surgery_interest": 9,
        "cardiology_interest": 5,
        "neurology_interest": 4,
        "ent_interest": 2,
        "manual_dexterity": 9,
        "stress_tolerance": 8,
        "empathy": 6,
        "analytical_thinking": 7,
        "communication_skill": 6,
        "research_orientation": 4,
        "night_shift_readiness": 9,
        "cardiovascular_endurance": 8,
        "auditory_attention": 4,
        "precision_focus": 9,
        "student_note": "Тестовый кейс.",
    }

    prediction = client.post("/predict", json=payload)
    print("Predict:", prediction.status_code, prediction.json())


if __name__ == "__main__":
    main()
