from __future__ import annotations

import importlib.util
import json
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


def build_payload() -> dict:
    return {
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


def main() -> None:
    app = load_app()

    with TestClient(app) as client:
        health_response = client.get("/health")
        health_data = health_response.json()
        print("=== HEALTH ===")
        print(json.dumps(health_data, ensure_ascii=False, indent=2))

        predict_response = client.post("/predict", json=build_payload())
        predict_data = predict_response.json()
        print("\n=== PREDICT ===")
        print(json.dumps(predict_data, ensure_ascii=False, indent=2))

    if health_response.status_code != 200:
        print(f"\nПРОВЕРКА НЕ ПРОЙДЕНА: /health вернул {health_response.status_code}.")
        raise SystemExit(1)

    if predict_response.status_code != 200:
        print(f"\nПРОВЕРКА НЕ ПРОЙДЕНА: /predict вернул {predict_response.status_code}.")
        raise SystemExit(1)

    if not health_data.get("model_loaded") or not health_data.get("model_usable"):
        print("\nПРЕДУПРЕЖДЕНИЕ: API не считает нейросетевую модель пригодной для использования.")
        print("Сервис продолжит работать на одной методике (weighted_profile_assessor).")
    elif predict_data.get("model_role") != "supporting_classifier":
        print("\nПРОВЕРКА НЕ ПРОЙДЕНА: модель загружена, но не используется как supporting_classifier.")
        raise SystemExit(1)
    else:
        print("\nПРОВЕРКА ПРОЙДЕНА: API загрузил и использует обученную модель.")


if __name__ == "__main__":
    main()
