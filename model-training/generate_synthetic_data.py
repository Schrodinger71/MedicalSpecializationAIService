from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from medical_methodology import demo_profiles, predict_specialty  # noqa: E402

OUTPUT_FILE = Path(__file__).resolve().parent / "synthetic_med_students.csv"
SPECIALTY_RANGES = {
    "surgery": {
        "surgery_interest": (7, 10),
        "manual_dexterity": (7, 10),
        "stress_tolerance": (7, 10),
        "night_shift_readiness": (7, 10),
    },
    "cardiology": {
        "cardiology_interest": (7, 10),
        "physiology_score": (84, 100),
        "communication_skill": (7, 10),
        "analytical_thinking": (7, 10),
    },
    "neurology": {
        "neurology_interest": (7, 10),
        "research_orientation": (7, 10),
        "analytical_thinking": (8, 10),
        "precision_focus": (7, 10),
    },
    "otolaryngology": {
        "ent_interest": (7, 10),
        "auditory_attention": (8, 10),
        "communication_skill": (7, 10),
        "manual_dexterity": (6, 10),
    },
}


def make_variant(source: dict, index: int) -> dict:
    record = dict(source)
    record["student_name"] = f"{source['student_name']} #{index:03d}"
    record["course_year"] = random.randint(3, 6)
    record["academic_score"] = max(70, min(100, int(record["academic_score"] + random.randint(-4, 4))))
    record["anatomy_score"] = max(65, min(100, int(record["anatomy_score"] + random.randint(-6, 6))))
    record["physiology_score"] = max(65, min(100, int(record["physiology_score"] + random.randint(-6, 6))))

    for field_name in [
        "surgery_interest",
        "cardiology_interest",
        "neurology_interest",
        "ent_interest",
        "manual_dexterity",
        "stress_tolerance",
        "empathy",
        "analytical_thinking",
        "communication_skill",
        "research_orientation",
        "night_shift_readiness",
        "cardiovascular_endurance",
        "auditory_attention",
        "precision_focus",
    ]:
        record[field_name] = max(0, min(10, int(record[field_name] + random.randint(-2, 2))))

    specialty_key = predict_specialty(record)["predicted_specialty_key"]
    for field_name, (minimum, maximum) in SPECIALTY_RANGES[specialty_key].items():
        record[field_name] = random.randint(minimum, maximum)

    record["student_note"] = "Синтетически сгенерированный профиль для учебной DevOps-демонстрации."
    return record


def main() -> None:
    random.seed(42)
    base_profiles = [item for item in demo_profiles()]
    clean_profiles = []
    for item in base_profiles:
        profile = dict(item)
        profile.pop("id", None)
        profile.pop("created_at", None)
        profile.pop("predicted_specialty", None)
        profile.pop("predicted_specialty_key", None)
        profile.pop("confidence", None)
        profile.pop("score_breakdown", None)
        profile.pop("probabilities", None)
        profile.pop("top_specialties", None)
        profile.pop("key_factors", None)
        profile.pop("summary", None)
        profile.pop("recommendations", None)
        profile.pop("input_count", None)
        profile.pop("domain", None)
        clean_profiles.append(profile)

    generated_rows = []
    for index in range(5000):
        source = random.choice(clean_profiles)
        record = make_variant(source, index + 1)
        prediction = predict_specialty(record)
        generated_rows.append(
            {
                **record,
                "predicted_specialty_key": prediction["predicted_specialty_key"],
                "predicted_specialty": prediction["predicted_specialty"],
                "confidence": prediction["confidence"],
            }
        )

    fieldnames = list(generated_rows[0].keys())
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(generated_rows)

    print(f"Generated {len(generated_rows)} rows into {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
