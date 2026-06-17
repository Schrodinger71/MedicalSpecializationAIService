from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

DATASET_FILE = Path(__file__).resolve().parent / "synthetic_med_students.csv"
TARGET_FILE = Path(__file__).resolve().parents[1] / "inference-api" / "models" / "specialty_baseline.json"


def main() -> None:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_FILE}. Run generate_synthetic_data.py first."
        )

    counter: Counter[str] = Counter()
    with DATASET_FILE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            specialty_key = row.get("predicted_specialty_key", "").strip()
            if specialty_key:
                counter[specialty_key] += 1

    total = sum(counter.values()) or 1
    priors = {key: round(value / total, 4) for key, value in sorted(counter.items())}
    payload = {
        "model_name": "medical_specialty_baseline",
        "source_dataset": str(DATASET_FILE.name),
        "records_used": total,
        "class_priors": priors,
    }

    TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    TARGET_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Updated priors in {TARGET_FILE}")


if __name__ == "__main__":
    main()
