from __future__ import annotations

from datetime import datetime, timedelta
from math import exp
from typing import Mapping

import numpy as np

SPECIALTIES = {
    "surgery": "Хирург",
    "cardiology": "Кардиолог",
    "neurology": "Невролог",
    "otolaryngology": "Отоларинголог",
}

SPECIALTY_KEYS = list(SPECIALTIES.keys())
SPECIALTY_NAMES = [SPECIALTIES[key] for key in SPECIALTY_KEYS]
SPECIALTY_INDEX = {key: index for index, key in enumerate(SPECIALTY_KEYS)}

FEATURE_FIELDS = [
    "course_year",
    "academic_score",
    "anatomy_score",
    "physiology_score",
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
]
FEATURE_INPUT_DIM = len(FEATURE_FIELDS)

SPECIALTY_COLORS = {
    "surgery": "#b4492e",
    "cardiology": "#8f2f45",
    "neurology": "#365c8d",
    "otolaryngology": "#567341",
}

INPUT_FIELD_COUNT = 19

FIELD_LABELS = {
    "academic_score": "Средний академический балл",
    "anatomy_score": "Анатомия",
    "physiology_score": "Физиология",
    "surgery_interest": "Интерес к хирургии",
    "cardiology_interest": "Интерес к кардиологии",
    "neurology_interest": "Интерес к неврологии",
    "ent_interest": "Интерес к отоларингологии",
    "manual_dexterity": "Мануальная точность",
    "stress_tolerance": "Стрессоустойчивость",
    "empathy": "Эмпатия",
    "analytical_thinking": "Аналитическое мышление",
    "communication_skill": "Коммуникация с пациентом",
    "research_orientation": "Исследовательская направленность",
    "night_shift_readiness": "Готовность к дежурствам",
    "cardiovascular_endurance": "Выносливость",
    "auditory_attention": "Слуховая внимательность",
    "precision_focus": "Концентрация на деталях",
}

FIELD_LIMITS = {
    "course_year": (1, 6),
    "academic_score": (0, 100),
    "anatomy_score": (0, 100),
    "physiology_score": (0, 100),
    "surgery_interest": (0, 10),
    "cardiology_interest": (0, 10),
    "neurology_interest": (0, 10),
    "ent_interest": (0, 10),
    "manual_dexterity": (0, 10),
    "stress_tolerance": (0, 10),
    "empathy": (0, 10),
    "analytical_thinking": (0, 10),
    "communication_skill": (0, 10),
    "research_orientation": (0, 10),
    "night_shift_readiness": (0, 10),
    "cardiovascular_endurance": (0, 10),
    "auditory_attention": (0, 10),
    "precision_focus": (0, 10),
}

WEIGHTS = {
    "surgery": {
        "anatomy_score": 1.45,
        "surgery_interest": 1.55,
        "manual_dexterity": 1.35,
        "stress_tolerance": 1.10,
        "night_shift_readiness": 1.05,
        "precision_focus": 1.00,
        "cardiovascular_endurance": 0.90,
        "academic_score": 0.70,
        "communication_skill": 0.35,
        "course_year": 0.15,
    },
    "cardiology": {
        "physiology_score": 1.45,
        "cardiology_interest": 1.60,
        "analytical_thinking": 1.20,
        "empathy": 1.05,
        "communication_skill": 1.10,
        "cardiovascular_endurance": 1.00,
        "stress_tolerance": 0.90,
        "academic_score": 0.80,
        "research_orientation": 0.50,
        "course_year": 0.10,
    },
    "neurology": {
        "physiology_score": 1.15,
        "neurology_interest": 1.65,
        "analytical_thinking": 1.45,
        "research_orientation": 1.20,
        "empathy": 0.90,
        "precision_focus": 1.00,
        "communication_skill": 0.70,
        "academic_score": 0.80,
        "stress_tolerance": 0.55,
        "course_year": 0.10,
    },
    "otolaryngology": {
        "ent_interest": 1.65,
        "auditory_attention": 1.35,
        "manual_dexterity": 1.05,
        "communication_skill": 1.05,
        "empathy": 0.95,
        "precision_focus": 0.95,
        "anatomy_score": 0.85,
        "physiology_score": 0.70,
        "stress_tolerance": 0.55,
        "academic_score": 0.55,
    },
}

RECOMMENDATIONS = {
    "surgery": [
        "Усилить практику в операционном блоке и на симуляторах.",
        "Добавить больше кейсов на работу в стрессовых и ночных сменах.",
        "Поддерживать высокий уровень анатомической подготовки.",
    ],
    "cardiology": [
        "Сделать акцент на клиническом разборе ЭКГ и патофизиологии.",
        "Углублять навыки общения с пациентами и ведения длительного наблюдения.",
        "Развивать аналитическое мышление на диагностических сценариях.",
    ],
    "neurology": [
        "Расширять исследовательскую практику и работу с клиническими гипотезами.",
        "Тренировать внимательность к симптомокомплексам и тонкой диагностике.",
        "Углублять знания по нейрофизиологии и смежным дисциплинам.",
    ],
    "otolaryngology": [
        "Развивать точность манипуляций и навыки микроосмотра.",
        "Уделять внимание слуховой диагностике и коммуникации с пациентами.",
        "Укреплять анатомическую базу в области головы и шеи.",
    ],
}


def clamp(value: float | int, minimum: float, maximum: float) -> float:
    return max(minimum, min(float(value), maximum))


def scale_score(value: float | int) -> float:
    return clamp(float(value) / 10.0, 0.0, 10.0)


def scale_metric(value: float | int) -> float:
    return clamp(value, 0.0, 10.0)


def normalize_profile(profile: dict) -> dict:
    normalized = {
        "student_name": str(profile.get("student_name", "Студент")).strip() or "Студент",
        "course_year": int(clamp(profile.get("course_year", 4), *FIELD_LIMITS["course_year"])),
        "student_note": str(profile.get("student_note", "")).strip(),
    }

    for field_name in FIELD_LIMITS:
        if field_name == "course_year":
            continue
        minimum, maximum = FIELD_LIMITS[field_name]
        value = profile.get(field_name, minimum)
        normalized[field_name] = round(clamp(value, minimum, maximum), 2)

    return normalized


def scoring_vector(profile: dict) -> dict[str, float]:
    return {
        "course_year": scale_metric(profile["course_year"]),
        "academic_score": scale_score(profile["academic_score"]),
        "anatomy_score": scale_score(profile["anatomy_score"]),
        "physiology_score": scale_score(profile["physiology_score"]),
        "surgery_interest": scale_metric(profile["surgery_interest"]),
        "cardiology_interest": scale_metric(profile["cardiology_interest"]),
        "neurology_interest": scale_metric(profile["neurology_interest"]),
        "ent_interest": scale_metric(profile["ent_interest"]),
        "manual_dexterity": scale_metric(profile["manual_dexterity"]),
        "stress_tolerance": scale_metric(profile["stress_tolerance"]),
        "empathy": scale_metric(profile["empathy"]),
        "analytical_thinking": scale_metric(profile["analytical_thinking"]),
        "communication_skill": scale_metric(profile["communication_skill"]),
        "research_orientation": scale_metric(profile["research_orientation"]),
        "night_shift_readiness": scale_metric(profile["night_shift_readiness"]),
        "cardiovascular_endurance": scale_metric(profile["cardiovascular_endurance"]),
        "auditory_attention": scale_metric(profile["auditory_attention"]),
        "precision_focus": scale_metric(profile["precision_focus"]),
    }


def feature_vector(profile: Mapping) -> np.ndarray:
    normalized = normalize_profile(dict(profile))
    prepared = scoring_vector(normalized)
    return np.array([prepared[field] / 10.0 for field in FEATURE_FIELDS], dtype=np.float32)


def specialty_scores(profile: dict) -> dict[str, float]:
    prepared = scoring_vector(profile)
    scores: dict[str, float] = {}

    for specialty_key, weights in WEIGHTS.items():
        total = 0.0
        weight_sum = 0.0
        for field_name, weight in weights.items():
            total += prepared[field_name] * weight
            weight_sum += weight
        scores[specialty_key] = round(total / weight_sum, 4) if weight_sum else 0.0

    return scores


def probability_distribution(scores: dict[str, float], priors: dict[str, float] | None = None) -> dict[str, float]:
    if not scores:
        return {}

    max_score = max(scores.values())
    raw: dict[str, float] = {}
    for specialty_key, score in scores.items():
        prior = 1.0
        if priors:
            prior = max(float(priors.get(specialty_key, 0.25)), 0.01)
        raw[specialty_key] = exp((score - max_score) / 1.4) * prior

    total = sum(raw.values()) or 1.0
    return {key: round(value / total, 4) for key, value in raw.items()}


def heuristic_distribution(profile: dict, priors: dict[str, float] | None = None) -> np.ndarray:
    normalized = normalize_profile(profile)
    scores = specialty_scores(normalized)
    probabilities = probability_distribution(scores, priors=priors)
    return np.array([probabilities[key] for key in SPECIALTY_KEYS], dtype=np.float32)


def named_distribution(values: np.ndarray) -> dict[str, float]:
    return {
        SPECIALTIES[SPECIALTY_KEYS[index]]: float(round(values[index], 4))
        for index in range(min(len(values), len(SPECIALTY_KEYS)))
    }


def score_breakdown_from_distribution(scores: dict[str, float], distribution: np.ndarray) -> list[dict]:
    return [
        {
            "specialty_key": key,
            "specialty_name": SPECIALTIES[key],
            "score": scores[key],
            "probability": float(round(distribution[SPECIALTY_INDEX[key]], 4)),
        }
        for key in SPECIALTY_KEYS
    ]


def top_specialties(probabilities: dict[str, float]) -> list[dict]:
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return [
        {
            "specialty_key": specialty_key,
            "specialty_name": SPECIALTIES[specialty_key],
            "probability": round(probability, 4),
        }
        for specialty_key, probability in ordered
    ]


def factor_text(field_name: str, raw_value: float | int) -> str:
    label = FIELD_LABELS.get(field_name, field_name)
    if field_name.endswith("_score"):
        return f"{label}: {int(raw_value)}/100"
    return f"{label}: {round(float(raw_value), 1)}/10"


def key_factors(profile: dict, specialty_key: str) -> list[str]:
    factors: list[tuple[float, str]] = []
    weights = WEIGHTS[specialty_key]
    prepared = scoring_vector(profile)

    for field_name, weight in weights.items():
        contribution = prepared[field_name] * weight
        if field_name == "course_year":
            text = f"Курс обучения: {int(profile['course_year'])}"
        else:
            text = factor_text(field_name, profile[field_name])
        factors.append((contribution, text))

    factors.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in factors[:5]]


def build_summary(profile: dict, specialty_key: str, probability: float) -> str:
    label = SPECIALTIES[specialty_key]
    return (
        f"Профиль студента {profile['student_name']} ближе всего к специализации "
        f"«{label}». Уверенность модели составляет {probability:.0%}. "
        f"Решающими оказались сочетание профильных интересов, базовых дисциплин и клинических навыков."
    )


def predict_specialty(profile: dict, priors: dict[str, float] | None = None) -> dict:
    normalized = normalize_profile(profile)
    scores = specialty_scores(normalized)
    probabilities = probability_distribution(scores, priors=priors)
    ordered = top_specialties(probabilities)
    best = ordered[0]
    specialty_key = best["specialty_key"]

    return {
        "predicted_specialty_key": specialty_key,
        "predicted_specialty": SPECIALTIES[specialty_key],
        "confidence": round(best["probability"], 4),
        "score_breakdown": [
            {
                "specialty_key": key,
                "specialty_name": SPECIALTIES[key],
                "score": scores[key],
                "probability": probabilities[key],
            }
            for key in SPECIALTIES
        ],
        "probabilities": {SPECIALTIES[key]: value for key, value in probabilities.items()},
        "top_specialties": ordered,
        "key_factors": key_factors(normalized, specialty_key),
        "summary": build_summary(normalized, specialty_key, best["probability"]),
        "recommendations": RECOMMENDATIONS[specialty_key],
        "input_count": INPUT_FIELD_COUNT,
        "domain": "Лечение",
    }


def demo_profiles() -> list[dict]:
    start = datetime(2026, 6, 1, 9, 0, 0)
    profiles = [
        {
            "student_name": "Анна Котова",
            "course_year": 5,
            "academic_score": 89,
            "anatomy_score": 95,
            "physiology_score": 81,
            "surgery_interest": 10,
            "cardiology_interest": 5,
            "neurology_interest": 4,
            "ent_interest": 3,
            "manual_dexterity": 9,
            "stress_tolerance": 9,
            "empathy": 6,
            "analytical_thinking": 7,
            "communication_skill": 6,
            "research_orientation": 5,
            "night_shift_readiness": 10,
            "cardiovascular_endurance": 8,
            "auditory_attention": 5,
            "precision_focus": 9,
            "student_note": "Сильна в практических манипуляциях.",
        },
        {
            "student_name": "Дмитрий Орлов",
            "course_year": 6,
            "academic_score": 92,
            "anatomy_score": 84,
            "physiology_score": 96,
            "surgery_interest": 5,
            "cardiology_interest": 10,
            "neurology_interest": 6,
            "ent_interest": 2,
            "manual_dexterity": 7,
            "stress_tolerance": 8,
            "empathy": 8,
            "analytical_thinking": 9,
            "communication_skill": 9,
            "research_orientation": 7,
            "night_shift_readiness": 7,
            "cardiovascular_endurance": 8,
            "auditory_attention": 5,
            "precision_focus": 8,
            "student_note": "Любит диагностику и клинический анализ.",
        },
        {
            "student_name": "Елена Смирнова",
            "course_year": 6,
            "academic_score": 94,
            "anatomy_score": 80,
            "physiology_score": 94,
            "surgery_interest": 3,
            "cardiology_interest": 6,
            "neurology_interest": 10,
            "ent_interest": 3,
            "manual_dexterity": 6,
            "stress_tolerance": 7,
            "empathy": 8,
            "analytical_thinking": 10,
            "communication_skill": 7,
            "research_orientation": 10,
            "night_shift_readiness": 5,
            "cardiovascular_endurance": 6,
            "auditory_attention": 6,
            "precision_focus": 9,
            "student_note": "Сильна в исследовательских задачах.",
        },
        {
            "student_name": "Ирина Белова",
            "course_year": 4,
            "academic_score": 86,
            "anatomy_score": 87,
            "physiology_score": 82,
            "surgery_interest": 4,
            "cardiology_interest": 4,
            "neurology_interest": 5,
            "ent_interest": 10,
            "manual_dexterity": 8,
            "stress_tolerance": 7,
            "empathy": 8,
            "analytical_thinking": 7,
            "communication_skill": 9,
            "research_orientation": 5,
            "night_shift_readiness": 6,
            "cardiovascular_endurance": 5,
            "auditory_attention": 10,
            "precision_focus": 8,
            "student_note": "Тяготеет к микроманипуляциям и ЛОР-практике.",
        },
        {
            "student_name": "Михаил Соколов",
            "course_year": 5,
            "academic_score": 88,
            "anatomy_score": 93,
            "physiology_score": 85,
            "surgery_interest": 9,
            "cardiology_interest": 6,
            "neurology_interest": 4,
            "ent_interest": 3,
            "manual_dexterity": 9,
            "stress_tolerance": 8,
            "empathy": 5,
            "analytical_thinking": 7,
            "communication_skill": 6,
            "research_orientation": 4,
            "night_shift_readiness": 9,
            "cardiovascular_endurance": 9,
            "auditory_attention": 4,
            "precision_focus": 9,
            "student_note": "Любит динамичные клинические задачи.",
        },
        {
            "student_name": "Софья Романова",
            "course_year": 5,
            "academic_score": 90,
            "anatomy_score": 82,
            "physiology_score": 95,
            "surgery_interest": 4,
            "cardiology_interest": 9,
            "neurology_interest": 7,
            "ent_interest": 3,
            "manual_dexterity": 6,
            "stress_tolerance": 8,
            "empathy": 9,
            "analytical_thinking": 9,
            "communication_skill": 10,
            "research_orientation": 7,
            "night_shift_readiness": 7,
            "cardiovascular_endurance": 8,
            "auditory_attention": 5,
            "precision_focus": 8,
            "student_note": "Хорошо работает с пациентскими сценариями.",
        },
        {
            "student_name": "Никита Жуков",
            "course_year": 6,
            "academic_score": 91,
            "anatomy_score": 79,
            "physiology_score": 92,
            "surgery_interest": 2,
            "cardiology_interest": 6,
            "neurology_interest": 9,
            "ent_interest": 3,
            "manual_dexterity": 5,
            "stress_tolerance": 7,
            "empathy": 7,
            "analytical_thinking": 10,
            "communication_skill": 7,
            "research_orientation": 9,
            "night_shift_readiness": 5,
            "cardiovascular_endurance": 5,
            "auditory_attention": 6,
            "precision_focus": 9,
            "student_note": "Силен в клиническом анализе и наблюдении.",
        },
        {
            "student_name": "Мария Тихонова",
            "course_year": 4,
            "academic_score": 84,
            "anatomy_score": 85,
            "physiology_score": 80,
            "surgery_interest": 3,
            "cardiology_interest": 4,
            "neurology_interest": 5,
            "ent_interest": 9,
            "manual_dexterity": 8,
            "stress_tolerance": 6,
            "empathy": 9,
            "analytical_thinking": 7,
            "communication_skill": 9,
            "research_orientation": 4,
            "night_shift_readiness": 5,
            "cardiovascular_endurance": 5,
            "auditory_attention": 9,
            "precision_focus": 8,
            "student_note": "Хороший контакт с пациентом и точность движений.",
        },
        {
            "student_name": "Павел Громов",
            "course_year": 5,
            "academic_score": 87,
            "anatomy_score": 92,
            "physiology_score": 83,
            "surgery_interest": 9,
            "cardiology_interest": 5,
            "neurology_interest": 4,
            "ent_interest": 2,
            "manual_dexterity": 10,
            "stress_tolerance": 9,
            "empathy": 5,
            "analytical_thinking": 7,
            "communication_skill": 5,
            "research_orientation": 4,
            "night_shift_readiness": 9,
            "cardiovascular_endurance": 8,
            "auditory_attention": 4,
            "precision_focus": 10,
            "student_note": "Предпочитает практические клинические действия.",
        },
        {
            "student_name": "Алина Назарова",
            "course_year": 6,
            "academic_score": 93,
            "anatomy_score": 81,
            "physiology_score": 97,
            "surgery_interest": 4,
            "cardiology_interest": 10,
            "neurology_interest": 7,
            "ent_interest": 2,
            "manual_dexterity": 6,
            "stress_tolerance": 8,
            "empathy": 8,
            "analytical_thinking": 9,
            "communication_skill": 9,
            "research_orientation": 8,
            "night_shift_readiness": 7,
            "cardiovascular_endurance": 8,
            "auditory_attention": 5,
            "precision_focus": 8,
            "student_note": "Устойчива к длительным диагностическим задачам.",
        },
        {
            "student_name": "Виктор Лебедев",
            "course_year": 5,
            "academic_score": 90,
            "anatomy_score": 78,
            "physiology_score": 90,
            "surgery_interest": 3,
            "cardiology_interest": 5,
            "neurology_interest": 10,
            "ent_interest": 2,
            "manual_dexterity": 5,
            "stress_tolerance": 7,
            "empathy": 8,
            "analytical_thinking": 10,
            "communication_skill": 6,
            "research_orientation": 9,
            "night_shift_readiness": 4,
            "cardiovascular_endurance": 5,
            "auditory_attention": 6,
            "precision_focus": 9,
            "student_note": "Тяготеет к сложным интеллектуальным кейсам.",
        },
        {
            "student_name": "Ольга Демина",
            "course_year": 3,
            "academic_score": 83,
            "anatomy_score": 84,
            "physiology_score": 78,
            "surgery_interest": 4,
            "cardiology_interest": 3,
            "neurology_interest": 4,
            "ent_interest": 9,
            "manual_dexterity": 7,
            "stress_tolerance": 6,
            "empathy": 9,
            "analytical_thinking": 6,
            "communication_skill": 9,
            "research_orientation": 4,
            "night_shift_readiness": 5,
            "cardiovascular_endurance": 4,
            "auditory_attention": 10,
            "precision_focus": 8,
            "student_note": "Рано определилась с ЛОР-направлением.",
        },
        {
            "student_name": "Георгий Савельев",
            "course_year": 6,
            "academic_score": 86,
            "anatomy_score": 94,
            "physiology_score": 82,
            "surgery_interest": 10,
            "cardiology_interest": 4,
            "neurology_interest": 4,
            "ent_interest": 2,
            "manual_dexterity": 9,
            "stress_tolerance": 9,
            "empathy": 5,
            "analytical_thinking": 6,
            "communication_skill": 5,
            "research_orientation": 3,
            "night_shift_readiness": 10,
            "cardiovascular_endurance": 9,
            "auditory_attention": 4,
            "precision_focus": 9,
            "student_note": "Готов к интенсивной хирургической нагрузке.",
        },
        {
            "student_name": "Татьяна Максимова",
            "course_year": 5,
            "academic_score": 91,
            "anatomy_score": 82,
            "physiology_score": 93,
            "surgery_interest": 4,
            "cardiology_interest": 8,
            "neurology_interest": 8,
            "ent_interest": 2,
            "manual_dexterity": 6,
            "stress_tolerance": 8,
            "empathy": 9,
            "analytical_thinking": 9,
            "communication_skill": 9,
            "research_orientation": 8,
            "night_shift_readiness": 6,
            "cardiovascular_endurance": 7,
            "auditory_attention": 5,
            "precision_focus": 8,
            "student_note": "Сильна и в аналитике, и в клиническом диалоге.",
        },
        {
            "student_name": "Кирилл Федоров",
            "course_year": 4,
            "academic_score": 85,
            "anatomy_score": 88,
            "physiology_score": 83,
            "surgery_interest": 5,
            "cardiology_interest": 4,
            "neurology_interest": 5,
            "ent_interest": 8,
            "manual_dexterity": 8,
            "stress_tolerance": 7,
            "empathy": 7,
            "analytical_thinking": 7,
            "communication_skill": 8,
            "research_orientation": 4,
            "night_shift_readiness": 6,
            "cardiovascular_endurance": 6,
            "auditory_attention": 9,
            "precision_focus": 8,
            "student_note": "Хорошо справляется с тонкими манипуляциями.",
        },
    ]

    result = []
    for index, profile in enumerate(profiles, start=1):
        prediction = predict_specialty(profile)
        result.append(
            {
                "id": f"MED-{index:03d}",
                "created_at": (start + timedelta(hours=index * 5)).isoformat(timespec="seconds"),
                **normalize_profile(profile),
                **prediction,
            }
        )

    return result
