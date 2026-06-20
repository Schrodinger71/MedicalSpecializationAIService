from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from medical_methodology import (  # noqa: E402
    INPUT_FIELD_COUNT,
    SPECIALTY_COLORS,
    SPECIALTIES,
    demo_profiles,
    predict_specialty,
)

API_URL = os.getenv("API_URL", "http://localhost:8018")
DATA_FILE = Path(os.getenv("DATA_FILE", "data/students.json"))

SPECIALTY_ICONS = {
    "surgery": "🔪",
    "cardiology": "❤️",
    "neurology": "🧠",
    "otolaryngology": "👂",
}


def ensure_data_file() -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_FILE.exists():
        DATA_FILE.write_text(
            json.dumps(demo_profiles(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    try:
        current = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        current = []

    if not current:
        DATA_FILE.write_text(
            json.dumps(demo_profiles(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_students() -> list[dict]:
    ensure_data_file()
    try:
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return demo_profiles()


def save_students(items: list[dict]) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def ping_api() -> tuple[bool, dict]:
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException:
        return False, {"status": "offline", "model_role": "local_fallback"}


def remote_predict(payload: dict) -> dict | None:
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def predict_student(payload: dict) -> tuple[dict, str]:
    response = remote_predict(payload)
    if response is not None:
        return response, "api"

    local = predict_specialty(payload)
    local["normalized_profile"] = payload
    local["model_loaded"] = False
    local["model_usable"] = False
    local["model_source"] = "dashboard_local_fallback"
    local["model_role"] = "local_fallback"
    local["available_specialties"] = list(SPECIALTIES.values())
    return local, "local"


def students_frame(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(
            columns=[
                "id",
                "student_name",
                "course_year",
                "predicted_specialty",
                "predicted_specialty_key",
                "confidence",
                "created_at",
            ]
        )

    frame = pd.DataFrame(items).copy()
    for column, default in {
        "id": "-",
        "student_name": "Студент",
        "course_year": 0,
        "predicted_specialty": "-",
        "predicted_specialty_key": "",
        "confidence": 0.0,
        "created_at": None,
    }.items():
        if column not in frame.columns:
            frame[column] = default

    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    return frame.sort_values("created_at", ascending=False, na_position="last")


def registry_frame(frame: pd.DataFrame) -> pd.DataFrame:
    registry = frame.copy()
    registry["created_at"] = registry["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    registry = registry.rename(
        columns={
            "id": "ID",
            "student_name": "Студент",
            "course_year": "Курс",
            "predicted_specialty": "Прогноз",
            "confidence": "Уверенность",
            "academic_score": "Средний балл",
            "anatomy_score": "Анатомия",
            "physiology_score": "Физиология",
            "created_at": "Создано",
        }
    )
    columns = [
        "ID",
        "Студент",
        "Курс",
        "Средний балл",
        "Анатомия",
        "Физиология",
        "Прогноз",
        "Уверенность",
        "Создано",
    ]
    available = [column for column in columns if column in registry.columns]
    return registry[available]


def probability_frame(probabilities: dict[str, float]) -> pd.DataFrame:
    if not probabilities:
        return pd.DataFrame(columns=["Специализация", "Вероятность"])

    frame = pd.DataFrame(
        [{"Специализация": key, "Вероятность": float(value)} for key, value in probabilities.items()]
    )
    return frame.sort_values("Вероятность", ascending=False)


def score_breakdown_frame(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["Специализация", "Интегральный балл", "Вероятность"])

    frame = pd.DataFrame(items).copy()
    return frame.rename(
        columns={
            "specialty_name": "Специализация",
            "score": "Интегральный балл",
            "probability": "Вероятность",
        }
    )[["Специализация", "Интегральный балл", "Вероятность"]]


def install_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-deep: #0b2230;
            --bg-soft: #f4f9f8;
            --surface: #ffffff;
            --ink: #102a2f;
            --muted: #5d7a7a;
            --primary: #0e9a8a;
            --primary-dark: #097564;
            --primary-soft: rgba(14, 154, 138, 0.12);
            --accent: #2f6fed;
            --accent-soft: rgba(47, 111, 237, 0.12);
            --warn: #e0793a;
            --border: rgba(16, 42, 47, 0.10);
            --shadow-sm: 0 6px 18px rgba(11, 34, 48, 0.06);
            --shadow-md: 0 18px 44px rgba(11, 34, 48, 0.10);
        }
        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", "Trebuchet MS", sans-serif;
        }
        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(14, 154, 138, 0.10), transparent 38%),
                radial-gradient(circle at 92% 6%, rgba(47, 111, 237, 0.10), transparent 32%),
                linear-gradient(180deg, var(--bg-soft) 0%, #eef5f4 100%);
            color: var(--ink);
        }
        .block-container {
            max-width: 1500px;
            padding-top: 1.3rem;
            padding-bottom: 2.4rem;
        }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes shimmer {
            0% { background-position: -120% 0; }
            100% { background-position: 120% 0; }
        }
        .hero-panel, .panel-card, div[data-testid="stMetric"] {
            animation: fadeUp 0.5s ease-out both;
        }
        .hero-panel {
            position: relative;
            overflow: hidden;
            padding: 32px 34px;
            border-radius: 28px;
            background: linear-gradient(125deg, #0b2230 0%, #0e3a45 45%, #0e6f63 100%);
            color: #f3fbfa;
            box-shadow: var(--shadow-md);
            margin-bottom: 22px;
        }
        .hero-panel::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                radial-gradient(circle at 85% -10%, rgba(255, 255, 255, 0.14), transparent 40%),
                radial-gradient(circle at -5% 110%, rgba(255, 255, 255, 0.08), transparent 40%);
            pointer-events: none;
        }
        .hero-grid {
            position: relative;
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 26px;
            align-items: center;
        }
        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            color: #8fe9d8;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-weight: 800;
            margin-bottom: 14px;
        }
        .hero-panel h1 {
            margin: 0 0 14px 0;
            font-size: 2.25rem;
            line-height: 1.12;
            font-weight: 800;
            color: #ffffff;
        }
        .hero-panel p {
            margin: 0;
            color: #d7ece8;
            font-size: 1.02rem;
            line-height: 1.55;
            max-width: 640px;
        }
        .hero-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 18px;
        }
        .hero-pill {
            padding: 7px 14px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.16);
            font-size: 0.82rem;
            font-weight: 700;
            color: #eafff9;
        }
        .hero-side {
            position: relative;
            display: grid;
            gap: 12px;
        }
        .status-chip {
            padding: 13px 16px;
            border-radius: 16px;
            font-weight: 800;
            font-size: 0.92rem;
            display: flex;
            align-items: center;
            gap: 10px;
            backdrop-filter: blur(6px);
        }
        .status-chip.ok {
            background: rgba(45, 211, 158, 0.16);
            border: 1px solid rgba(45, 211, 158, 0.40);
            color: #c8ffe9;
        }
        .status-chip.fail {
            background: rgba(255, 145, 110, 0.16);
            border: 1px solid rgba(255, 145, 110, 0.40);
            color: #ffe0d2;
        }
        .dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
            display: inline-block;
        }
        .dot.ok { background: #2dd39e; box-shadow: 0 0 0 4px rgba(45, 211, 158, 0.25); }
        .dot.fail { background: #ff916e; box-shadow: 0 0 0 4px rgba(255, 145, 110, 0.25); }
        .hero-note {
            padding: 13px 16px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #eafff9;
            font-size: 0.88rem;
            line-height: 1.4;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b2230 0%, #103a40 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        section[data-testid="stSidebar"] * {
            color: #e7f6f3 !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] input {
            background: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(255, 255, 255, 0.16) !important;
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="tag"] {
            background: rgba(14, 154, 138, 0.45) !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.10);
        }
        div[data-testid="stMetric"] {
            background: var(--surface);
            border-radius: 18px;
            padding: 16px 18px;
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        }
        div[data-testid="stMetric"] label {
            color: var(--muted) !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink) !important;
            font-weight: 800 !important;
        }
        .panel-card {
            padding: 20px 20px 12px 20px;
            border-radius: 22px;
            background: var(--surface);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
            margin-bottom: 18px;
        }
        .panel-card h3, .panel-card .stSubheader {
            margin-top: 0;
        }
        .section-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 800;
            font-size: 1.05rem;
            color: var(--ink);
            margin-bottom: 4px;
        }
        .section-sub {
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 14px;
        }
        .result-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 14px;
            padding: 9px 16px;
            border-radius: 999px;
            color: #ffffff;
            font-weight: 800;
            font-size: 0.92rem;
            box-shadow: 0 10px 22px rgba(11, 34, 48, 0.16);
        }
        .result-summary {
            padding: 18px;
            border-radius: 18px;
            background: linear-gradient(180deg, #f7fbfa 0%, #f0f7f6 100%);
            border: 1px solid var(--border);
            margin-bottom: 16px;
        }
        .confidence-bar-track {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            background: rgba(16, 42, 47, 0.08);
            overflow: hidden;
            margin-top: 10px;
        }
        .confidence-bar-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--primary), var(--accent));
        }
        .soft-note {
            padding: 13px 15px;
            border-radius: 14px;
            background: var(--primary-soft);
            border: 1px solid rgba(14, 154, 138, 0.22);
            color: #0a4a41;
            font-size: 0.92rem;
        }
        .ai-note {
            margin: 4px 0 16px 0;
            padding: 13px 15px;
            background: var(--accent-soft);
            border: 1px solid rgba(47, 111, 237, 0.22);
            border-radius: 14px;
            color: #14305f;
            font-size: 0.92rem;
        }
        .empty-state {
            padding: 26px;
            border-radius: 16px;
            text-align: center;
            color: var(--muted);
            background: rgba(16, 42, 47, 0.03);
            border: 1px dashed var(--border);
        }
        h1, h2, h3, p, label, li, span { color: var(--ink); }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding-bottom: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            background: var(--surface) !important;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 10px 18px;
        }
        .stTabs [data-baseweb="tab"] p {
            color: #436461 !important;
            font-weight: 700 !important;
        }
        .stTabs [aria-selected="true"] {
            background: var(--primary-soft) !important;
            border: 1px solid rgba(14, 154, 138, 0.35) !important;
        }
        .stTabs [aria-selected="true"] p {
            color: var(--primary-dark) !important;
        }
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        div[data-baseweb="select"] > div {
            background: #fdfffe !important;
            color: var(--ink) !important;
            border: 1px solid rgba(16, 42, 47, 0.18) !important;
            border-radius: 12px !important;
        }
        .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
            border: 1px solid var(--primary) !important;
            box-shadow: 0 0 0 3px var(--primary-soft) !important;
        }
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: var(--primary) !important;
            border-color: var(--primary) !important;
        }
        div[data-testid="stSliderTrackColor"], .stSlider [data-testid="stTickBar"] + div > div {
            background: var(--primary) !important;
        }
        .stButton > button,
        .stForm button,
        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(120deg, var(--primary), var(--primary-dark)) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 14px !important;
            min-height: 48px !important;
            font-weight: 800 !important;
            box-shadow: 0 14px 28px rgba(14, 154, 138, 0.24);
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        .stButton > button:hover,
        .stForm button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 32px rgba(14, 154, 138, 0.30);
        }
        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        details[data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 16px;
            background: var(--surface);
        }
        details[data-testid="stExpander"] summary {
            color: var(--primary-dark) !important;
            font-weight: 800 !important;
        }
        .legend-line {
            margin-top: 14px;
            color: #cfe9e3;
            font-size: 0.86rem;
        }
        .legend-line strong { color: #ffffff; }
        .factor-chip {
            display: inline-block;
            margin: 3px 6px 3px 0;
            padding: 6px 12px;
            border-radius: 999px;
            background: var(--primary-soft);
            border: 1px solid rgba(14, 154, 138, 0.22);
            color: #0a4a41;
            font-size: 0.84rem;
            font-weight: 600;
        }
        .rec-item {
            padding: 10px 12px;
            margin-bottom: 6px;
            border-radius: 12px;
            background: rgba(16, 42, 47, 0.03);
            border-left: 3px solid var(--primary);
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(api_ok: bool, api_info: dict, total_students: int) -> None:
    status_class = "ok" if api_ok else "fail"
    status_text = (
        "AI API онлайн — нейросеть подключена и готова к прогнозу"
        if api_ok
        else "AI API недоступен — используется локальный расчёт без контейнера"
    )
    model_role = api_info.get("model_role", "weighted_profile_assessor")
    model_label = "Нейросеть + методика" if model_role == "supporting_classifier" else "Только методика"

    st.markdown(
        f"""
        <div class="hero-panel">
          <div class="hero-grid">
            <div>
              <div class="hero-eyebrow">⚕ DevOps • Docker Personal • Прикладная область: Лечение</div>
              <h1>Прогноз будущей врачебной специализации студента мединститута</h1>
              <p>
                Сервис сочетает прозрачную экспертную методику оценки профиля студента
                с обученной нейросетью-классификатором, чтобы предложить наиболее вероятную
                специализацию вместе с объяснением и рекомендациями для развития.
              </p>
              <div class="hero-pills">
                <span class="hero-pill">🔪 Хирург</span>
                <span class="hero-pill">❤️ Кардиолог</span>
                <span class="hero-pill">🧠 Невролог</span>
                <span class="hero-pill">👂 Отоларинголог</span>
              </div>
              <div class="legend-line">
                Входных полей: <strong>{INPUT_FIELD_COUNT}</strong> &nbsp;•&nbsp;
                Профилей в реестре: <strong>{total_students}</strong> &nbsp;•&nbsp;
                Режим модели: <strong>{model_label}</strong>
              </div>
            </div>
            <div class="hero-side">
              <div class="status-chip {status_class}">
                <span class="dot {status_class}"></span>{status_text}
              </div>
              <div class="hero-note">
                <strong>API model role:</strong> {model_role}<br>
                <strong>Docker edition:</strong> Docker Personal
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction(result: dict | None) -> None:
    if not result:
        st.markdown(
            '<div class="empty-state">После расчёта здесь появятся специализация, '
            "вероятности и объяснение прогноза.</div>",
            unsafe_allow_html=True,
        )
        return

    specialty_key = result.get("predicted_specialty_key", "")
    color = SPECIALTY_COLORS.get(specialty_key, "#0e9a8a")
    icon = SPECIALTY_ICONS.get(specialty_key, "🩺")
    confidence = float(result.get("confidence", 0.0))

    st.markdown(
        f"""
        <div class="result-summary" style="border-left: 6px solid {color};">
          <span class="result-badge" style="background:{color};">{icon} {result.get('predicted_specialty', '-')}</span>
          <h3 style="margin: 0 0 8px 0;">Итоговый прогноз</h3>
          <p style="margin: 0;">{result.get('summary', '-')}</p>
          <div class="confidence-bar-track">
            <div class="confidence-bar-fill" style="width:{confidence*100:.0f}%;"></div>
          </div>
          <div class="soft-note" style="margin-top:10px;">Уверенность прогноза: <strong>{confidence:.0%}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Уверенность", f"{confidence:.0%}")
    metric_col2.metric("Входных полей", result.get("input_count", INPUT_FIELD_COUNT))
    role = result.get("model_role", "weighted_profile_assessor")
    metric_col3.metric("Роль модели", "NN + методика" if role == "supporting_classifier" else "Методика")

    st.markdown("**Ключевые факторы**")
    st.markdown(
        "".join(f'<span class="factor-chip">{item}</span>' for item in result.get("key_factors", [])),
        unsafe_allow_html=True,
    )

    st.markdown("**Рекомендации по развитию**")
    for item in result.get("recommendations", []):
        st.markdown(f'<div class="rec-item">💡 {item}</div>', unsafe_allow_html=True)

    with st.expander("Вероятности по специализациям", expanded=True):
        st.dataframe(
            probability_frame(result.get("probabilities", {})),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Специализация": st.column_config.TextColumn(width="large"),
                "Вероятность": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            },
        )

    with st.expander("Интегральные баллы и ранжирование", expanded=False):
        st.dataframe(
            score_breakdown_frame(result.get("score_breakdown", [])),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Специализация": st.column_config.TextColumn(width="large"),
                "Интегральный балл": st.column_config.NumberColumn(format="%.2f"),
                "Вероятность": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            },
        )

    heuristic_probs = result.get("heuristic_probabilities")
    model_probs = result.get("model_probabilities")
    if heuristic_probs and model_probs:
        with st.expander("Вклад ИИ: методика vs нейросеть", expanded=False):
            st.markdown(
                """
                <div class="ai-note">
                    Итоговый прогноз объединяет два сигнала: прозрачную взвешенную методику
                    (35%) и обученную нейросеть-классификатор (65%). Так результат остаётся
                    объяснимым, но опирается на модель, обученную на датасете профилей.
                </div>
                """,
                unsafe_allow_html=True,
            )
            ai_col1, ai_col2 = st.columns(2, gap="medium")
            with ai_col1:
                st.caption("Методика (эвристика)")
                st.dataframe(
                    probability_frame(heuristic_probs),
                    use_container_width=True,
                    hide_index=True,
                    height=180,
                    column_config={
                        "Специализация": st.column_config.TextColumn(width="large"),
                        "Вероятность": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                    },
                )
            with ai_col2:
                st.caption("Нейросеть")
                st.dataframe(
                    probability_frame(model_probs),
                    use_container_width=True,
                    hide_index=True,
                    height=180,
                    column_config={
                        "Специализация": st.column_config.TextColumn(width="large"),
                        "Вероятность": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                    },
                )
            st.caption(f"Источник модели: `{result.get('model_source', '-')}` • usable: {result.get('model_usable', False)}")


def build_payload() -> dict | None:
    with st.form("student_specialty_form", clear_on_submit=False):
        st.markdown('<div class="section-title">🎓 Учебный профиль</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        student_name = c1.text_input("ФИО студента", "Екатерина Новикова")
        course_year = c2.selectbox("Курс", [1, 2, 3, 4, 5, 6], index=4)
        academic_score = c3.number_input("Средний академический балл", min_value=0, max_value=100, value=88)

        c4, c5 = st.columns(2)
        anatomy_score = c4.number_input("Оценка по анатомии", min_value=0, max_value=100, value=90)
        physiology_score = c5.number_input("Оценка по физиологии", min_value=0, max_value=100, value=86)

        st.markdown('<div class="section-title">🧭 Интерес к специализациям</div>', unsafe_allow_html=True)
        i1, i2, i3, i4 = st.columns(4)
        surgery_interest = i1.slider("🔪 Хирургия", 0, 10, 8)
        cardiology_interest = i2.slider("❤️ Кардиология", 0, 10, 6)
        neurology_interest = i3.slider("🧠 Неврология", 0, 10, 5)
        ent_interest = i4.slider("👂 Отоларингология", 0, 10, 4)

        st.markdown('<div class="section-title">🩺 Клинические и личностные навыки</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        manual_dexterity = s1.slider("Мануальная точность", 0, 10, 8)
        stress_tolerance = s2.slider("Стрессоустойчивость", 0, 10, 8)
        empathy = s3.slider("Эмпатия", 0, 10, 7)

        s4, s5, s6 = st.columns(3)
        analytical_thinking = s4.slider("Аналитическое мышление", 0, 10, 8)
        communication_skill = s5.slider("Коммуникация с пациентом", 0, 10, 7)
        research_orientation = s6.slider("Исследовательская направленность", 0, 10, 6)

        s7, s8, s9, s10 = st.columns(4)
        night_shift_readiness = s7.slider("Готовность к дежурствам", 0, 10, 7)
        cardiovascular_endurance = s8.slider("Выносливость", 0, 10, 7)
        auditory_attention = s9.slider("Слуховая внимательность", 0, 10, 5)
        precision_focus = s10.slider("Концентрация на деталях", 0, 10, 8)

        student_note = st.text_area(
            "Краткая заметка к профилю",
            "Студент активно проявляет себя на клинической практике.",
            height=90,
        )

        submitted = st.form_submit_button("✨ Рассчитать вероятную специализацию", use_container_width=True)
        if not submitted:
            return None

        return {
            "student_name": student_name,
            "course_year": course_year,
            "academic_score": academic_score,
            "anatomy_score": anatomy_score,
            "physiology_score": physiology_score,
            "surgery_interest": surgery_interest,
            "cardiology_interest": cardiology_interest,
            "neurology_interest": neurology_interest,
            "ent_interest": ent_interest,
            "manual_dexterity": manual_dexterity,
            "stress_tolerance": stress_tolerance,
            "empathy": empathy,
            "analytical_thinking": analytical_thinking,
            "communication_skill": communication_skill,
            "research_orientation": research_orientation,
            "night_shift_readiness": night_shift_readiness,
            "cardiovascular_endurance": cardiovascular_endurance,
            "auditory_attention": auditory_attention,
            "precision_focus": precision_focus,
            "student_note": student_note,
        }


def main() -> None:
    st.set_page_config(
        page_title="Medical Specialization Dashboard",
        page_icon="⚕️",
        layout="wide",
    )
    install_styles()

    students = load_students()
    frame = students_frame(students)
    api_ok, api_info = ping_api()

    if "last_prediction" not in st.session_state and students:
        st.session_state["last_prediction"] = students[-1]

    with st.sidebar:
        st.markdown("### ⚕️ Статус сервиса")
        st.caption("Прогноз специализации на базе методики + нейросети.")
        st.write(f"API URL: `{API_URL}`")
        st.write(f"Статус API: {'🟢 online' if api_ok else '🔴 offline'}")
        st.write(f"Модель загружена: {'да' if api_info.get('model_loaded') else 'нет'}")
        st.write(f"Источник модели: `{api_info.get('model_source', '-')}`")
        st.write(f"Входных полей: `{INPUT_FIELD_COUNT}`")

        st.divider()
        st.markdown("### 🔍 Фильтры реестра")
        specialty_options = sorted([value for value in frame["predicted_specialty"].dropna().unique()]) if not frame.empty else []
        year_options = sorted([int(value) for value in frame["course_year"].dropna().unique()]) if not frame.empty else []

        filter_specialties = st.multiselect("Специализация", specialty_options)
        filter_years = st.multiselect("Курс", year_options)

    render_hero(api_ok, api_info, len(students))

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Всего профилей", len(students))
    metric_col2.metric(
        "Прогнозов хирургии",
        int((frame["predicted_specialty"] == "Хирург").sum()) if not frame.empty else 0,
    )
    metric_col3.metric(
        "Средняя уверенность",
        f"{frame['confidence'].mean():.0%}" if not frame.empty else "0%",
    )
    metric_col4.metric(
        "Вариантов специализации",
        len(set(frame["predicted_specialty"].dropna())) if not frame.empty else 0,
    )

    tab_new, tab_registry, tab_analytics = st.tabs(["🧬 Новый прогноз", "🗂️ Реестр студентов", "📊 Аналитика"])

    with tab_new:
        form_col, result_col = st.columns([1.3, 0.92], gap="large")

        with form_col:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.subheader("Профиль студента")
            st.caption("Форма содержит 19 параметров и соответствует требованию задания по количеству входных данных.")

            payload = build_payload()
            if payload is not None:
                with st.spinner("Рассчитываем прогноз..."):
                    result, source = predict_student(payload)
                normalized_profile = result.get("normalized_profile", payload)
                record = {
                    "id": f"MED-{uuid4().hex[:6].upper()}",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    **normalized_profile,
                    **{key: value for key, value in result.items() if key != "normalized_profile"},
                    "prediction_source": source,
                }
                students.append(record)
                save_students(students)
                st.session_state["last_prediction"] = record
                if source == "api":
                    st.toast("Прогноз рассчитан через контейнеризированный FastAPI-сервис.", icon="✅")
                    st.success("Прогноз рассчитан через контейнеризированный FastAPI-сервис.")
                else:
                    st.toast("API недоступен — использован локальный расчёт.", icon="⚠️")
                    st.warning("API недоступен, поэтому использован локальный расчёт внутри dashboard.")
            st.markdown("</div>", unsafe_allow_html=True)

        with result_col:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.subheader("Последний результат")
            render_prediction(st.session_state.get("last_prediction"))
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_registry:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Локальный реестр профилей")

        filtered = frame.copy()
        if filter_specialties:
            filtered = filtered[filtered["predicted_specialty"].isin(filter_specialties)]
        if filter_years:
            filtered = filtered[filtered["course_year"].isin(filter_years)]

        registry = registry_frame(filtered)
        if registry.empty:
            st.markdown('<div class="empty-state">По текущим фильтрам записи не найдены.</div>', unsafe_allow_html=True)
        else:
            st.dataframe(
                registry,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Студент": st.column_config.TextColumn(width="large"),
                    "Прогноз": st.column_config.TextColumn(width="medium"),
                    "Уверенность": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                },
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_analytics:
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.subheader("Распределение по специализациям")
            if not frame.empty:
                specialty_counts = (
                    frame["predicted_specialty"]
                    .value_counts()
                    .rename_axis("Специализация")
                    .reset_index(name="Количество")
                    .set_index("Специализация")
                )
                st.bar_chart(specialty_counts, use_container_width=True, color="#0e9a8a")
            else:
                st.markdown('<div class="empty-state">Недостаточно данных для графика.</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.subheader("Средняя уверенность по курсам")
            if not frame.empty:
                confidence_by_year = (
                    frame.groupby("course_year", dropna=True)["confidence"]
                    .mean()
                    .round(4)
                    .rename("Уверенность")
                    .to_frame()
                )
                st.line_chart(confidence_by_year, use_container_width=True, color="#2f6fed")
            else:
                st.markdown('<div class="empty-state">Недостаточно данных для графика.</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Сводка по последнему прогнозу")
        last_prediction = st.session_state.get("last_prediction")
        if last_prediction:
            top_table = pd.DataFrame(last_prediction.get("top_specialties", []))
            if not top_table.empty:
                top_table = top_table.rename(
                    columns={
                        "specialty_name": "Специализация",
                        "probability": "Вероятность",
                    }
                )[["Специализация", "Вероятность"]]
                st.dataframe(
                    top_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Вероятность": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                    },
                )
            else:
                st.markdown('<div class="empty-state">Ранжирование появится после первого расчёта.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state">Сначала выполните хотя бы один расчёт в соседней вкладке.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
