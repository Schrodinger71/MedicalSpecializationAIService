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
            --paper: #f8f1e8;
            --paper-strong: #fff9f3;
            --ink: #2d241d;
            --muted: #726154;
            --accent: #b95638;
            --accent-dark: #8e4028;
            --forest: #3e5d48;
            --card-border: rgba(116, 88, 66, 0.16);
            --shadow: 0 18px 50px rgba(81, 51, 34, 0.10);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(185, 86, 56, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(62, 93, 72, 0.12), transparent 26%),
                linear-gradient(180deg, #f4ebdf 0%, #efe3d3 100%);
            color: var(--ink);
            font-family: "Trebuchet MS", "Georgia", serif;
        }
        .block-container {
            max-width: 1480px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .hero-panel,
        .clinic-card,
        div[data-testid="stMetric"] {
            animation: fadeUp 0.55s ease-out both;
        }
        .hero-panel {
            padding: 28px 30px;
            border-radius: 28px;
            border: 1px solid rgba(93, 63, 45, 0.12);
            background:
                linear-gradient(135deg, rgba(255, 249, 243, 0.96), rgba(249, 238, 224, 0.96)),
                repeating-linear-gradient(
                    135deg,
                    rgba(185, 86, 56, 0.03) 0,
                    rgba(185, 86, 56, 0.03) 12px,
                    rgba(255, 255, 255, 0.00) 12px,
                    rgba(255, 255, 255, 0.00) 26px
                );
            box-shadow: var(--shadow);
            margin-bottom: 20px;
        }
        .hero-eyebrow {
            color: var(--accent);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .hero-grid {
            display: grid;
            grid-template-columns: 1.4fr 0.95fr;
            gap: 22px;
            align-items: start;
        }
        .hero-panel h1 {
            margin: 0 0 12px 0;
            font-size: 2.15rem;
            line-height: 1.08;
            color: #382920;
        }
        .hero-panel p {
            margin: 0;
            color: #5c4d42;
            font-size: 1rem;
        }
        .hero-side {
            display: grid;
            gap: 12px;
        }
        .side-note {
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--card-border);
            color: #48382d;
        }
        .status-ok,
        .status-warn {
            padding: 12px 14px;
            border-radius: 16px;
            font-weight: 700;
        }
        .status-ok {
            background: rgba(62, 93, 72, 0.12);
            color: #244232;
            border: 1px solid rgba(62, 93, 72, 0.22);
        }
        .status-warn {
            background: rgba(185, 86, 56, 0.12);
            color: #7a341f;
            border: 1px solid rgba(185, 86, 56, 0.22);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f1e3d2 0%, #ead7c1 100%);
            border-right: 1px solid rgba(113, 85, 63, 0.12);
        }
        section[data-testid="stSidebar"] * {
            color: #3a2c23 !important;
        }
        .clinic-card {
            padding: 18px 18px 8px 18px;
            border-radius: 24px;
            background: rgba(255, 250, 244, 0.92);
            border: 1px solid var(--card-border);
            box-shadow: var(--shadow);
            margin-bottom: 16px;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 250, 244, 0.94);
            border-radius: 18px;
            padding: 14px;
            border: 1px solid var(--card-border);
            box-shadow: 0 10px 22px rgba(81, 51, 34, 0.06);
        }
        .result-badge {
            display: inline-block;
            margin-bottom: 12px;
            padding: 8px 12px;
            border-radius: 999px;
            color: #fff8f2;
            font-weight: 800;
            font-size: 0.88rem;
        }
        .result-summary {
            padding: 16px;
            border-radius: 20px;
            background: #fffaf5;
            border: 1px solid var(--card-border);
            margin-bottom: 16px;
        }
        .soft-note {
            padding: 12px 14px;
            border-radius: 14px;
            background: rgba(62, 93, 72, 0.07);
            border: 1px solid rgba(62, 93, 72, 0.14);
            color: #344e3d;
            font-size: 0.94rem;
        }
        h1, h2, h3, p, label, li, span {
            color: var(--ink);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            padding-bottom: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 250, 244, 0.88) !important;
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 10px 16px;
        }
        .stTabs [data-baseweb="tab"] p {
            color: #4a3a2f !important;
            font-weight: 800 !important;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(185, 86, 56, 0.14) !important;
            border: 1px solid rgba(185, 86, 56, 0.28) !important;
        }
        .stTabs [aria-selected="true"] p {
            color: #8c3f26 !important;
        }
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        div[data-baseweb="select"] > div {
            background: #fffdf9 !important;
            color: #342920 !important;
            border: 1px solid rgba(113, 85, 63, 0.28) !important;
            border-radius: 14px !important;
        }
        .stButton > button,
        .stForm button,
        div[data-testid="stFormSubmitButton"] button {
            background: var(--accent) !important;
            color: #fff9f4 !important;
            border: 1px solid var(--accent) !important;
            border-radius: 14px !important;
            min-height: 46px !important;
            font-weight: 800 !important;
            box-shadow: 0 14px 26px rgba(185, 86, 56, 0.16);
        }
        .stButton > button:hover,
        .stForm button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {
            background: var(--accent-dark) !important;
            border-color: var(--accent-dark) !important;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--card-border);
        }
        details[data-testid="stExpander"] {
            border: 1px solid var(--card-border);
            border-radius: 16px;
            background: rgba(255, 250, 244, 0.85);
        }
        details[data-testid="stExpander"] summary {
            color: #6a2f1d !important;
            font-weight: 800 !important;
        }
        .legend-line {
            margin-top: 10px;
            color: var(--muted);
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(api_ok: bool, api_info: dict, total_students: int) -> None:
    status_class = "status-ok" if api_ok else "status-warn"
    status_text = (
        "API активен: контейнерная версия отвечает и готова к прогнозу."
        if api_ok
        else "API недоступен: интерфейс временно использует локальный расчёт без контейнера."
    )
    st.markdown(
        f"""
        <div class="hero-panel">
          <div class="hero-grid">
            <div>
              <div class="hero-eyebrow">DevOps • Docker Personal • Applied Domain</div>
              <h1>Определение будущей врачебной специализации студента мединститута</h1>
              <p>
                Отдельный проект на основе исходного шаблона. В этой версии интерфейс собран
                как клинический портал: тёплая палитра, карточная навигация и акцент на объяснимый прогноз.
              </p>
              <div class="legend-line">
                Прикладная область: <strong>Лечение</strong> • Входных полей: <strong>{INPUT_FIELD_COUNT}</strong> •
                Демонстрационных записей: <strong>{total_students}</strong>
              </div>
            </div>
            <div class="hero-side">
              <div class="{status_class}">{status_text}</div>
              <div class="side-note">
                <strong>Поддерживаемые специализации:</strong><br>
                Хирург, кардиолог, невролог, отоларинголог.
              </div>
              <div class="side-note">
                <strong>Docker edition:</strong> Docker Personal<br>
                <strong>API model role:</strong> {api_info.get('model_role', 'weighted_profile_assessor')}
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction(result: dict | None) -> None:
    if not result:
        st.info("После расчёта здесь появятся специализация, вероятности и объяснение прогноза.")
        return

    specialty_key = result.get("predicted_specialty_key", "")
    color = SPECIALTY_COLORS.get(specialty_key, "#7a4b38")
    confidence = float(result.get("confidence", 0.0))

    st.markdown(
        f"""
        <div class="result-summary" style="border-left: 6px solid {color};">
          <span class="result-badge" style="background:{color};">{result.get('predicted_specialty', '-')}</span>
          <h3 style="margin: 0 0 8px 0;">Итоговый прогноз</h3>
          <p style="margin: 0 0 10px 0;">{result.get('summary', '-')}</p>
          <div class="soft-note">Уверенность прогноза: <strong>{confidence:.0%}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Уверенность", f"{confidence:.0%}")
    metric_col2.metric("Входных полей", result.get("input_count", INPUT_FIELD_COUNT))
    metric_col3.metric("Источник", result.get("model_role", "weighted_profile_assessor"))

    st.write("**Ключевые факторы**")
    for item in result.get("key_factors", []):
        st.markdown(f"- {item}")

    st.write("**Рекомендации**")
    for item in result.get("recommendations", []):
        st.markdown(f"- {item}")

    with st.expander("Вероятности по специализациям", expanded=True):
        st.dataframe(
            probability_frame(result.get("probabilities", {})),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Специализация": st.column_config.TextColumn(width="large"),
                "Вероятность": st.column_config.NumberColumn(format="%.2f"),
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
                "Вероятность": st.column_config.NumberColumn(format="%.2f"),
            },
        )


def build_payload() -> dict | None:
    with st.form("student_specialty_form", clear_on_submit=False):
        st.markdown("### Учебный профиль")
        c1, c2, c3 = st.columns(3)
        student_name = c1.text_input("ФИО студента", "Екатерина Новикова")
        course_year = c2.selectbox("Курс", [1, 2, 3, 4, 5, 6], index=4)
        academic_score = c3.number_input("Средний академический балл", min_value=0, max_value=100, value=88)

        c4, c5 = st.columns(2)
        anatomy_score = c4.number_input("Оценка по анатомии", min_value=0, max_value=100, value=90)
        physiology_score = c5.number_input("Оценка по физиологии", min_value=0, max_value=100, value=86)

        st.markdown("### Интерес к специализациям")
        i1, i2, i3, i4 = st.columns(4)
        surgery_interest = i1.slider("Хирургия", 0, 10, 8)
        cardiology_interest = i2.slider("Кардиология", 0, 10, 6)
        neurology_interest = i3.slider("Неврология", 0, 10, 5)
        ent_interest = i4.slider("Отоларингология", 0, 10, 4)

        st.markdown("### Клинические и личностные навыки")
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
            height=100,
        )

        submitted = st.form_submit_button("Рассчитать вероятную специализацию", use_container_width=True)
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
        page_icon="MD",
        layout="wide",
    )
    install_styles()

    students = load_students()
    frame = students_frame(students)
    api_ok, api_info = ping_api()

    if "last_prediction" not in st.session_state and students:
        st.session_state["last_prediction"] = students[-1]

    st.sidebar.title("Фильтры и статус")
    st.sidebar.caption("Отдельная копия проекта на той же архитектурной базе.")
    st.sidebar.write(f"API URL: `{API_URL}`")
    st.sidebar.write(f"Статус API: {'online' if api_ok else 'offline'}")
    st.sidebar.write(f"Входных полей: `{INPUT_FIELD_COUNT}`")

    specialty_options = sorted([value for value in frame["predicted_specialty"].dropna().unique()]) if not frame.empty else []
    year_options = sorted([int(value) for value in frame["course_year"].dropna().unique()]) if not frame.empty else []

    filter_specialties = st.sidebar.multiselect("Специализация", specialty_options)
    filter_years = st.sidebar.multiselect("Курс", year_options)

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

    tab_registry, tab_new, tab_analytics = st.tabs(["Реестр студентов", "Новый прогноз", "Аналитика"])

    with tab_registry:
        st.markdown('<div class="clinic-card">', unsafe_allow_html=True)
        st.subheader("Локальный реестр профилей")

        filtered = frame.copy()
        if filter_specialties:
            filtered = filtered[filtered["predicted_specialty"].isin(filter_specialties)]
        if filter_years:
            filtered = filtered[filtered["course_year"].isin(filter_years)]

        registry = registry_frame(filtered)
        if registry.empty:
            st.warning("По текущим фильтрам записи не найдены.")
        else:
            st.dataframe(
                registry,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Студент": st.column_config.TextColumn(width="large"),
                    "Прогноз": st.column_config.TextColumn(width="medium"),
                    "Уверенность": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_new:
        form_col, result_col = st.columns([1.3, 0.92], gap="large")

        with form_col:
            st.markdown('<div class="clinic-card">', unsafe_allow_html=True)
            st.subheader("Профиль студента")
            st.caption("Форма содержит 19 параметров и соответствует требованию задания по количеству входных данных.")

            payload = build_payload()
            if payload is not None:
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
                    st.success("Прогноз рассчитан через контейнеризированный FastAPI-сервис.")
                else:
                    st.warning("API недоступен, поэтому использован локальный расчёт внутри dashboard.")
            st.markdown("</div>", unsafe_allow_html=True)

        with result_col:
            st.markdown('<div class="clinic-card">', unsafe_allow_html=True)
            st.subheader("Последний результат")
            render_prediction(st.session_state.get("last_prediction"))
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_analytics:
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown('<div class="clinic-card">', unsafe_allow_html=True)
            st.subheader("Распределение по специализациям")
            if not frame.empty:
                specialty_counts = (
                    frame["predicted_specialty"]
                    .value_counts()
                    .rename_axis("Специализация")
                    .reset_index(name="Количество")
                    .set_index("Специализация")
                )
                st.bar_chart(specialty_counts, use_container_width=True)
            else:
                st.info("Недостаточно данных для графика.")
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="clinic-card">', unsafe_allow_html=True)
            st.subheader("Средняя уверенность по курсам")
            if not frame.empty:
                confidence_by_year = (
                    frame.groupby("course_year", dropna=True)["confidence"]
                    .mean()
                    .round(4)
                    .rename("Уверенность")
                    .to_frame()
                )
                st.line_chart(confidence_by_year, use_container_width=True)
            else:
                st.info("Недостаточно данных для графика.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="clinic-card">', unsafe_allow_html=True)
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
                        "Вероятность": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
            else:
                st.info("Ранжирование появится после первого расчёта.")
        else:
            st.info("Сначала выполните хотя бы один расчёт в соседней вкладке.")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
