# Medical Student Specialization Service

Проект создан на базе исходного шаблона `KIIIncidentAIService`, но адаптирован под новую прикладную область:
по данным студента медицинского института определить его вероятную будущую врачебную специализацию:

- хирург
- кардиолог
- невролог
- отоларинголог

Краткая прикладная область: `Лечение`

## DevOps-основание

- Docker edition: `Docker Personal`
- Сайт: `https://docker.com/products/personal/`
- Оркестрация локального запуска: `docker compose`

## Что внутри

Проект повторяет архитектуру исходного шаблона:

- `inference-api` на `FastAPI` принимает профиль студента и возвращает прогноз специализации.
- `dashboard` на `Streamlit` даёт веб-интерфейс для ввода данных, просмотра результатов и локального реестра.
- `model-training` содержит вспомогательные скрипты для генерации синтетических профилей и обновления базовых приоритетов модели.

## Входные данные

В форме используется `19` входных полей, то есть требование `не менее 15 входных данных` выполнено.

Основные группы признаков:

- учебный прогресс
- профильные оценки
- интерес к направлениям
- клинические и коммуникативные навыки
- устойчивость к нагрузке
- исследовательская склонность

## Структура проекта

```text
MedicalSpecializationAIService/
  README.md
  docker-compose.yml
  medical_methodology.py
  dashboard/
    app.py
    Dockerfile
    requirements.txt
    data/
      students.json
      incidents.json
  inference-api/
    app.py
    Dockerfile
    requirements.txt
    models/
      specialty_baseline.json
  model-training/
    generate_synthetic_data.py
    retrain_model.py
    validate_inference.py
    finetune_kii.py
    requirements.txt
  Tools/
    DataConverter/
```

## Логика определения специализации

Сервис использует взвешенную экспертную модель:

1. Нормализует профиль студента.
2. Рассчитывает баллы по четырём специальностям.
3. Преобразует баллы в вероятности.
4. Возвращает наиболее вероятную специализацию, объяснение, ключевые факторы и рекомендации.

Это удобно для учебной DevOps-демонстрации:

- логика прозрачная
- API легко контейнеризируется
- интерфейс разворачивается отдельно
- проект можно запускать и сопровождать через Docker Compose

## Быстрый старт

Перейдите в каталог нового проекта:

```powershell
cd .\MedicalSpecializationAIService
```

Запуск:

```powershell
docker compose up --build
```

Фоновый режим:

```powershell
docker compose up --build -d
```

Остановка:

```powershell
docker compose down
```

## Адреса после старта

- Dashboard: `http://localhost:8511`
- API: `http://localhost:8018`
- Swagger UI: `http://localhost:8018/docs`
- Health endpoint: `http://localhost:8018/health`

## Сервисы в Docker Compose

### `inference-api`

- запускается на внутреннем порту `8000`
- публикуется наружу как `8018`
- использует файл `inference-api/models/specialty_baseline.json`

### `dashboard`

- запускается на внутреннем порту `8501`
- публикуется наружу как `8511`
- хранит локальный реестр в `dashboard/data/students.json`

## Полезные сценарии

### Генерация синтетического датасета

```powershell
cd .\model-training
python .\generate_synthetic_data.py
cd ..
```

Результат будет сохранён в:

- `model-training/synthetic_med_students.csv`

### Обновление базовых вероятностей по датасету

```powershell
cd .\model-training
python .\retrain_model.py
cd ..
```

Результат будет сохранён в:

- `inference-api/models/specialty_baseline.json`

## Особенности интерфейса

В этой версии специально изменён стиль сайта относительно исходного проекта:

- тёплая палитра вместо холодной синей
- карточный интерфейс в стиле клинического дашборда
- другой визуальный акцент на реестр студентов и прогноз специализации

## Примечание

Проект сделан как отдельная папка-копия исходного решения и может разворачиваться независимо от базового сервиса.
