from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from medical_methodology import (  # noqa: E402
    FEATURE_INPUT_DIM,
    SPECIALTY_INDEX,
    SPECIALTY_KEYS,
    SPECIALTY_NAMES,
    feature_vector,
)


def detect_default_data_path() -> Path:
    candidates = [
        Path("synthetic_med_students.csv"),
        Path("../model-training/synthetic_med_students.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DATA_PATH = Path(os.getenv("DATA_PATH", str(detect_default_data_path())))
BASE_MODEL_PATH = Path(os.getenv("BASE_MODEL", "../inference-api/models/model.keras"))
OUTPUT_MODEL_KERAS = Path(os.getenv("OUTPUT_MODEL_KERAS", "../inference-api/models/model.keras"))
OUTPUT_MODEL_H5 = Path(os.getenv("OUTPUT_MODEL_H5", "../inference-api/models/model.h5"))
EPOCHS = int(os.getenv("EPOCHS", "60"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", "0.2"))


def read_dataset(filepath: Path) -> pd.DataFrame:
    print(f"Загрузка датасета: {filepath}")
    if not filepath.exists():
        raise FileNotFoundError(
            f"Файл не найден: {filepath}. Сначала запустите generate_synthetic_data.py."
        )
    return pd.read_csv(filepath)


def build_training_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    records = df.to_dict("records")
    X = np.stack([feature_vector(record) for record in records]).astype(np.float32)
    y = np.array([SPECIALTY_INDEX[record["predicted_specialty_key"]] for record in records], dtype=np.int32)

    print(f"  Профилей: {len(df)}, признаков на входе модели: {X.shape[1]}")
    unique, counts = np.unique(y, return_counts=True)
    distribution = ", ".join(
        f"{SPECIALTY_NAMES[label]}={count}" for label, count in zip(unique, counts)
    )
    print(f"  Распределение классов: {distribution}")
    return X, y


def load_base_model(path: Path, expected_input_dim: int) -> tf.keras.Model | None:
    if not path.exists():
        print("Базовая модель не найдена, будет создана новая.")
        return None

    print(f"Попытка загрузки базовой модели: {path}")
    try:
        model = tf.keras.models.load_model(path, compile=False)
    except Exception as exc:
        print(f"  Не удалось загрузить базовую модель: {exc}")
        return None

    input_dim = int(model.input_shape[-1])
    if input_dim != expected_input_dim:
        print(f"  Размер входа базовой модели = {input_dim}, а нужен {expected_input_dim}. Создаю новую модель.")
        return None

    print("  Базовая модель совместима по размерности входа.")
    return model


def create_model(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(len(SPECIALTY_KEYS), activation="softmax"),
        ]
    )
    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fine_tuning: bool,
) -> tf.keras.Model:
    learning_rate = 1e-4 if fine_tuning else 1e-3
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Checkpointing straight to a .keras file mid-training hits a save_format
    # incompatibility in some tensorflow/keras combinations, so we rely on
    # EarlyStopping's restore_best_weights and save explicitly once at the end.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nЗапуск обучения...")
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )
    return model


def evaluate_model(model: tf.keras.Model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    predictions = model.predict(X_val, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    labels = list(range(len(SPECIALTY_KEYS)))
    print("\nМатрица ошибок:")
    print(confusion_matrix(y_val, predicted_classes, labels=labels))

    print("\nОтчет по классам:")
    print(
        classification_report(
            y_val,
            predicted_classes,
            labels=labels,
            target_names=SPECIALTY_NAMES,
            zero_division=0,
        )
    )

    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Validation loss: {loss:.4f}")
    print(f"  Validation accuracy: {accuracy:.2%}")
    return float(accuracy)


def save_artifacts(model: tf.keras.Model) -> None:
    print("\nСохранение артефактов...")
    OUTPUT_MODEL_KERAS.parent.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_MODEL_KERAS)
    print(f"  Keras-модель: {OUTPUT_MODEL_KERAS}")

    try:
        model.save(OUTPUT_MODEL_H5)
        print(f"  H5-модель: {OUTPUT_MODEL_H5}")
    except Exception as exc:
        print(f"  H5 сохранить не удалось: {exc}")


def main() -> None:
    print("=" * 72)
    print("ОБУЧЕНИЕ МОДЕЛИ ПРОГНОЗА ВРАЧЕБНОЙ СПЕЦИАЛИЗАЦИИ СТУДЕНТА")
    print("=" * 72)
    print(f"Ожидаемая размерность входа модели: {FEATURE_INPUT_DIM}")
    print(f"Поддерживаемые специализации: {', '.join(SPECIALTY_NAMES)}")

    dataset = read_dataset(DATA_PATH)
    X, y = build_training_arrays(dataset)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SPLIT,
        random_state=42,
        stratify=y,
    )

    print(f"\n  Обучение: {len(X_train)} записей")
    print(f"  Валидация: {len(X_val)} записей")

    base_model = load_base_model(BASE_MODEL_PATH, expected_input_dim=X.shape[1])
    if base_model is None:
        model = create_model(X.shape[1])
        fine_tuning = False
        print("\n  Создана новая модель.")
    else:
        model = base_model
        fine_tuning = True
        print("\n  Загружена совместимая базовая модель, запускаю дообучение.")

    model.summary()
    train_model(model, X_train, y_train, X_val, y_val, fine_tuning=fine_tuning)
    evaluate_model(model, X_val, y_val)
    save_artifacts(model)

    print("\n" + "=" * 72)
    print("ГОТОВО. Модель сохранена в inference-api/models.")
    print("Для проверки API запустите model-training/validate_inference.py")
    print("=" * 72)


if __name__ == "__main__":
    main()
