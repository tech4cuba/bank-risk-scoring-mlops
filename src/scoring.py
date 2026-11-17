"""Inference using the 18-feature model from the credit-scoring notebook.

The resulting score is an illustrative transform, not an official banking rating.
"""
import math
from typing import Mapping

import numpy as np
import pandas as pd

# Exact feature order used to fit modelo_final in the Colab notebook.
FEATURE_COLUMNS = (
    "ingresos_medios_6m_eur",
    "gastos_ordinarios_medios_6m_eur",
    "variabilidad_ingresos_6m",
    "saldo_total_ultimo_mes_eur",
    "limite_credito_total_ultimo_mes_eur",
    "credito_utilizado_ultimo_mes_eur",
    "utilizacion_credito_previa",
    "cuota_deuda_externa_eur",
    "mora_previa_max_dias",
    "dti_con_nueva_cuota",
    "margen_libre_tras_prestamo_eur",
    "accesos_digitales_6m",
    "antiguedad_laboral_meses",
    "situacion_laboral",
    "importe_solicitado_eur",
    "plazo_meses",
    "cuota_propuesta_eur",
    "tasa_nominal_anual_simulada",
)

SCORE_BASE = 600
PD_BASE = 0.10
PDO = 50


class ModelCompatibilityError(ValueError):
    """The artifact is not the expected fitted binary classifier."""


def validate_model(model: object) -> None:
    """Fail closed when a supplied artifact cannot serve this feature contract."""
    actual_features = getattr(model, "feature_names_in_", None)
    if actual_features is None or tuple(actual_features) != FEATURE_COLUMNS:
        raise ModelCompatibilityError("Feature schema differs from notebook model.")
    labels = getattr(model, "classes_", None)
    if labels is None or set(np.asarray(labels).tolist()) != {0, 1}:
        raise ModelCompatibilityError("Expected fitted classifier with classes {0, 1}.")
    if not callable(getattr(model, "predict_proba", None)):
        raise ModelCompatibilityError("Model must support predict_proba.")


def score_from_probability(probability: float) -> int:
    """Match the notebook's rounded and clipped 300..850 transformation."""
    p = float(probability)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("Probability must be a finite number between 0 and 1.")
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    odds_base = (1 - PD_BASE) / PD_BASE
    odds_client = (1 - p) / p
    score = SCORE_BASE + PDO / math.log(2) * math.log(odds_client / odds_base)
    return int(np.clip(np.rint(score), 300, 850))


def predict_credit_risk(model: object, features: Mapping[str, object]) -> dict:
    """Return risk probability and an experimental illustrative credit score."""
    validate_model(model)
    if set(features) != set(FEATURE_COLUMNS):
        raise ValueError("Expected exactly the 18 notebook features.")

    frame = pd.DataFrame([{key: features[key] for key in FEATURE_COLUMNS}])
    probabilities = np.asarray(model.predict_proba(frame))
    labels = np.asarray(model.classes_).tolist()
    if probabilities.shape != (1, 2):
        raise ModelCompatibilityError("Unexpected prediction dimensions.")
    probability = float(probabilities[0, labels.index(1)])
    return {
        "probabilidad_incumplimiento": probability,
        "score_crediticio": score_from_probability(probability),
        "modelo_version": "1.0-experimental",
    }
