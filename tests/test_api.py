"""Tests for the experimental FastAPI inference service.

A deterministic stand-in is used ONLY inside tests. No real model is inferred.
"""
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from src.api import CreditFeatures, create_app
from src.scoring import (
    FEATURE_COLUMNS,
    ModelCompatibilityError,
    predict_credit_risk,
    score_from_probability,
    validate_model,
)


EXAMPLE = {
    "ingresos_medios_6m_eur": 2958.29,
    "gastos_ordinarios_medios_6m_eur": 1862.16,
    "variabilidad_ingresos_6m": 0.0435,
    "saldo_total_ultimo_mes_eur": 59049.95,
    "limite_credito_total_ultimo_mes_eur": 0.0,
    "credito_utilizado_ultimo_mes_eur": 0.0,
    "utilizacion_credito_previa": 0.0,
    "cuota_deuda_externa_eur": 495.06,
    "mora_previa_max_dias": 0,
    "dti_con_nueva_cuota": 0.3041,
    "margen_libre_tras_prestamo_eur": 196.47,
    "accesos_digitales_6m": 42,
    "antiguedad_laboral_meses": 132,
    "situacion_laboral": "ASALARIADO",
    "importe_solicitado_eur": 16196.71,
    "plazo_meses": 48,
    "cuota_propuesta_eur": 404.60,
    "tasa_nominal_anual_simulada": 0.0920,
}


class FakeFittedModel:
    """Unit-test double. Does not emulate a learned statistical relationship."""

    feature_names_in_ = np.asarray(FEATURE_COLUMNS)
    classes_ = np.asarray([0, 1])

    def predict_proba(self, frame):
        assert list(frame.columns) == list(FEATURE_COLUMNS)
        return np.asarray([[0.8, 0.2]])


class TestInferenceApi(TestCase):
    def test_schema_is_exactly_the_18_notebook_features(self):
        self.assertEqual(set(CreditFeatures.model_fields), set(FEATURE_COLUMNS))
        self.assertEqual(len(FEATURE_COLUMNS), 18)

    def test_liveness_and_readiness_with_test_model(self):
        with TestClient(create_app(model=FakeFittedModel())) as client:
            self.assertEqual(client.get("/health/live").json(), {"status": "alive"})
            self.assertEqual(client.get("/health/ready").json(), {"status": "ready"})

    def test_prediction_uses_model_and_returns_bounded_score(self):
        with TestClient(create_app(model=FakeFittedModel())) as client:
            response = client.post("/predict", json=EXAMPLE)
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertAlmostEqual(result["probabilidad_incumplimiento"], 0.2)
            self.assertEqual(result["score_crediticio"], score_from_probability(0.2))
            self.assertEqual(result["modelo_version"], "1.0-experimental")

    def test_missing_or_unknown_fields_rejected(self):
        with TestClient(create_app(model=FakeFittedModel())) as client:
            incomplete = dict(EXAMPLE)
            del incomplete["plazo_meses"]
            self.assertEqual(client.post("/predict", json=incomplete).status_code, 422)
            unexpected = dict(EXAMPLE, cliente_id="CLI000001")
            self.assertEqual(client.post("/predict", json=unexpected).status_code, 422)

    def test_invalid_category_and_negative_income_rejected(self):
        with TestClient(create_app(model=FakeFittedModel())) as client:
            invalid = dict(EXAMPLE, situacion_laboral="DESCONOCIDO")
            self.assertEqual(client.post("/predict", json=invalid).status_code, 422)
            invalid = dict(EXAMPLE, ingresos_medios_6m_eur=-20)
            self.assertEqual(client.post("/predict", json=invalid).status_code, 422)

    def test_model_is_unavailable_without_artifact(self):
        missing = str(Path("/tmp/modelo_scoring_inexistente_para_ci.joblib"))
        with patch.dict(os.environ, {"MODEL_PATH": missing}):
            with TestClient(create_app()) as client:
                self.assertEqual(client.get("/health/live").status_code, 200)
                self.assertEqual(client.get("/health/ready").status_code, 503)
                self.assertEqual(client.post("/predict", json=EXAMPLE).status_code, 503)

    def test_score_formula_anchor_bounds_and_monotonicity(self):
        self.assertEqual(score_from_probability(0.10), 600)
        values = [score_from_probability(p) for p in [0, 0.01, 0.1, 0.2, 0.9, 1]]
        self.assertTrue(all(300 <= v <= 850 for v in values))
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_incompatible_model_feature_contract_fails(self):
        bad = FakeFittedModel()
        bad.feature_names_in_ = np.asarray(FEATURE_COLUMNS[:-1])
        with self.assertRaises(ModelCompatibilityError):
            validate_model(bad)

    def test_result_matches_direct_inference_helper(self):
        result = predict_credit_risk(FakeFittedModel(), EXAMPLE)
        self.assertEqual(result["score_crediticio"], score_from_probability(0.2))
