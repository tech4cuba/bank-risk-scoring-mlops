"""E2E CI de serialización con datos de juguete; no usa el ZIP bancario ni TEST original."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from scripts.reproducir_modelo import MODEL_NAME, METADATA_NAME, reproducir_modelo
from scripts.probar_modelo_real import probar_modelo
from src.scoring import FEATURE_COLUMNS

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "solicitud_sintetica.json"


class TestReproducibleWorkflow(unittest.TestCase):
    def test_training_persistence_and_fastapi(self):
        rng = np.random.default_rng(2026)
        size = 120
        frame = pd.DataFrame({
            feature: np.arange(size, dtype=float) * (i + 1) / 10.0
            + 10 + i + rng.normal(0, 0.7, size)
            for i, feature in enumerate(FEATURE_COLUMNS)
            if feature != "situacion_laboral"
        })
        frame["situacion_laboral"] = [
            ("ASALARIADO", "AUTONOMO", "TEMPORAL", "PENSIONISTA")[i % 4]
            for i in range(size)
        ]
        frame = frame[list(FEATURE_COLUMNS)]
        frame["solicitud_id"] = [f"SOL{i:06d}" for i in range(size)]
        frame["cliente_id"] = [f"CLI{i:06d}" for i in range(size)]
        frame["particion_temporal"] = [
            "TRAIN" if i < 80 else "VALIDACION" if i < 100 else "TEST_TEMPORAL"
            for i in range(size)
        ]
        frame["fecha_snapshot"] = "2025-01-01"
        frame["meses_historia_disponibles_6m"] = 6
        frame["cuota_min_tarjeta_estimada_eur"] = 0.0
        loans = frame[["solicitud_id", "cliente_id", "particion_temporal"]].copy()
        loans["decision_simulada"] = "APROBADA"
        loans["etiqueta_madura"] = True
        loans["incumplimiento_90d_12m"] = [i % 5 == 0 for i in range(size)]

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            archive = folder / "toy-synthetic.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("features_solicitud.csv", frame.to_csv(index=False))
                z.writestr("solicitudes_prestamos.csv", loans.to_csv(index=False))
            output = folder / "models"
            metadata = reproducir_modelo(archive, output)
            self.assertEqual(metadata["filas_train"], 80)
            self.assertEqual(metadata["filas_validacion"], 20)
            self.assertEqual(metadata["filas_test_reservadas"], 20)
            model_path = output / MODEL_NAME
            result = probar_modelo(model_path, SAMPLE, output / METADATA_NAME)
            self.assertTrue(0 <= result["probabilidad_incumplimiento"] <= 1)
            self.assertTrue(300 <= result["score_crediticio"] <= 850)

            wrong_metadata = dict(metadata, sha256_modelo="0000")
            (output / METADATA_NAME).write_text(json.dumps(wrong_metadata))
            with self.assertRaisesRegex(ValueError, "hash"):
                probar_modelo(model_path, SAMPLE, output / METADATA_NAME)


if __name__ == "__main__":
    unittest.main()
