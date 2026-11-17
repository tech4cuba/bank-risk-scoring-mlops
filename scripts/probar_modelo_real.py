"""Prueba la API completa con un modelo propio guardado en un archivo .joblib.

Solo se debe cargar un modelo creado por uno mismo o de procedencia fiable.
Ejemplo: python -m scripts.probar_modelo_real --model models/modelo_scoring_crediticio_v1.joblib
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from src.api import create_app
from src.scoring import FEATURE_COLUMNS, score_from_probability, validate_model


def probar_modelo(model_path: Path, sample_path: Path, metadata_path: Path = None) -> dict:
    model_path = Path(model_path).resolve()
    sample_path = Path(sample_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"No se encuentra el modelo: {model_path}")
    if metadata_path is not None:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        expected_sha = metadata.get("sha256_modelo")
        if not expected_sha:
            raise ValueError("Los metadatos carecen del hash del modelo.")
        actual_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError("El hash del modelo no coincide con los metadatos.")
    features = json.loads(sample_path.read_text(encoding="utf-8"))
    if set(features) != set(FEATURE_COLUMNS):
        raise ValueError("El JSON no contiene exactamente las 18 variables.")
    # Carga intencional: exclusivamente un artefacto propio y previamente confiable.
    model = joblib.load(model_path)
    validate_model(model)
    direct = float(model.predict_proba(
        pd.DataFrame([features], columns=list(FEATURE_COLUMNS))
    )[0, 1])
    with patch.dict(os.environ, {"MODEL_PATH": str(model_path)}):
        with TestClient(create_app()) as client:
            assert client.get("/health/live").status_code == 200
            assert client.get("/health/ready").status_code == 200
            response = client.post("/predict", json=features)
            if response.status_code != 200:
                raise RuntimeError(f"Respuesta inesperada: {response.status_code}")
            result = response.json()
    if abs(result["probabilidad_incumplimiento"] - direct) > 1e-12:
        raise AssertionError("La API y el pipeline no producen igual probabilidad.")
    if result["score_crediticio"] != score_from_probability(direct):
        raise AssertionError("El score no coincide con la transformación original.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=Path("examples/solicitud_sintetica.json"))
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args(argv)
    result = probar_modelo(args.model, args.sample, args.metadata)
    print(json.dumps({"estado": "E2E OK", **result}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
