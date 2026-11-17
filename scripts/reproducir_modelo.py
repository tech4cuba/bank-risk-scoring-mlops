"""Reproduce el modelo final del cuaderno usando el ZIP sintético original.

No reentrena con TEST ni carga código desde el ZIP. Es un prototipo académico.
Ejemplo: python -m scripts.reproducir_modelo --dataset-zip dataset.zip
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.scoring import FEATURE_COLUMNS, validate_model

TARGET = "incumplimiento_90d_12m"
EXCLUDE = (
    "solicitud_id", "cliente_id", "fecha_snapshot", "particion_temporal",
    "cuota_min_tarjeta_estimada_eur", TARGET, "meses_historia_disponibles_6m",
)
MODEL_NAME = "modelo_scoring_crediticio_v1.joblib"
METADATA_NAME = "metadatos_modelo_v1.json"


def _sha256(path: Path) -> str:
    hash_obj = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def reproducir_modelo(dataset_zip: Path, output_dir: Path) -> dict:
    """Entrena solo en TRAIN con parámetros y preprocesado del cuaderno."""
    dataset_zip = Path(dataset_zip)
    output_dir = Path(output_dir)
    with zipfile.ZipFile(dataset_zip) as archive:
        features = pd.read_csv(archive.open("features_solicitud.csv"))
        applications = pd.read_csv(archive.open("solicitudes_prestamos.csv"))

    labels = applications.loc[
        (applications["decision_simulada"] == "APROBADA")
        & (applications["etiqueta_madura"] == True)
        & applications[TARGET].notna(),
        ["solicitud_id", "cliente_id", "particion_temporal", TARGET],
    ].copy()
    dataset = features.merge(
        labels,
        on=["solicitud_id", "cliente_id", "particion_temporal"],
        how="inner", validate="one_to_one",
    )
    if len(dataset) != len(labels):
        raise ValueError("Hay préstamos etiquetados sin características válidas.")
    if not set(EXCLUDE).issubset(dataset.columns):
        raise ValueError("El dataset no contiene todas las columnas esperadas.")

    x = dataset.drop(columns=list(EXCLUDE))
    if tuple(x.columns) != FEATURE_COLUMNS:
        raise ValueError("El orden o las 18 variables difieren del modelo original.")
    y = dataset[TARGET].astype(int)
    train = dataset["particion_temporal"].eq("TRAIN")
    validation = dataset["particion_temporal"].eq("VALIDACION")
    test = dataset["particion_temporal"].eq("TEST_TEMPORAL")
    if train.sum() == 0 or validation.sum() == 0 or test.sum() == 0:
        raise ValueError("Se requieren las tres particiones temporales originales.")
    if not {0, 1}.issubset(set(y.loc[train])):
        raise ValueError("TRAIN debe contener las dos clases.")
    if (train | validation | test).sum() != len(dataset):
        raise ValueError("Se encontraron préstamos etiquetados sin partición conocida.")

    num = x.select_dtypes(include=["number"]).columns.tolist()
    cat = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    model = Pipeline([
        ("preprocesamiento", ColumnTransformer([
            ("numericas", Pipeline([
                ("imputador", SimpleImputer(strategy="median")),
                ("escalador", StandardScaler()),
            ]), num),
            ("categoricas", Pipeline([
                ("imputador", SimpleImputer(strategy="most_frequent")),
                ("codificador", OneHotEncoder(handle_unknown="ignore")),
            ]), cat),
        ])),
        ("clasificador", LogisticRegression(
            C=0.1, penalty="l2", max_iter=2000, random_state=42,
        )),
    ])
    model.fit(x.loc[train], y.loc[train])
    validate_model(model)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_NAME
    joblib.dump(model, model_path, compress=3)
    metadata = {
        "version": "1.0-experimental",
        "provenance": "reentrenado desde el ZIP sintetico con configuracion del cuaderno",
        "modelo": MODEL_NAME,
        "sha256_modelo": _sha256(model_path),
        "sha256_dataset": _sha256(dataset_zip),
        "variables": list(FEATURE_COLUMNS),
        "filas_train": int(train.sum()),
        "filas_validacion": int(validation.sum()),
        "filas_test_reservadas": int(test.sum()),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "restricciones": (
            "Datos simulados. TEST no se utiliza para el ajuste. "
            "No apto para decisiones bancarias reales."
        ),
    }
    (output_dir / METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args(argv)
    metadata = reproducir_modelo(args.dataset_zip, args.output_dir)
    print(json.dumps({
        "resultado": "modelo entrenado",
        "archivo": str(args.output_dir / MODEL_NAME),
        "train": metadata["filas_train"],
        "validacion": metadata["filas_validacion"],
        "test_reservado": metadata["filas_test_reservadas"],
        "sha256": metadata["sha256_modelo"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
