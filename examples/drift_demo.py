"""Demo reproducible de detección de drift. Datos inventados, sin clientes reales.

Ejecutar: python -m examples.drift_demo --output-dir reports/monitoring
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring import monitor_feature_drift, render_markdown
from src.scoring import FEATURE_COLUMNS

EXAMPLE_FILE = Path(__file__).resolve().parent / "solicitud_sintetica.json"


def create_demo_cohorts(size: int = 200):
    """Referencias idénticas salvo desplazamientos deliberados en dos variables."""
    example = json.loads(EXAMPLE_FILE.read_text(encoding="utf-8"))
    data = {}
    for index, feature in enumerate(FEATURE_COLUMNS):
        base = example[feature]
        if feature == "situacion_laboral":
            data[feature] = [
                ("ASALARIADO", "AUTONOMO", "TEMPORAL", "PENSIONISTA")[i % 4]
                for i in range(size)
            ]
        else:
            values = np.linspace(0.90, 1.10, size) * float(base)
            if float(base) == 0:
                values = np.linspace(0, 0.1 * (index + 1), size)
            data[feature] = values
    reference = pd.DataFrame(data)[list(FEATURE_COLUMNS)]
    current = reference.sample(frac=1, random_state=2026).reset_index(drop=True)
    current["gastos_ordinarios_medios_6m_eur"] += 1200
    current["accesos_digitales_6m"] += 40
    return reference, current


def main():
    parser = argparse.ArgumentParser(description="Ejemplo académico de drift.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/monitoring")
    )
    args = parser.parse_args()
    reference, current = create_demo_cohorts()
    report = monitor_feature_drift(reference, current)
    report["example_only"] = True
    report["cohorts"] = {
        "baseline": "DEMO_SINTETICA_REFERENCIA",
        "current": "DEMO_SINTETICA_CON_CAMBIOS",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "drift_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "drift_report.md").write_text(
        "> **DEMOSTRACIÓN SINTÉTICA. No es una monitorización del banco.**\n\n"
        + render_markdown(report),
        encoding="utf-8",
    )
    print("Demo monitoring status:", report["status"])
    print("Affected variables:", [
        m["feature"] for m in report["features"] if m["status"] != "ok"
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
