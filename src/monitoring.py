"""Monitorización exploratoria de drift de características de scoring crediticio.

Genera métricas agregadas, nunca registros individuales. El PSI es descriptivo
y los umbrales son convenciones de demostración, no reglas bancarias validadas.
La degradación predictiva requiere etiquetas maduras y predicciones históricas.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from src.scoring import FEATURE_COLUMNS

CATEGORICAL_COLUMNS = frozenset({"situacion_laboral"})
DEFAULT_MIN_RECORDS = 30
PSI_WARNING = 0.10
PSI_CRITICAL = 0.25
MISSING_WARNING = 0.05
MISSING_CRITICAL = 0.15

STATUS_RANK = {"ok": 0, "warning": 1, "critical": 2}


def _numeric_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values[~np.isfinite(values)] = np.nan
    return values


def _numeric_counts(
    reference: pd.Series, current: pd.Series, bins: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    ref_values = _numeric_values(reference)
    cur_values = _numeric_values(current)
    ref_present = ref_values[np.isfinite(ref_values)]
    cur_present = cur_values[np.isfinite(cur_values)]

    if ref_present.size == 0:
        cutoffs = np.array([], dtype=float)
    elif np.min(ref_present) == np.max(ref_present):
        value = float(ref_present[0])
        tolerance = max(1e-9, abs(value) * 1e-9)
        cutoffs = np.array([value - tolerance, value + tolerance])
    else:
        percentiles = np.quantile(
            ref_present, np.linspace(0.0, 1.0, bins + 1)
        )
        cutoffs = np.unique(percentiles[1:-1])

    n_intervals = len(cutoffs) + 1
    ref_ids = np.digitize(ref_present, cutoffs)
    cur_ids = np.digitize(cur_present, cutoffs)
    reference_counts = np.bincount(ref_ids, minlength=n_intervals)
    current_counts = np.bincount(cur_ids, minlength=n_intervals)

    reference_counts = np.append(
        reference_counts, len(ref_values) - len(ref_present)
    )
    current_counts = np.append(
        current_counts, len(cur_values) - len(cur_present)
    )
    return (
        reference_counts,
        current_counts,
        1 - len(ref_present) / len(ref_values),
        1 - len(cur_present) / len(cur_values),
    )


def _categorical_counts(
    reference: pd.Series, current: pd.Series
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    missing_token = "\x00__MISSING__"
    ref = reference.astype("string").replace("", pd.NA).fillna(missing_token)
    cur = current.astype("string").replace("", pd.NA).fillna(missing_token)
    reference_categories = set(ref.unique().tolist())
    current_categories = set(cur.unique().tolist())
    categories = sorted(reference_categories | current_categories)
    reference_counts = ref.value_counts()
    current_counts = cur.value_counts()
    unknown = len((current_categories - reference_categories) - {missing_token})
    return (
        np.array([reference_counts.get(c, 0) for c in categories], dtype=float),
        np.array([current_counts.get(c, 0) for c in categories], dtype=float),
        float((ref == missing_token).mean()),
        float((cur == missing_token).mean()),
        unknown,
    )


def population_stability_index(
    baseline_counts: np.ndarray, current_counts: np.ndarray
) -> float:
    """PSI for aligned histograms with small additive probability smoothing."""
    baseline_counts = np.asarray(baseline_counts, dtype=float)
    current_counts = np.asarray(current_counts, dtype=float)
    if (
        baseline_counts.ndim != 1
        or baseline_counts.shape != current_counts.shape
        or baseline_counts.size == 0
        or np.any(baseline_counts < 0)
        or np.any(current_counts < 0)
        or baseline_counts.sum() <= 0
        or current_counts.sum() <= 0
    ):
        raise ValueError("Expected aligned, nonempty, nonnegative histograms.")
    smooth = 1e-4
    base = (baseline_counts + smooth) / (
        baseline_counts.sum() + smooth * baseline_counts.size
    )
    new = (current_counts + smooth) / (
        current_counts.sum() + smooth * current_counts.size
    )
    return float(np.sum((new - base) * np.log(new / base)))


def monitor_feature_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    *,
    min_records: int = DEFAULT_MIN_RECORDS,
    bins: int = 10,
) -> dict:
    """Compare input distributions; labels and customer IDs are never required."""
    if min_records < 1 or bins < 2:
        raise ValueError("min_records must be >= 1 and bins must be >= 2.")
    if baseline.empty or current.empty:
        raise ValueError("Both input cohorts must contain observations.")

    missing = [
        name for name in FEATURE_COLUMNS
        if name not in baseline.columns or name not in current.columns
    ]
    if missing:
        raise ValueError("Missing required scoring features: " + ", ".join(missing))

    insufficient = min(len(baseline), len(current)) < min_records
    metrics = []
    for feature in FEATURE_COLUMNS:
        if feature in CATEGORICAL_COLUMNS:
            b_counts, c_counts, b_miss, c_miss, unknown = _categorical_counts(
                baseline[feature], current[feature]
            )
            variable_type = "categorical"
        else:
            b_counts, c_counts, b_miss, c_miss = _numeric_counts(
                baseline[feature], current[feature], bins
            )
            unknown = 0
            variable_type = "numeric"

        psi = population_stability_index(b_counts, c_counts)
        missing_delta = abs(c_miss - b_miss)
        if insufficient:
            status = "insufficient_data"
        elif psi >= PSI_CRITICAL or missing_delta >= MISSING_CRITICAL:
            status = "critical"
        elif psi >= PSI_WARNING or missing_delta >= MISSING_WARNING:
            status = "warning"
        else:
            status = "ok"
        metrics.append({
            "feature": feature,
            "kind": variable_type,
            "psi": round(psi, 6),
            "missing_baseline_rate": round(b_miss, 6),
            "missing_current_rate": round(c_miss, 6),
            "missing_rate_gap": round(missing_delta, 6),
            "new_category_count": unknown,
            "status": status,
        })

    worst = max(
        (STATUS_RANK[m["status"]] for m in metrics if m["status"] in STATUS_RANK),
        default=0,
    )
    status = (
        "insufficient_data"
        if insufficient
        else next(s for s, rank in STATUS_RANK.items() if rank == worst)
    )
    counts = {
        category: sum(m["status"] == category for m in metrics)
        for category in ("ok", "warning", "critical", "insufficient_data")
    }
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "baseline_count": int(len(baseline)),
        "current_count": int(len(current)),
        "feature_count": len(metrics),
        "status_counts": counts,
        "configuration": {
            "min_records": min_records,
            "numeric_quantile_bins": bins,
            "psi_warning": PSI_WARNING,
            "psi_critical": PSI_CRITICAL,
            "missing_rate_warning": MISSING_WARNING,
            "missing_rate_critical": MISSING_CRITICAL,
        },
        "features": metrics,
        "interpretation": (
            "Exploratory covariate drift only; PSI cutoffs are heuristic. "
            "This does not establish model-performance degradation, "
            "probability miscalibration or the need to retrain. "
            "Only aggregate figures are exported."
        ),
    }


def render_markdown(report: dict) -> str:
    """A plain-language report suitable for a GitHub Actions job summary."""
    lines = [
        "# Informe exploratorio de drift del scoring",
        "",
        f"**Estado:** {report['status']}",
        f"**Observaciones de referencia:** {report['baseline_count']}",
        f"**Observaciones recientes:** {report['current_count']}",
        "",
        "| Variable | PSI | Faltantes referencia | Faltantes recientes | Estado |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for entry in report["features"]:
        name = entry["feature"]
        lines.append(
            f"| {name} | {entry['psi']:.3f} | "
            f"{entry['missing_baseline_rate']:.1%} | "
            f"{entry['missing_current_rate']:.1%} | {entry['status']} |"
        )
    lines += [
        "",
        "**Interpretación:** El PSI evalúa cambios en distribuciones, no "
        "la precisión del modelo. Umbrales descriptivos: aviso desde 0,10 "
        "y crítico desde 0,25. Las cohortes pequeñas y las categorías "
        "raras pueden generar alertas inestables.",
        "",
        "**Limitaciones:** La degradación predictiva solo podrá evaluarse "
        "cuando haya predicciones históricas y etiquetas maduras; "
        "la monitorización no utiliza datos personales ni resultados de impago.",
        "",
    ]
    return "\n".join(lines)


def _read_features_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        candidates = [
            info for info in zf.infolist()
            if not info.is_dir() and Path(info.filename).name == "features_solicitud.csv"
        ]
        if len(candidates) != 1:
            raise ValueError(
                "ZIP must contain exactly one features_solicitud.csv."
            )
        with zf.open(candidates[0]) as handle:
            return pd.read_csv(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Monitor de cambios en las 18 variables del scoring."
    )
    parser.add_argument("--dataset-zip", type=Path)
    parser.add_argument("--baseline-csv", type=Path)
    parser.add_argument("--current-csv", type=Path)
    parser.add_argument("--baseline-partition", default="TRAIN")
    parser.add_argument(
        "--current-partition", default="MONITORIZACION_SIN_ETIQUETA_MADURA"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/monitoring"))
    parser.add_argument("--min-records", type=int, default=DEFAULT_MIN_RECORDS)
    parser.add_argument("--fail-on-critical", action="store_true")
    args = parser.parse_args(argv)

    if args.dataset_zip:
        if args.baseline_csv or args.current_csv:
            parser.error("--dataset-zip cannot be combined with separate CSV files.")
        features = _read_features_from_zip(args.dataset_zip)
        if "particion_temporal" not in features.columns:
            parser.error("Dataset features must contain particion_temporal.")
        baseline = features.loc[
            features["particion_temporal"] == args.baseline_partition
        ]
        current = features.loc[
            features["particion_temporal"] == args.current_partition
        ]
    elif args.baseline_csv and args.current_csv:
        baseline = pd.read_csv(args.baseline_csv)
        current = pd.read_csv(args.current_csv)
    else:
        parser.error("Specify a ZIP or both --baseline-csv and --current-csv.")

    report = monitor_feature_drift(
        baseline, current, min_records=args.min_records
    )
    report["cohorts"] = {
        "baseline": (
            args.baseline_partition if args.dataset_zip else "supplied CSV"
        ),
        "current": (
            args.current_partition if args.dataset_zip else "supplied CSV"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "drift_report.json"
    markdown_path = args.output_dir / "drift_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"Drift status: {report['status']}. "
        f"Counts: {report['status_counts']}. "
        f"Reports: {json_path}, {markdown_path}"
    )
    return 2 if args.fail_on_critical and report["status"] == "critical" else 0


if __name__ == "__main__":
    raise SystemExit(main())
