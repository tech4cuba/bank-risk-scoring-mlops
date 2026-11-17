"""Pruebas deterministas del monitor de drift. No utilizan clientes reales."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

from src.monitoring import main, monitor_feature_drift, population_stability_index
from src.scoring import FEATURE_COLUMNS


def synthetic_features(size=200):
    """Muestra sintética de esquema completo con variabilidad reproducible."""
    data = {
        feature: np.linspace(10.0 + i, 80.0 + i, size)
        for i, feature in enumerate(FEATURE_COLUMNS)
        if feature != "situacion_laboral"
    }
    data["situacion_laboral"] = [
        ("ASALARIADO", "AUTONOMO", "TEMPORAL", "PENSIONISTA")[i % 4]
        for i in range(size)
    ]
    return pd.DataFrame(data)[list(FEATURE_COLUMNS)]


class TestDriftMonitoring(unittest.TestCase):
    def setUp(self):
        self.baseline = synthetic_features()
        self.current = self.baseline.copy()

    def test_unchanged_cohort_has_zero_psi_and_no_alerts(self):
        report = monitor_feature_drift(self.baseline, self.current)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["feature_count"], 18)
        self.assertEqual(report["status_counts"]["ok"], 18)
        self.assertTrue(all(x["psi"] == 0 for x in report["features"]))

    def test_numeric_distribution_shift_is_detected(self):
        self.current["gastos_ordinarios_medios_6m_eur"] += 200
        report = monitor_feature_drift(self.baseline, self.current)
        self.assertEqual(report["status"], "critical")
        metric = next(
            x for x in report["features"]
            if x["feature"] == "gastos_ordinarios_medios_6m_eur"
        )
        self.assertGreaterEqual(metric["psi"], 0.25)
        self.assertEqual(metric["status"], "critical")
        self.assertEqual(
            sum(x["status"] != "ok" for x in report["features"]), 1
        )

    def test_new_category_is_reported(self):
        self.current.loc[:100, "situacion_laboral"] = "CATEGORIA_NUEVA"
        report = monitor_feature_drift(self.baseline, self.current)
        metric = next(
            x for x in report["features"]
            if x["feature"] == "situacion_laboral"
        )
        self.assertEqual(metric["new_category_count"], 1)
        self.assertEqual(metric["status"], "critical")

    def test_missingness_change_triggers_alert(self):
        self.current.loc[:49, "mora_previa_max_dias"] = np.nan
        report = monitor_feature_drift(self.baseline, self.current)
        metric = next(
            x for x in report["features"]
            if x["feature"] == "mora_previa_max_dias"
        )
        self.assertAlmostEqual(metric["missing_current_rate"], 0.25)
        self.assertEqual(metric["status"], "critical")

    def test_reference_feature_with_constant_values_detects_shift(self):
        self.baseline["accesos_digitales_6m"] = 5
        self.current["accesos_digitales_6m"] = 30
        report = monitor_feature_drift(self.baseline, self.current)
        metric = next(
            x for x in report["features"]
            if x["feature"] == "accesos_digitales_6m"
        )
        self.assertEqual(metric["status"], "critical")

    def test_small_cohorts_are_marked_insufficient(self):
        report = monitor_feature_drift(
            self.baseline.iloc[:20], self.current.iloc[:20]
        )
        self.assertEqual(report["status"], "insufficient_data")
        self.assertEqual(report["status_counts"]["insufficient_data"], 18)

    def test_missing_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            monitor_feature_drift(
                self.baseline.drop(columns=["situacion_laboral"]),
                self.current,
            )

    def test_invalid_histograms_are_rejected(self):
        with self.assertRaises(ValueError):
            population_stability_index(np.array([0, 0]), np.array([1, 2]))

    def test_csv_cli_creates_only_aggregated_reports_and_exit_code(self):
        self.current["gastos_ordinarios_medios_6m_eur"] += 200
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            baseline_csv = folder / "baseline.csv"
            current_csv = folder / "current.csv"
            output = folder / "reports"
            self.baseline.assign(cliente_id="CLI_TEST_001").to_csv(
                baseline_csv, index=False
            )
            self.current.assign(cliente_id="CLI_TEST_002").to_csv(
                current_csv, index=False
            )
            code = main([
                "--baseline-csv", str(baseline_csv),
                "--current-csv", str(current_csv),
                "--output-dir", str(output),
                "--fail-on-critical",
            ])
            self.assertEqual(code, 2)
            report = json.loads(
                (output / "drift_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "critical")
            self.assertNotIn("cliente_id", json.dumps(report))
            self.assertNotIn("CLI_TEST_001", json.dumps(report))
            self.assertIn(
                "Informe exploratorio",
                (output / "drift_report.md").read_text(encoding="utf-8"),
            )

    def test_dataset_zip_partition_selection(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            zip_path = folder / "synthetic.zip"
            dataset = pd.concat([
                self.baseline.assign(particion_temporal="TRAIN"),
                self.current.assign(particion_temporal="TEST_TEMPORAL"),
            ])
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr(
                    "synthetic/features_solicitud.csv",
                    dataset.to_csv(index=False),
                )
            code = main([
                "--dataset-zip", str(zip_path),
                "--current-partition", "TEST_TEMPORAL",
                "--output-dir", str(folder / "results"),
            ])
            self.assertEqual(code, 0)
            report = json.loads(
                (folder / "results" / "drift_report.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["baseline_count"], 200)
            self.assertEqual(report["current_count"], 200)
            self.assertEqual(report["status"], "ok")


if __name__ == "__main__":
    unittest.main()
