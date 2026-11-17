"""Pruebas del cuaderno de scoring crediticio.

Solo valida la integridad estructural y la funcion de puntuacion.
No vuelve a entrenar el modelo ni reevalua el conjunto de prueba reservado.
"""
import ast
import json
from pathlib import Path
import unittest

import numpy as np

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "02_scoring_crediticio.ipynb"
)


def source(cell):
    value = cell.get("source", [])
    return "".join(value) if isinstance(value, list) else value


class TestCreditScoringNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = cls.notebook["cells"]
        cls.code = [
            c for c in cls.cells if c.get("cell_type") == "code"
        ]
        cls.trees = [ast.parse(source(c)) for c in cls.code]
        cls.code_text = "\n".join(source(c) for c in cls.code)

    def test_notebook_format_and_cells(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreater(len(self.code), 0)
        self.assertGreater(
            len([c for c in self.cells if c.get("cell_type") == "markdown"]),
            0,
        )

    def test_python_cells_compile(self):
        for i, cell in enumerate(self.code):
            with self.subTest(code_cell=i):
                compile(source(cell), f"scoring-cell-{i}", "exec")

    def test_saved_outputs_contain_no_execution_errors(self):
        for i, cell in enumerate(self.code):
            with self.subTest(code_cell=i):
                self.assertFalse(
                    [
                        output for output in cell.get("outputs", [])
                        if output.get("output_type") == "error"
                    ]
                )

    def test_target_and_temporal_partitions_are_explicit(self):
        for expected in (
            "incumplimiento_90d_12m",
            "TRAIN",
            "VALIDACION",
            "TEST_TEMPORAL",
            "X_train",
            "X_val",
            "X_test",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.code_text)

    def test_logistic_regression_parameters(self):
        calls = [
            node
            for tree in self.trees
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "LogisticRegression"
        ]
        self.assertTrue(calls, "Falta la regresion logistica")
        parameters = [
            {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in call.keywords
                if keyword.arg in {"C", "penalty", "random_state"}
            }
            for call in calls
        ]
        self.assertTrue(any(params.get("C") == 0.1 for params in parameters))
        self.assertTrue(
            any(params.get("penalty") == "l2" for params in parameters)
        )

    def test_training_uses_training_partition_not_test(self):
        training_calls = [
            node
            for tree in self.trees
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "fit"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"modelo_scoring", "modelo_final"}
        ]
        self.assertTrue(training_calls)
        for call in training_calls:
            with self.subTest(model=call.func.value.id):
                self.assertIsInstance(call.args[0], ast.Name)
                self.assertTrue(call.args[0].id.startswith("X_train"))

    def test_metrics_are_calculated(self):
        for metric in (
            "roc_auc_score",
            "average_precision_score",
            "brier_score_loss",
            "confusion_matrix",
        ):
            with self.subTest(metric=metric):
                self.assertIn(metric, self.code_text)

    def test_credit_score_is_bounded_and_monotonic(self):
        declarations = []
        scoring_function = None
        for tree in self.trees:
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    if any(
                        isinstance(t, ast.Name)
                        and t.id in {"SCORE_BASE", "PD_BASE", "PDO"}
                        for t in node.targets
                    ):
                        declarations.append(node)
                elif (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "calcular_score"
                ):
                    scoring_function = node

        self.assertIsNotNone(scoring_function)
        module = ast.fix_missing_locations(
            ast.Module(body=[*declarations, scoring_function], type_ignores=[])
        )
        environment = {"np": np}
        exec(compile(module, "credit-score-function", "exec"), environment)
        scores = environment["calcular_score"](
            np.array([0.0, 0.02, 0.10, 0.20, 0.50, 1.0])
        )
        self.assertEqual(scores.dtype.kind, "i")
        self.assertTrue(np.all((scores >= 300) & (scores <= 850)))
        self.assertTrue(np.all(np.diff(scores) <= 0))
        self.assertEqual(int(scores[2]), 600)


if __name__ == "__main__":
    unittest.main()
