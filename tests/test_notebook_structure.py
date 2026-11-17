"""Comprobaciones estáticas del cuaderno original de generación de datos.

Estas pruebas no ejecutan la simulación ni validan la calidad estadística del dataset.
"""
import ast
import json
from pathlib import Path
import unittest

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "01_generacion_dataset.ipynb"
)


def cell_source(cell):
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


class TestDatasetNotebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = cls.notebook["cells"]
        cls.code_cells = [c for c in cls.cells if c["cell_type"] == "code"]
        cls.markdown_cells = [
            c for c in cls.cells if c["cell_type"] == "markdown"
        ]

    def test_notebook_is_well_formed(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertGreater(len(self.code_cells), 0)
        self.assertGreater(len(self.markdown_cells), 0)
        for cell in self.cells:
            self.assertIn(cell["cell_type"], ("code", "markdown", "raw"))
            self.assertIsInstance(cell_source(cell), str)

    def test_all_python_cells_have_valid_syntax(self):
        for index, cell in enumerate(self.cells):
            if cell["cell_type"] == "code":
                with self.subTest(cell_index=index):
                    compile(
                        cell_source(cell),
                        f"notebook-cell-{index}",
                        "exec",
                    )

    def test_no_saved_execution_errors(self):
        for index, cell in enumerate(self.code_cells):
            with self.subTest(code_cell=index):
                errors = [
                    o for o in cell.get("outputs", [])
                    if o.get("output_type") == "error"
                ]
                self.assertFalse(errors, "El cuaderno guarda errores de ejecución")

    def test_reproducibility_and_credit_label_are_documented(self):
        code = "\n".join(map(cell_source, self.code_cells))
        self.assertIn("SEMILLA = 2026", code)
        self.assertIn("default_rng(SEMILLA)", code)
        self.assertIn("incumplimiento_90d_12m", code)
        self.assertIn("particion_temporal", code)

    def test_registry_contains_17_tables(self):
        registry = None
        for cell in self.code_cells:
            tree = ast.parse(cell_source(cell))
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "TABLAS"
                    for target in node.targets
                ):
                    registry = node.value

        self.assertIsInstance(registry, ast.Dict)
        names = [ast.literal_eval(key) for key in registry.keys]
        self.assertEqual(len(names), 17)
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("solicitudes_prestamos", names)
        self.assertIn("features_solicitud", names)


if __name__ == "__main__":
    unittest.main()
