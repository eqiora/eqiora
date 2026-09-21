"""Check notebook presentation against the single maintained equation source."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("example", Path(__file__).parents[1] / "example.py")
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


class ExampleTest(unittest.TestCase):
    def test_cells_keep_kernel_and_exact_maintained_source(self):
        notebook = EXAMPLE.notebook()
        self.assertEqual(notebook["metadata"]["kernelspec"]["language"], "python")
        source = "".join(notebook["cells"][1]["source"])
        self.assertEqual(source, "%%eqiora model\n" + (EXAMPLE.ROOT / "examples/decay.eqi").read_text())
        self.assertTrue(all(cell["execution_count"] is None and cell["outputs"] == [] for cell in notebook["cells"]))
        self.assertIn("model.digest", "".join(notebook["cells"][2]["source"]))


if __name__ == "__main__":
    unittest.main()
