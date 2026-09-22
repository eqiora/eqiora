from pathlib import Path
import tempfile
import unittest

from tools.editor.check_jupyter import verify_assets


class JupyterAssetTests(unittest.TestCase):
    def test_only_a_complete_identical_distribution_passes(self):
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            expected, generated = (
                Path(temporary) / "expected",
                Path(temporary) / "generated",
            )
            expected.mkdir()
            generated.mkdir()
            with self.assertRaises(ValueError):
                verify_assets(expected, generated)
            for root in (expected, generated):
                (root / "entry.js").write_bytes(b"const source = 'current';")
            verify_assets(expected, generated)
            (generated / "entry.js").write_bytes(b"const source = 'stale';")
            with self.assertRaisesRegex(ValueError, "entry.js"):
                verify_assets(expected, generated)
            (generated / "entry.js").write_bytes((expected / "entry.js").read_bytes())
            (generated / "extra.js").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "extra.js"):
                verify_assets(expected, generated)
            (generated / "extra.js").unlink()
            (generated / "entry.js").unlink()
            with self.assertRaisesRegex(ValueError, "entry.js"):
                verify_assets(expected, generated)


if __name__ == "__main__":
    unittest.main()
