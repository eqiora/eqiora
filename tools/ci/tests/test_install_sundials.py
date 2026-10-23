"""The native build must not execute unverified archive contents."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import install_sundials


class SundialsInstallTests(unittest.TestCase):
    def test_corrupt_archive_is_rejected_before_extract_or_execution(self):
        with tempfile.TemporaryDirectory() as root:
            prefix = Path(root) / "install"
            with (
                mock.patch.object(install_sundials.urllib.request, "urlopen", return_value=io.BytesIO(b"corrupt archive")),
                mock.patch.object(install_sundials.tarfile, "open") as extract,
                mock.patch.object(install_sundials.subprocess, "run") as execute,
            ):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    install_sundials.install(prefix)
                extract.assert_not_called()
                execute.assert_not_called()
            self.assertFalse(prefix.exists())

    def test_existing_installation_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            prefix = Path(root)
            sentinel = prefix / "keep"
            sentinel.write_text("existing installation")
            with mock.patch.object(install_sundials.urllib.request, "urlopen") as download:
                with self.assertRaisesRegex(ValueError, "must not already exist"):
                    install_sundials.install(prefix)
                download.assert_not_called()
            self.assertEqual(sentinel.read_text(), "existing installation")
