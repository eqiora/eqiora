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

    def test_missing_portable_compiler_fails_before_download(self):
        with tempfile.TemporaryDirectory() as root:
            with (
                mock.patch.object(install_sundials.importlib.util, "find_spec", return_value=None),
                mock.patch.object(install_sundials.urllib.request, "urlopen") as download,
            ):
                with self.assertRaisesRegex(ValueError, "requires the ziglang package"):
                    install_sundials.install(Path(root) / "install", zig_target="x86_64-linux-gnu.2.17")
                download.assert_not_called()

    def test_portable_compiler_floor_is_bound_to_every_build_command(self):
        import hashlib
        import shlex
        import tarfile
        from types import SimpleNamespace

        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz"):
            pass
        payload = archive.getvalue()
        with tempfile.TemporaryDirectory() as root:
            zig_package = Path(root) / "tools with spaces" / "ziglang"
            with (
                mock.patch.object(install_sundials.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(zig_package / "__init__.py"))),
                mock.patch.object(install_sundials.urllib.request, "urlopen", return_value=io.BytesIO(payload)),
                mock.patch.object(install_sundials, "SHA256", hashlib.sha256(payload).hexdigest()),
                mock.patch.object(install_sundials.subprocess, "run") as execute,
            ):
                install_sundials.install(Path(root) / "install", zig_target="x86_64-linux-gnu.2.17")
            self.assertEqual(len(execute.call_args_list), 3)
            for call in execute.call_args_list:
                self.assertEqual(shlex.split(call.kwargs["env"]["CC"]), [str(zig_package / "zig"), "cc", "-target", "x86_64-linux-gnu.2.17"])
                self.assertTrue(call.kwargs["check"])
